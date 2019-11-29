import os
# from torchnlp.word_to_vector import FastText
import torch
from torch.utils.tensorboard import SummaryWriter
from torchnlp.encoders.text import stack_and_pad_tensors

from model import HAN, HGRULWAN, HCapsNet, HCapsNetMultiHeadAtt
from data_utils.data_utils import get_embedding, _convert_word_to_idx
from utils.radam import RAdam
from utils.logger import get_logger, Progbar
from utils.metrics import *
from document_model import Document
import fasttext
from torchnlp.word_to_vector.pretrained_word_vectors import _PretrainedWordVectors
# from flair.embeddings import WordEmbeddings

try:
	from apex import amp
	APEX_AVAILABLE = True
except ModuleNotFoundError:
	APEX_AVAILABLE = False


class FastText(_PretrainedWordVectors):
	""" Enriched word vectors with subword information from Facebook's AI Research (FAIR) lab.
	A approach based on the skipgram model, where each word is represented as a bag of character
	n-grams. A vector representation is associated to each character n-gram; words being
	represented as the sum of these representations.
	References:
		* https://arxiv.org/abs/1607.04606
		* https://fasttext.cc/
		* https://arxiv.org/abs/1710.04087
	Args:
		language (str): language of the vectors
		aligned (bool): if True: use multilingual embeddings where words with
			the same meaning share (approximately) the same position in the
			vector space across languages. if False: use regular FastText
			embeddings. All available languages can be found under
			https://github.com/facebookresearch/MUSE#multilingual-word-embeddings
		cache (str, optional): directory for cached vectors
		unk_init (callback, optional): by default, initialize out-of-vocabulary word vectors
			to zero vectors; can be any function that takes in a Tensor and
			returns a Tensor of the same size
		is_include (callable, optional): callable returns True if to include a token in memory
			vectors cache; some of these embedding files are gigantic so filtering it can cut
			down on the memory usage. We do not cache on disk if ``is_include`` is defined.
	Example:
		>>> from torchnlp.word_to_vector import FastText  # doctest: +SKIP
		>>> vectors = FastText()  # doctest: +SKIP
		>>> vectors['hello']  # doctest: +SKIP
		-0.1595
		-0.1826
		...
		0.2492
		0.0654
		[torch.FloatTensor of size 300]
	"""
	url_base = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{}.vec'
	aligned_url_base = 'https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.{}.align.vec'

	def __init__(self, language="en", aligned=False, **kwargs):
		if aligned:
			url = self.aligned_url_base.format(language)
		else:
			url = self.url_base.format(language)
		name = kwargs.get('name', None) #os.path.basename(url)
		super(FastText, self).__init__(name) #, url=url, **kwargs)


class MultiLabelTextClassifier:

	def __init__(self, model_name, word_to_idx, label_to_idx, label_map, min_freq_word = 100,
				 tensorboard_dir = 'runs', B_train = 16, B_eval = 32, weight_decay = 1e-4, lr = 1e-3,
				 dropout = 0.1, K=5, verbose=True, **kwargs):

		self.model_name = model_name
		self.use_doc_encoding = 'caps' in model_name.lower()
		self.model = None
		self.word_to_idx = word_to_idx
		self.label_to_idx = label_to_idx # Maps label_ids to index in model
		if label_map:
			self.label_map = label_map
		else:
			self.label_map = {v:k for k,v in label_to_idx.items()}# Maps label ids to EUROVOC description
		self.log_path = kwargs.get('log_path', 'log.txt')
		self.save_dir = kwargs.get('save_dir', None) #TODO: use save_dir
		self.tensorboard_dir = tensorboard_dir
		self.B_train = B_train
		self.B_eval = B_eval
		self.lr = lr
		self.weight_decay = weight_decay
		self.dropout = dropout
		self.min_freq_word = min_freq_word
		self.K = K
		self.verbose = verbose

		# Placeholders for attributes to be initialized
		# TODO: use kwarg arguments downstream  --> or just keep for load method?
		self.embed_size = kwargs.get('embed_size', None)
		self.word_hidden = kwargs.get('word_hidden', None)
		self.sent_hidden = kwargs.get('sent_hidden', None)
		self.nhead_doc = kwargs.get('nhead_doc', None)
		self.word_encoder = kwargs.get('word_encoder', None)
		self.sent_encoder = kwargs.get('sent_encoder', None)
		self.pretrained_path = kwargs.get('pretrained_path', None)
		self.ulmfit_pretrained_path = kwargs.get('ulmfit_pretrained_path', None)
		self.optimizer = kwargs.get('optimizer', None)
		self.criterion = kwargs.get('criterion', None)

		# Other attributes
		self.vocab_size = len(word_to_idx)
		self.num_labels = len(label_to_idx)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.logger = get_logger(self.log_path)
		self.writer = SummaryWriter(log_dir= tensorboard_dir,
									comment='Model={}-Labels={}-B={}-L2={}-Dropout={}'.format(model_name, self.num_labels, B_train, weight_decay, dropout))

	def save(self, path):
		self.pretrained_path = path

		params = self.model.get_init_params()
		params["state_dict"] = self.model.state_dict()
		params["word_to_idx"] = self.word_to_idx
		params["label_to_idx"] = self.label_to_idx
		params["label_map"] = self.label_map
		params["criterion"] = self.criterion

		torch.save(params, path)


	@classmethod
	def load(cls, path):
		params = torch.load(path)
		params_no_weight = {k:v for k,v in params.items() if k != 'state_dict'}

		self = cls(**params_no_weight)

		num_tokens = len(params_no_weight['word_to_idx'])
		num_classes = len(params_no_weight['label_to_idx'])

		if params['model_name'].lower() == 'han':
			model = HAN(num_tokens = num_tokens, num_classes = num_classes, **params_no_weight)
		elif params['model_name'].lower() == 'hgrulwan':
			model = HGRULWAN(num_tokens = num_tokens, num_classes = num_classes, **params_no_weight)
		elif params['model_name'].lower() == 'hcapsnet':
			model = HCapsNet(num_tokens = num_tokens, num_classes = num_classes, **params_no_weight)
		elif params['model_name'].lower() == 'hcapsnetmultiheadatt':
			model = HCapsNetMultiHeadAtt(num_tokens = num_tokens, num_classes = num_classes, **params_no_weight)

		model.load_state_dict(params['state_dict'])
		model.to(self.device)
		self.model = model

		return self

	def init_model(self, embed_dim, word_hidden, sent_hidden, dropout, vector_path, use_glove, word_encoder = 'gru', sent_encoder = 'gru',
				   dim_caps=16, num_caps = 25, num_compressed_caps = 100, dropout_caps = 0.2, lambda_reg_caps = 0.0005, pos_weight=None, nhead_doc=5,
				   ulmfit_pretrained_path = None, dropout_factor_ulmfit = 1.0):

		self.embed_size = embed_dim
		self.word_hidden = word_hidden
		self.sent_hidden = sent_hidden
		self.dropout = dropout
		self.word_encoder = word_encoder
		self.sent_encoder = sent_encoder
		self.ulmfit_pretrained_path = ulmfit_pretrained_path

		# Initialize model and load pretrained weights if given
		self.logger.info("Building model...")
		if self.model_name.lower() == 'han':
			self.model = HAN(self.vocab_size, embed_dim, word_hidden, sent_hidden, self.num_labels, dropout=dropout,
							 word_encoder = word_encoder, sent_encoder = sent_encoder, ulmfit_pretrained_path=ulmfit_pretrained_path,
							 dropout_factor_ulmfit=dropout_factor_ulmfit) #TODO: also adapt for other models
		elif self.model_name.lower() == 'hgrulwan':
			self.model = HGRULWAN(self.vocab_size, embed_dim, word_hidden, sent_hidden, self.num_labels, dropout=dropout, word_encoder=word_encoder,
								  ulmfit_pretrained_path=ulmfit_pretrained_path,dropout_factor_ulmfit=dropout_factor_ulmfit)
		elif self.model_name.lower() == 'hcapsnet':
			self.model = HCapsNet(self.vocab_size, embed_dim, word_hidden, sent_hidden, self.num_labels, dropout=dropout,
							 		word_encoder = word_encoder, sent_encoder = sent_encoder, dropout_caps = dropout_caps,
									dim_caps=dim_caps, num_caps=num_caps, num_compressed_caps=num_compressed_caps,
								  	ulmfit_pretrained_path=ulmfit_pretrained_path, dropout_factor_ulmfit=dropout_factor_ulmfit,
								  	lambda_reg_caps = lambda_reg_caps)
		elif self.model_name.lower() == 'hcapsnetmultiheadatt':
			self.model = HCapsNetMultiHeadAtt(self.vocab_size, embed_dim, word_hidden, sent_hidden, self.num_labels, dropout=dropout,
							 		word_encoder = word_encoder, sent_encoder = sent_encoder, dropout_caps = dropout_caps,
									dim_caps=dim_caps, num_caps=num_caps, num_compressed_caps=num_compressed_caps, nhead_doc=nhead_doc,
									ulmfit_pretrained_path=ulmfit_pretrained_path,dropout_factor_ulmfit=dropout_factor_ulmfit,
									lambda_reg_caps = lambda_reg_caps)

		# Initialize training attributes
		if 'caps' in self.model_name.lower():
			self.criterion = torch.nn.BCELoss()
		else:
			if pos_weight:
				pos_weight = torch.tensor(pos_weight).to(self.device)
			self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')

		self.optimizer = RAdam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

		# Load embeddings
		if word_encoder.lower() != 'ulmfit':
			vectors = fasttext.load_model(vector_path)

			glove = WordEmbeddings('glove') if use_glove else None # Extra GloVe embeddings
			embed_table = get_embedding(vectors, self.word_to_idx, embed_dim, glove=glove)
			self.model.set_embedding(embed_table)

		self.model.to(self.device)


	def pred_to_labels(self, pred, top=5):
		# Converts prediction for single doc to original label names
		if isinstance(pred, torch.Tensor):
			pred = pred.cpu().numpy()
			
		pred = pred[0]

		ind = np.argpartition(pred, -top)[-top:]
		highest_scoring = ind[np.argsort(pred[ind])]

		idx_to_label = {v:k for k,v in self.label_to_idx.items()}

		label_ids = [idx_to_label[p] for p in highest_scoring]
		label_descriptions = [self.label_map[ix]['label'] for ix in label_ids]
		return label_descriptions

	def predict_doc(self, doc):

		assert self.model is not None, "Model needs to be initialized before inference can be done."
		self.model.eval()
		# convert doc to tensors
		sample = {}
		# collect tags
		tags = [int(self.label_to_idx[tag]) for tag in doc.tags if tag in self.label_to_idx]
		# convert sentences to indices of words
		sents = [torch.LongTensor([_convert_word_to_idx(w, self.word_to_idx, None, None) for w in sent]) for
				 sent in doc.sentences]

		# convert to tensors
		sample['tags'] = np.zeros(len(self.label_to_idx))
		sample['tags'][tags] = 1
		sample['tags'] = torch.FloatTensor(sample['tags']).to(self.device)  # One Hot Encoded target
		sents, sents_len = stack_and_pad_tensors(sents)
		sample['sents'] = sents.to(self.device)   # , _ =
		sample['sents_len'] = sents_len.to(self.device)
		sample['doc_len'] = [len(sents)]

		# For usage of Pytorch RNN
		transpose = (lambda b: b.t_().squeeze(0).contiguous())

		# Get predictions
		with torch.no_grad():
			if self.word_encoder.lower() == 'gru':
				sample['sents'] = transpose(sample['sents'])

			preds, word_attention_scores, sent_attention_scores = self.model(sample['sents'], sample['sents_len'],
																	sample['doc_len'])

		# convert to lists
		preds = list(preds.cpu().numpy())
		if self.word_encoder.lower() == 'gru':
			word_attention_scores = transpose(word_attention_scores)
		word_attention_scores = word_attention_scores.squeeze().cpu().numpy().tolist()
		sent_attention_scores = sent_attention_scores.cpu().numpy().tolist()
		# Flatten list
		sent_attention_scores = [l[0] for sublist in sent_attention_scores for l in sublist]

		# Filter predictions for padding
		sents_len = sents_len.cpu().numpy()
		if len(sents_len) > 1:
			word_attention_scores = [score[:l] for l,score in zip(sents_len, word_attention_scores)]
		else:
			word_attention_scores = [word_attention_scores]

		# word_attention_scores = [[s*100 for s in sen] for sen in word_attention_scores]
		return preds, word_attention_scores, sent_attention_scores

	def predict_text(self, text, return_doc=False):
		# convert text to Document
		sentences = [text]
		doc = Document('', [], sentences=sentences, discard_short_sents=False, split_size_long_seqs=50)
		# predict as doc
		if return_doc:
			p, w, s = self.predict_doc(doc)
			return p, w, s, doc
		else:
			return self.predict_doc(doc)



	def train(self, dataloader_train, dataloader_dev, pos_weight, num_epochs, eval_every):

		# Train epoch
		best_score, best_loss, train_step = (0,0,0)

		to_freeze = 3 # total nr of layers to freeze in ULMFiT

		for epoch in range(num_epochs):
			torch.cuda.empty_cache()
			self.logger.info("Epoch: {}".format(epoch))
			if (self.word_encoder.lower() == 'ulmfit') and (epoch <= to_freeze):
				self.model.sent_encoder.word_encoder.freeze_to(epoch)

			# continue
			best_score, best_loss, train_step = self._train_epoch(dataloader_train, dataloader_dev, self.optimizer,
																  self.criterion, eval_every, train_step, best_score,
																  best_loss)

	def _train_epoch(self, dataloader_train, dataloader_dev, optimizer, criterion, eval_every, train_step, best_score, best_loss):

		prog = Progbar(len(dataloader_train))

		tr_loss = 0

		for batch_idx, batch in enumerate(dataloader_train):
			train_step += 1
			optimizer.zero_grad()

			(sents, sents_len, doc_lens, target, doc_encoding) = batch
			if self.use_doc_encoding: # Capsule based models
				preds, word_attention_scores, sent_attention_scores, rec_loss = self.model(sents, sents_len, doc_lens, doc_encoding)
			else: # Other models
				preds, word_attention_scores, sent_attention_scores, rec_loss = self.model(sents, sents_len, doc_lens) # rec loss defaults to 0 for non-CapsNet models

			loss = criterion(preds, target)
			loss += rec_loss
			tr_loss += loss.item()

			if APEX_AVAILABLE:
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
			else:
				loss.backward()

			# torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
			optimizer.step()
			prog.update(batch_idx + 1, values=[("train loss", loss.item()), ("recon loss", rec_loss)])
			torch.cuda.empty_cache()

			if train_step % eval_every == 0:
				best_score, best_loss = self._eval_model(dataloader_train, dataloader_dev, best_score, best_loss, train_step)

		return best_score, best_loss, train_step


	def _eval_model(self, dataloader_train, dataloader_dev, best_score, best_loss, train_step):
		# Eval dev
		r_k_dev, p_k_dev, rp_k_dev, ndcg_k_dev, avg_loss_dev,  \
			hamming_dev, emr_dev, f1_micro_dev, f1_macro_dev = self.eval_dataset(dataloader_dev, K=self.K)

		# Eval Train
		r_k_tr, p_k_tr, rp_k_tr, ndcg_k_tr, avg_loss_tr,  hamming_tr, \
			emr_tr, f1_micro_tr, f1_macro_tr = self.eval_dataset(dataloader_train, K=self.K,
																	max_samples=len(dataloader_dev))

		# Save model if best
		if best_score < f1_micro_dev:
			best_score = f1_micro_dev

			self.save(os.path.join(self.save_dir, self.model_name + '_loss={0:.5f}_RP{1}={2:.3f}.pt'.format(avg_loss_dev,self.K, rp_k_dev)))
			# torch.save(self.model.state_dict(),
			# 		   os.path.join(self.save_dir, self.model_name + '_loss={0:.5f}_RP{1}={2:.3f}.pt'.format(avg_loss_dev, self.K, rp_k_dev)))
			self.logger.info("Saved model with new best score: {0:.3f}".format(rp_k_dev))
		# elif best_loss > avg_loss_dev:
		# 	best_loss = avg_loss_dev
		#
		# 	self.save(os.path.join(self.save_dir, self.model_name + '_loss={0:.5f}_RP{1}={2:.3f}.pt'.format(avg_loss_dev,self.K, rp_k_dev)))
		#
		# 	self.logger.info("Saved model with new best loss: {0:.5f}".format(avg_loss_dev))

		# Write to Tensorboard
		self.writer.add_scalars("Loss",
						   {"Train": avg_loss_tr,
							"Dev": avg_loss_dev}, train_step)
		self.writer.add_scalars("R_{}".format(self.K),
						   {"Train": r_k_tr,
							"Dev": r_k_dev},
						   train_step)
		self.writer.add_scalars("P_{}".format(self.K),
						   {"Train": p_k_tr,
							"Dev": p_k_dev},
						   train_step)
		self.writer.add_scalars("RP_{}".format(self.K),
						   {"Train": rp_k_tr,
							"Dev": rp_k_dev},
						   train_step)
		self.writer.add_scalars("NDCG_{}".format(self.K),
						   {"Train": ndcg_k_tr,
							"Dev": ndcg_k_dev},
						   train_step)

		# Return to training mode
		self.model.train()

		return best_score, best_loss

	def eval_dataset(self, dataloader, K=5, max_samples=None):
		self.logger.info("Evaluating model")
		self.model.eval()
		y_pred = []
		y_true = []
		eval_loss = 0

		# Get all predictions on dataset
		with torch.no_grad():
			for batch_idx, batch in enumerate(dataloader):

				(sents, sents_len, doc_lens, target, doc_encoding) = batch

				if self.use_doc_encoding:  # Capsule based models
					preds = self.model(sents, sents_len, doc_lens, doc_encoding)[0]
				else:
					preds = self.model(sents, sents_len, doc_lens)[0]
				loss = self.criterion(preds, target)
				eval_loss += loss.item()
				# store predictions and targets
				y_pred.extend(list(preds.cpu().detach().numpy()))
				y_true.extend(list(target.cpu().detach().numpy()))

				if max_samples:
					if batch_idx >= max_samples:
						break

		avg_loss = eval_loss / len(dataloader)

		hamming, emr, f1_micro, f1_macro = accuracy(y_true, y_pred, False)

		self.logger.info("Hamming loss {:1.3f} | Exact Match Ratio {:1.3f} | Micro F1 {:1.3f} | Macro F1 {:1.3f}".format(hamming, emr, f1_micro, f1_macro))
		template = 'F1@{0} : {1:1.3f} R@{0} : {2:1.3f}   P@{0} : {3:1.3f}   RP@{0} : {4:1.3f}   NDCG@{0} : {5:1.3f}'

		for i in range(1, K + 1):
			r_k = mean_recall_k(y_true,
								y_pred, k=i)
			p_k = mean_precision_k(y_true,
								   y_pred, k=i)
			rp_k = mean_rprecision_k(y_true,
									 y_pred, k=i)
			ndcg_k = mean_ndcg_score(y_true,
									 y_pred, k=i)

			f1_k = (2*r_k*p_k)/(r_k+p_k)
			self.logger.info(template.format(i, f1_k, r_k, p_k, rp_k, ndcg_k))
		self.logger.info('----------------------------------------------------')

		return r_k, p_k, rp_k, ndcg_k, avg_loss, hamming, emr, f1_micro, f1_macro

