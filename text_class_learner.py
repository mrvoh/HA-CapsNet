import torch
import os
import pickle
import json
from torchnlp.word_to_vector import FastText
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import gc
from torchnlp.encoders.text import stack_and_pad_tensors
from scipy.special import expit

from model import HAN, HGRULWAN
from data_utils import process_dataset, get_data_loader, get_embedding, _convert_word_to_idx
from radam import RAdam
from logger import get_logger, Progbar
from metrics import *
from document_model import Document

try:
	from apex import amp
	APEX_AVAILABLE = True
except ModuleNotFoundError:
	APEX_AVAILABLE = False


class MultiLabelTextClassifier:

	def __init__(self, model_name, word_to_idx, label_to_idx, label_map, min_freq_word = 100,
				 tensorboard_path = 'runs', B_train = 16, B_eval = 32, weight_decay = 1e-4, lr = 1e-3,
				 dropout = 0.1, K=5, verbose=True, **kwargs):

		self.model_name = model_name
		self.model = None
		self.word_to_idx = word_to_idx
		self.label_to_idx = label_to_idx # Maps label_ids to index in model
		self.label_map = label_map # Maps label ids to EUROVOC description
		self.log_path = kwargs.get('log_path', 'log.txt')
		self.save_dir = kwargs.get('save_dir', None) #TODO: use save_dir
		self.tensorboard_path = tensorboard_path
		self.B_train = B_train
		self.B_eval = B_eval
		self.lr = lr
		self.weight_decay = weight_decay
		self.dropout = dropout
		self.min_freq_word = min_freq_word
		self.K = K
		self.verbose = verbose

		# Placeholders for attributes to be initialized
		self.embed_size = kwargs.get('embed_size', None)
		self.word_hidden = kwargs.get('word_hidden', None)
		self.sent_hidden = kwargs.get('sent_hidden', None)
		self.word_encoder = kwargs.get('word_encoder', None)
		self.sent_encoder = kwargs.get('sent_encoder', None)

		# Other attributes
		self.vocab_size = len(word_to_idx)
		self.num_labels = len(label_to_idx)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.logger = get_logger(self.log_path)
		self.writer = SummaryWriter(comment='Model={}-Labels={}-B={}-L2={}-Dropout={}'.format(model_name, self.num_labels, B_train, weight_decay, dropout))

	def save(self, path):
		params = {
			"model_name":self.model_name,
			"word_to_idx":self.word_to_idx,
			"label_to_idx":self.label_to_idx,
			"label_map":self.label_map,
			"embed_size":self.embed_size,
			"word_hidden":self.word_hidden,
			"sent_hidden":self.sent_hidden,
			"dropout":self.dropout,
			"word_encoder":self.word_encoder,
			"sent_encoder":self.sent_encoder,
			"state_dict":self.model.state_dict()
		}

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

		model.load_state_dict(params['state_dict'])
		model.to(self.device)
		self.model = model

		return self

	def init_model(self, embed_dim, word_hidden, sent_hidden, dropout, vector_cache, word_encoder = 'gru', sent_encoder = 'gru'):

		self.embed_size = embed_dim
		self.word_hidden = word_hidden
		self.sent_hidden = sent_hidden
		self.dropout = dropout
		self.word_encoder = word_encoder
		self.sent_encoder = sent_encoder

		# Initialize model and load pretrained weights if given
		self.logger.info("Building model...")
		if self.model_name.lower() == 'han':
			self.model = HAN(self.vocab_size, embed_dim, word_hidden, sent_hidden, self.num_labels, dropout=dropout,
							 word_encoder = word_encoder, sent_encoder = sent_encoder)
		elif self.model_name.lower() == 'hgrulwan':
			self.model = HGRULWAN(self.vocab_size, embed_dim, word_hidden, sent_hidden, self.num_labels, dropout=dropout)

		# Load embeddings
		vectors = FastText(aligned=True, cache=vector_cache, language='en')
		embed_table = get_embedding(vectors, self.word_to_idx)
		self.model.set_embedding(embed_table)

		self.model.to(self.device)


	def pred_to_labels(self, pred, top=5):
		# Converts prediction for single doc to original label names
		if isinstance(pred, torch.Tensor):
			pred = pred.cpu().numpy()
			
		pred = pred[0]


		# normed = [x/max(pred[0]) for x in pred[0]]
		ind = np.argpartition(pred, -top)[-top:]
		highest_scoring = ind[np.argsort(pred[ind])]


		# pred = expit(pred)
		#
		#
		# _, sig_pred = np.where(pred > 0.5)
		# print(pred)

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
			preds, word_attention_scores, sent_attention_scores = self.model(transpose(sample['sents']), sample['sents_len'],
																	sample['doc_len'])

		# convert to lists
		preds = list(preds.cpu().numpy())
		word_attention_scores = transpose(word_attention_scores.squeeze()).cpu().numpy().tolist()
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

		# # Get dataloaders for training and evaluation
		# dataloader_train, dataloader_dev, pos_weight = self._get_dataloaders(train_path, dev_path)

		criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(self.device), reduction='sum')
		optimizer = RAdam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

		# Train epoch
		best_score, best_loss, train_step = (0,0,0)

		for epoch in range(num_epochs):
			torch.cuda.empty_cache()
			self.logger.info("Epoch: {}".format(epoch))
			best_score, best_loss, train_step = self._train_epoch(dataloader_train, dataloader_dev, optimizer,
																  criterion, eval_every, train_step, best_score,
																  best_loss)

	def _train_epoch(self, dataloader_train, dataloader_dev, optimizer, criterion, eval_every, train_step, best_score, best_loss):

		prog = Progbar(len(dataloader_train))

		tr_loss = 0

		for batch_idx, batch in enumerate(dataloader_train):
			train_step += 1
			optimizer.zero_grad()

			(sents, sents_len, doc_lens, target) = batch
			preds, word_attention_scores, sent_attention_scores = self.model(sents, sents_len, doc_lens)

			loss = criterion(preds, target)
			tr_loss += loss.item()

			if APEX_AVAILABLE:
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
			else:
				loss.backward()

			optimizer.step()
			prog.update(batch_idx + 1, values=[("train loss", loss.item())])
			torch.cuda.empty_cache()

			if train_step % eval_every == 0:
				best_score, best_loss = self._eval_model(dataloader_train, dataloader_dev, criterion, best_score, best_loss, train_step)

		return best_score, best_loss, train_step


	def _eval_model(self, dataloader_train, dataloader_dev, criterion, best_score, best_loss, train_step):
		# Eval dev
		r_k_dev, p_k_dev, rp_k_dev, ndcg_k_dev, avg_loss_dev = self._eval_dataset(dataloader_dev, criterion, K=self.K)

		# Eval Train
		r_k_tr, p_k_tr, rp_k_tr, ndcg_k_tr, avg_loss_tr = self._eval_dataset(dataloader_train, criterion, K=self.K,
																	   max_samples=len(dataloader_dev))

		# Save model if best
		if best_score < rp_k_dev:
			best_score = rp_k_dev

			self.save(os.path.join(self.save_dir, self.model_name + '_loss={0:.5f}_RP{1}={2:.3f}.pt'.format(avg_loss_dev,self.K, rp_k_dev)))
			# torch.save(self.model.state_dict(),
			# 		   os.path.join(self.save_dir, self.model_name + '_loss={0:.5f}_RP{1}={2:.3f}.pt'.format(avg_loss_dev, self.K, rp_k_dev)))
			self.logger.info("Saved model with new best score: {0:.3f}".format(rp_k_dev))
		elif best_loss > avg_loss_dev:
			best_loss = avg_loss_dev

			self.save(os.path.join(self.save_dir, self.model_name + '_loss={0:.5f}_RP{1}={2:.3f}.pt'.format(avg_loss_dev,self.K, rp_k_dev)))

			# torch.save(self.model.state_dict(), os.path.join(self.save_dir,
			# 											self.model_name + '_loss={0:.5f}_RP{1}={2:.3f}.pt'.format(avg_loss_dev,
			# 																					  self.K, rp_k_dev)))
			self.logger.info("Saved model with new best loss: {0:.5f}".format(avg_loss_dev))

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

	def _eval_dataset(self, dataloader, criterion, K=5, max_samples=None):
		self.logger.info("Evaluating model")
		self.model.eval()
		y_pred = []
		y_true = []
		eval_loss = 0

		# Get all predictions on dataset
		with torch.no_grad():
			for batch_idx, batch in enumerate(dataloader):

				(sents, sents_len, doc_lens, target) = batch

				preds, _, _ = self.model(sents, sents_len, doc_lens)
				loss = criterion(preds, target)
				eval_loss += loss.item()
				# store predictions and targets
				y_pred.extend(list(preds.cpu().detach().numpy()))
				y_true.extend(list(target.cpu().detach().numpy()))

				if max_samples:
					if batch_idx >= max_samples:
						break

		avg_loss = eval_loss / len(dataloader)

		template = 'R@{} : {:1.3f}   P@{} : {:1.3f}   RP@{} : {:1.3f}   NDCG@{} : {:1.3f}'

		for i in range(1, K + 1):
			r_k = mean_recall_k(y_true,
								y_pred, k=i)
			p_k = mean_precision_k(y_true,
								   y_pred, k=i)
			rp_k = mean_rprecision_k(y_true,
									 y_pred, k=i)
			ndcg_k = mean_ndcg_score(y_true,
									 y_pred, k=i)
			self.logger.info(template.format(i, r_k, i, p_k, i, rp_k, i, ndcg_k))
		self.logger.info('----------------------------------------------------')

		return r_k, p_k, rp_k, ndcg_k, avg_loss

	def _get_dataloaders(self, train_path, dev_path):

		self.logger.info("Loading data...")
		# load docs into memory
		with open(train_path, 'rb') as f:
			train_docs = pickle.load(f)

		with open(dev_path, 'rb') as f:
			dev_docs = pickle.load(f)

		# Load dataset
		train_dataset, word_to_idx, tag_counter_train = process_dataset(train_docs, word_to_idx=self.word_to_idx,
																		label_to_idx=self.label_to_idx,
																		min_freq_word=self.min_freq_word)
		pos_weight = [v / len(train_dataset) for k, v in tag_counter_train.items()]
		dataloader_train = get_data_loader(train_dataset, self.B_train, True)

		# Save word_mapping
		if self.word_to_idex_path:
			with open(self.word_to_idx_path, 'w') as f:
				json.dump(word_to_idx, f)

		# Free some memory
		del train_dataset
		del train_docs

		dev_dataset, word_to_idx, tag_counter_dev = process_dataset(dev_docs, word_to_idx=word_to_idx,
																	label_to_idx=self.label_to_idx,
																	min_freq_word=self.min_freq_word)
		dataloader_dev = get_data_loader(dev_dataset, self.B_eval, False)
		# Free some memory
		del dev_dataset
		del dev_docs

		return dataloader_train, dataloader_dev, pos_weight
