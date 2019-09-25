""""
TODO: Data pre-processing
	Own word vecs
	LASER sentences embeddings
	Embed label descriptions?
"""
import os
import pickle
import json
from torchnlp.word_to_vector import FastText
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import gc

from model import HAN, HGRULWAN
from data_utils import process_dataset, get_data_loader, get_embedding
from radam import RAdam
from logger import get_logger, Progbar
from metrics import *
from text_class_learner import MultiLabelTextClassifier

import cProfile, pstats, io
from pstats import SortKey



def eval_dataset(model, dataloader, criterion, K=5, max_samples = None):
	logger.info("Evaluating model")
	model.eval()
	y_pred = []
	y_true = []
	eval_loss = 0

	# Get all predictions on dataset
	with torch.no_grad():
		for batch_idx, batch in enumerate(dataloader):

			(sents, sents_len, doc_lens, target) = batch

			preds, _, _ = model(sents, sents_len, doc_lens)
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
		logger.info(template.format(i, r_k, i, p_k, i, rp_k, i, ndcg_k))
	logger.info('----------------------------------------------------')

	return r_k, p_k, rp_k, ndcg_k, avg_loss


if __name__ == '__main__':

	try:
		from apex import amp
		APEX_AVAILABLE = True
	except ModuleNotFoundError:
		APEX_AVAILABLE = False

	# pr = cProfile.Profile()
	# pr.enable()
	for dataset in ['_500']:
		for model_name in ['HAN']:
			for dropout in [0.1, 0.25, 0.35, 0.5]:
				###################### MOVE TO ARGPARSER ##################################
				pretrained_path = None #os.path.join('models', 'HGRULWAN_loss=0.04070_RP5=0.745.pt')
				# dataset = ''
				dev_path = os.path.join('dataset', 'dev{}.pkl'.format(dataset))
				train_path = os.path.join('dataset', 'train{}.pkl'.format(dataset))
				models_path = 'models'
				min_freq_word = 50
				label_to_idx_path = os.path.join('dataset', 'label_to_idx{}.json'.format(dataset))
				label_map_path = os.path.join('dataset', 'label_map.json')
				word_to_idx_path = os.path.join('dataset', 'word_to_idx_python.json'.format(min_freq_word))
				preload_word_to_idx = False
				B_train = 32 if dataset == '' else 64
				B_eval = 32 if dataset == '' else 64
				vector_cache = "D:\\UvA\\Statistical Methods For Natural Language Semantics\\Assignments\\2\\LASERWordEmbedder\\src\\.word_vectors_cache"
				path_log = "log.txt"
				n_epochs = 25
				eval_every = 2000
				K = 5 # Cut-off value for metrics (e.g. Precision@K)
				word_encoder = 'transformer'
				sent_encoder = 'transformer' #'transformer'
				num_labels = 30 if dataset == '' else 500
				reduction = 'mean'
				# model_name = 'HAN'
				learning_rate = 1e-3
				weight_decay = 1e-4
				# dropout = 0.25
				embed_dim = 300
				word_hidden = sent_hidden = 160
				use_rnn = word_encoder == 'gru'
				###########################################################################
				with open(label_to_idx_path, 'r') as f:
					# idx_to_label = json.load(f)
					label_to_idx = json.load(f)

				with open(label_map_path, 'r') as f:
					# idx_to_label = json.load(f)
					label_map = json.load(f)
				if preload_word_to_idx:
					with open(word_to_idx_path, 'r') as f:
						# idx_to_label = json.load(f)
						word_to_idx = json.load(f)
				else:
					word_to_idx = None




				# logger = get_logger(path_log)
				# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
				# writer = SummaryWriter(comment='Model={}-Labels={}-B={}-L2={}-Dropout={}'.format(model_name, num_labels, B_train, weight_decay, dropout))
				# logger.info("Loading data...")
				# load docs into memory
				with open(train_path, 'rb') as f:
					train_docs = pickle.load(f)

				with open(dev_path, 'rb') as f:
					dev_docs = pickle.load(f)



				# get dataloader
				train_dataset, word_to_idx, tag_counter_train = process_dataset(train_docs, word_to_idx=word_to_idx, label_to_idx=label_to_idx, min_freq_word=min_freq_word)
				pos_weight = [v/len(train_dataset) for k,v in tag_counter_train.items()]
				dataloader_train = get_data_loader(train_dataset, B_train, True, use_rnn)

				# Save word_mapping
				with open(word_to_idx_path, 'w') as f:
					json.dump(word_to_idx, f)

				# Free some memory
				del train_dataset
				del train_docs

				dev_dataset, word_to_idx, tag_counter_dev = process_dataset(dev_docs, word_to_idx=word_to_idx, label_to_idx=label_to_idx, min_freq_word=min_freq_word)
				dataloader_dev = get_data_loader(dev_dataset, B_eval, False, use_rnn)
				# Free some memory
				del dev_dataset
				del dev_docs

				# logger.info("Building model...")
				if pretrained_path:
					TextClassifier = MultiLabelTextClassifier.load(pretrained_path)
				else:
					TextClassifier = MultiLabelTextClassifier(model_name, word_to_idx, label_to_idx, label_map, path_log = path_log, save_dir=models_path,
															  min_freq_word=min_freq_word, word_to_idx_path=word_to_idx_path,
															  B_train=B_train, word_encoder = word_encoder,
															  B_eval=B_eval, weight_decay=weight_decay, lr=learning_rate)

					TextClassifier.init_model(embed_dim, word_hidden, sent_hidden, dropout, vector_cache, word_encoder=word_encoder, sent_encoder=sent_encoder)


				TextClassifier.train(dataloader_train, dataloader_dev, pos_weight, num_epochs=n_epochs, eval_every=eval_every)





	# # get embeddings for dataset
	# vectors = FastText(aligned=True, cache=vector_cache, language='en')
	# embed_table = get_embedding(vectors, word_to_idx)
	#
	# if model_name.lower() == 'han':
	# 	model = HAN(len(word_to_idx), embed_dim, word_gru_hidden, sent_gru_hidden, len(label_to_idx), dropout=dropout)
	# elif model_name.lower() == 'hgrulwan':
	# 	model = HGRULWAN(len(word_to_idx), embed_dim, word_gru_hidden, sent_gru_hidden, len(label_to_idx), dropout=dropout)
	# model.to(device)
	#
	# # embed_table = np.random.rand(len(word_to_idx), 300)
	# model.set_embedding(embed_table)
	# # Free some memory
	# del embed_table
	# del vectors
	# gc.collect()
	#
	# # Add model graph to Tensorboard
	# # (sents, sents_len, doc_lens, target) = next(iter(dataloader_train))
	# # writer.add_graph(model, (sents, sents_len, doc_lens))
	#
	# print("NR PARAMS: ",sum(p.numel() for p in model.parameters() if p.requires_grad))
	#
	# criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device), reduction=reduction)
	# optimizer = RAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
	#
	#
	# logger.info("Training model")
	# train_step = 0
	# best_score = 0
	# best_loss = 10**6
	# for epoch in range(n_epochs):
	# 	logger.info("Epoch: {}".format(epoch))
	# 	prog = Progbar(len(dataloader_train))
	#
	# 	tr_loss = 0
	#
	# 	for batch_idx, batch in enumerate(dataloader_train):
	# 		train_step += 1
	# 		optimizer.zero_grad()
	#
	# 		(sents, sents_len, doc_lens, target) = batch
	# 		preds, word_attention_scores, sent_attention_scores = model(sents, sents_len, doc_lens)
	#
	# 		loss = criterion(preds, target)
	# 		tr_loss += loss.item()
	#
	#
	# 		if APEX_AVAILABLE:
	# 			with amp.scale_loss(loss, optimizer) as scaled_loss:
	# 				scaled_loss.backward()
	# 		else:
	# 			loss.backward()
	#
	# 		optimizer.step()
	# 		prog.update(batch_idx + 1, values=[("train loss", loss.item())])
	#
	# 		if train_step % eval_every == 0:
	# 			# Eval dev
	# 			r_k_dev, p_k_dev, rp_k_dev, ndcg_k_dev, avg_loss_dev = eval_dataset(model, dataloader_dev, criterion, K=K)
	#
	# 			# Eval Train
	# 			r_k_tr, p_k_tr, rp_k_tr, ndcg_k_tr, avg_loss_tr = eval_dataset(model, dataloader_train, criterion, K=K, max_samples=len(dataloader_dev))
	#
	# 			# Save model if best
	# 			if best_score < rp_k_dev:
	# 				best_score = rp_k_dev
	# 				torch.save(model.state_dict(), os.path.join(models_path, model_name+'_loss={}_RP{}={}.pt'.format(avg_loss_dev, K, rp_k_dev)))
	# 				logger.info("Saved model with new best score: {}".format(rp_k_dev))
	# 			elif best_loss > avg_loss_dev:
	# 				best_loss =avg_loss_dev
	# 				torch.save(model.state_dict(), os.path.join(models_path,
	# 															model_name + '_loss={}_RP{}={}.pt'.format(avg_loss_dev,
	# 																									  K, rp_k_dev)))
	# 				logger.info("Saved model with new best loss: {}".format(avg_loss_dev))
	#
	# 			# Write to Tensorboard
	# 			writer.add_scalars("Loss",
	# 							   {"Train":avg_loss_tr,
	# 								"Dev":avg_loss_dev}, train_step)
	# 			writer.add_scalars("R_{}".format(K),
	# 				{"Train":r_k_tr,
	# 				"Dev":r_k_dev},
	# 				train_step)
	# 			writer.add_scalars("P_{}".format(K),
	# 				{"Train": p_k_tr,
	# 				"Dev": p_k_dev},
	# 				train_step)
	# 			writer.add_scalars("RP_{}".format(K),
	# 							   {"Train": rp_k_tr,
	# 								"Dev": rp_k_dev},
	# 							   train_step)
	# 			writer.add_scalars("NDCG_{}".format(K),
	# 							   {"Train": ndcg_k_tr,
	# 								"Dev": ndcg_k_dev},
	# 							   train_step)
	#
	#
	# 			# Return to training mode
	# 			model.train()
	#
	# 	# Write weight and gradient info to Tensorboard
	# 	for name, weight in model.named_parameters():
	# 		writer.add_histogram(name, weight.cpu().detach().numpy(), epoch)
	# 		# writer.add_histogram(f'{name}.grad', weight.grad.cpu().detach().numpy(), epoch) #TODO: check weight.grad
	#
	# # Close tensorboard session
	# writer.close()
	#


