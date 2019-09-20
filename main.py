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

			preds, attention_scores = model(sents, sents_len, doc_lens)
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
	###################### MOVE TO ARGPARSER ##################################
	dev_path = os.path.join('dataset', 'dev.pkl')
	train_path = os.path.join('dataset', 'train.pkl')
	models_path = 'models'
	label_to_idx_path = os.path.join('dataset', 'label_to_idx.json')
	B_train = 20
	B_eval = 20
	vector_cache = "D:\\UvA\\Statistical Methods For Natural Language Semantics\\Assignments\\2\\LASERWordEmbedder\\src\\.word_vectors_cache"
	path_log = "log.txt"
	n_epochs = 100
	eval_every = 2000
	K = 5 # Cut-off value for metrics (e.g. Precision@K)
	min_freq_word = 100
	num_labels = 25
	reduction = 'mean'
	model_name = 'HGRULWAN'
	learning_rate = 1e-3
	weight_decay = 0
	dropout = 0.1
	###########################################################################
	logger = get_logger(path_log)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	writer = SummaryWriter(comment='Model={}-Labels={}-B={}-Reduction={}'.format(model_name,num_labels, B_train, reduction))
	logger.info("Loading data...")
	# load docs into memory
	with open(train_path, 'rb') as f:
		train_docs = pickle.load(f)

	with open(dev_path, 'rb') as f:
		dev_docs = pickle.load(f)

	with open(label_to_idx_path, 'r') as f:
		# idx_to_label = json.load(f)
		label_to_idx = json.load(f)
		# label_to_idx = {v:k for k,v in idx_to_label.items()}
	# get dataloader
	train_dataset, word_to_idx, tag_counter_train = process_dataset(train_docs, label_to_idx)
	pos_weight = [v/len(train_dataset) for k,v in tag_counter_train.items()]
	dataloader_train = get_data_loader(train_dataset, B_train, True)

	# Free some memory
	del train_dataset
	del train_docs

	dev_dataset, word_to_idx, tag_counter_dev = process_dataset(dev_docs, word_to_idx=word_to_idx, label_to_idx=label_to_idx, min_freq_word=min_freq_word)
	dataloader_dev = get_data_loader(dev_dataset, B_eval, False)
	# Free some memory
	del dev_dataset
	del dev_docs

	logger.info("Building model...")
	# get embeddings for dataset
	vectors = FastText(aligned=True, cache=vector_cache, language='en')
	embed_table = get_embedding(vectors, word_to_idx)

	if model_name.lower() == 'han':
		model = HAN(len(word_to_idx), 300, 50, 50, len(label_to_idx), dropout=dropout)
	elif model_name.lower() == 'hgrulwan':
		model = HGRULWAN(len(word_to_idx), 300, 50, 50, len(label_to_idx), dropout=dropout)
	model.to(device)

	# embed_table = np.random.rand(len(word_to_idx), 300)
	model.set_embedding(embed_table)
	# Free some memory
	del embed_table
	del vectors
	gc.collect()

	# Add model graph to Tensorboard
	# (sents, sents_len, doc_lens, target) = next(iter(dataloader_train))
	# writer.add_graph(model, (sents, sents_len, doc_lens))

	print("NR PARAMS: ",sum(p.numel() for p in model.parameters() if p.requires_grad))

	criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device), reduction=reduction)
	optimizer = RAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


	logger.info("Training model")
	train_step = 0
	best_score = 0
	best_loss = 10**6
	for epoch in range(n_epochs):
		logger.info("Epoch: {}".format(epoch))
		prog = Progbar(len(dataloader_train))

		tr_loss = 0

		for batch_idx, batch in enumerate(dataloader_train):
			train_step += 1
			optimizer.zero_grad()

			(sents, sents_len, doc_lens, target) = batch
			preds, attention_scores = model(sents, sents_len, doc_lens)

			loss = criterion(preds, target)
			tr_loss += loss.item()


			if APEX_AVAILABLE:
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
			else:
				loss.backward()

			optimizer.step()
			prog.update(batch_idx + 1, values=[("train loss", loss.item())])

			if train_step % eval_every == 0:
				# Eval dev
				r_k_dev, p_k_dev, rp_k_dev, ndcg_k_dev, avg_loss_dev = eval_dataset(model, dataloader_dev, criterion, K=K)

				# Eval Train
				r_k_tr, p_k_tr, rp_k_tr, ndcg_k_tr, avg_loss_tr = eval_dataset(model, dataloader_train, criterion, K=K, max_samples=len(dataloader_dev))

				# Save model if best
				if best_score < rp_k_dev:
					best_score = rp_k_dev
					torch.save(model.state_dict(), os.path.join(models_path, model_name+'_loss={}_RP{}={}.pt'.format(avg_loss_dev, K, rp_k_dev)))
					logger.info("Saved model with new best score: {}".format(rp_k_dev))
				elif best_loss > avg_loss_dev:
					best_loss =avg_loss_dev
					torch.save(model.state_dict(), os.path.join(models_path,
																model_name + '_loss={}_RP{}={}.pt'.format(avg_loss_dev,
																										  K, rp_k_dev)))
					logger.info("Saved model with new best loss: {}".format(avg_loss_dev))

				# Write to Tensorboard
				writer.add_scalars("Loss",
								   {"Train":avg_loss_tr,
									"Dev":avg_loss_dev}, train_step)
				writer.add_scalars("R_{}".format(K),
					{"Train":r_k_tr,
					"Dev":r_k_dev},
				 	train_step)
				writer.add_scalars("P_{}".format(K),
					{"Train": p_k_tr,
					"Dev": p_k_dev},
					train_step)
				writer.add_scalars("RP_{}".format(K),
								   {"Train": rp_k_tr,
									"Dev": rp_k_dev},
								   train_step)
				writer.add_scalars("NDCG_{}".format(K),
								   {"Train": ndcg_k_tr,
									"Dev": ndcg_k_dev},
								   train_step)


				# Return to training mode
				model.train()

		# Write weight and gradient info to Tensorboard
		for name, weight in model.named_parameters():
			writer.add_histogram(name, weight.cpu().detach().numpy(), epoch)
			# writer.add_histogram(f'{name}.grad', weight.grad.cpu().detach().numpy(), epoch) #TODO: check weight.grad

	# Close tensorboard session
	writer.close()



