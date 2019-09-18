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
from tqdm import tqdm

from model import HAN
from data_utils import process_dataset, get_data_loader, get_embedding
from radam import RAdam
from logger import get_logger, Progbar
from metrics import *

import cProfile, pstats, io
from pstats import SortKey

def eval_dataset(model, dataloader, K=5):
	logger.info("Evaluating model")
	model.eval()
	y_pred = []
	y_true = []
	# Get all predictions on dataset
	for batch_idx, batch in enumerate(dataloader):

		(sents, sents_len, doc_lens, target) = batch

		preds, attention_scores = model(sents, sents_len, doc_lens)

		# store predictions and targets
		y_pred.extend(list(preds.cpu().detach().numpy()))
		y_true.extend(list(target.cpu().detach().numpy()))


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

	# return r_k, p_k,





if __name__ == '__main__':

	pr = cProfile.Profile()
	pr.enable()
	###################### MOVE TO ARGPARSER ##################################
	dev_path = os.path.join('dataset', 'dev.pkl')
	label_to_idx_path = os.path.join('dataset', 'label_to_idx.json')
	B_train = 32
	B_eval = 200
	vector_cache = "D:\\UvA\\Statistical Methods For Natural Language Semantics\\Assignments\\2\\LASERWordEmbedder\\src\\.word_vectors_cache"
	path_log = "log.txt"
	n_epochs = 10
	eval_every = 1
	###########################################################################

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	# load docs into memory
	with open(dev_path, 'rb') as f:
		docs = pickle.load(f)

	with open(label_to_idx_path, 'r') as f:
		# idx_to_label = json.load(f)
		label_to_idx = json.load(f)
		# label_to_idx = {v:k for k,v in idx_to_label.items()}
	# get dataloader
	dataset, word_to_idx = process_dataset(docs, label_to_idx)
	dataloader = get_data_loader(dataset, B_train, True)
	# get embeddings for dataset
	vectors = FastText(aligned=True, cache=vector_cache, language='en')
	embed_table = get_embedding(vectors, word_to_idx)

	model = HAN(B_train, len(word_to_idx), 300, 50, 50, len(label_to_idx))
	model.to(device)

	# embed_table = np.random.rand(len(word_to_idx), 300)
	model.set_embedding(embed_table)

	print(model)

	print("NR PARAMS: ",sum(p.numel() for p in model.parameters() if p.requires_grad))

	#TODO: compute class inbalance for weighted loss
	criterion = torch.nn.BCEWithLogitsLoss()
	optimizer = RAdam(model.parameters(), lr=0.001)
	logger = get_logger(path_log)

	logger.info("Training model")
	train_step = 0
	for epoch in range(n_epochs):
		logger.info("Epoch: {}".format(epoch))
		prog = Progbar(len(dataloader))
		for batch_idx, batch in enumerate(dataloader):
			train_step += 1
			optimizer.zero_grad()


			(sents, sents_len, doc_lens, target) = batch
			if sents.size()[0] > 1000:
				print('HIER: LARGE SENTENCE SIZE SKIPPED')
				continue
			preds, attention_scores = model(sents, sents_len, doc_lens)
			#TODO: tensorboard logging

			loss = criterion(preds, target)
			loss.backward()
			optimizer.step()
			prog.update(batch_idx + 1, values=[("train loss", loss.item())])

			if train_step % eval_every == 0:
				pr.disable()
				s = io.StringIO()
				sortby = SortKey.CUMULATIVE
				ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
				ps.print_stats()
				print(s.getvalue())
				# Get dataloader

				# Eval
				eval_dataset(model, dataloader)


				model.train()




