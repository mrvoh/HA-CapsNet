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


if __name__ == '__main__':

	###################### MOVE TO ARGPARSER ##################################
	dev_path = os.path.join('dataset', 'dev.pkl')
	label_to_idx_path = os.path.join('dataset', 'label_to_idx.json')
	B = 15
	vector_cache = "D:\\UvA\\Statistical Methods For Natural Language Semantics\\Assignments\\2\\LASERWordEmbedder\\src\\.word_vectors_cache"
	path_log = "log.txt"
	n_epochs = 10
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
	dataloader = get_data_loader(dataset, B, True)
	# get embeddings for dataset
	vectors = FastText(aligned=True, cache=vector_cache, language='en')
	embed_table = get_embedding(vectors, word_to_idx)

	model = HAN(B, len(word_to_idx), 300, 100, 100, len(label_to_idx))
	model.to(device)
	model.set_embedding(embed_table)

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
			optimizer.zero_grad()


			(sents, sents_len, doc_lens, target) = batch
			preds, attention_scores = model(sents, sents_len, doc_lens)
			#TODO: tensorboard logging

			loss = criterion(preds, target)
			loss.backward()
			optimizer.step()
			prog.update(batch_idx + 1, values=[("train loss", loss.item())])


