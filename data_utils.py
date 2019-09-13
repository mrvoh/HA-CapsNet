from torchnlp.word_to_vector import FastText
from urllib.request import urlopen
from torchnlp.datasets import Dataset
from torchnlp.samplers import BucketBatchSampler
from torchnlp.encoders.text import stack_and_pad_tensors, pad_tensor
from torch.utils.data import DataLoader
import torch
import os
import nltk
import numpy as np
import json
from collections import OrderedDict
import glob
import os
import tqdm
from collections import defaultdict
from json_loader import JSONLoader
import operator
import pickle


# vectors = FastText(aligned=True, cache='.word_vectors_cache', language='es')

def get_embedding(vecs, word_to_idx):
    embed_table = [vecs[key].numpy() for key in word_to_idx.keys()]
    embed_table = np.array(embed_table, dtype=float)
    return embed_table

def load_dataset(dataset_dir, dataset_name):
	"""
	Load dataset and return list of documents
	:param dataset_name: the name of the dataset
	:return: list of Document objects
	"""
	filenames = glob.glob(os.path.join(dataset_dir, dataset_name, '*.json'))
	loader = JSONLoader()

	documents = []
	for filename in tqdm.tqdm(sorted(filenames), desc=False):
		documents.append(loader.read_file(filename))

	return documents

def parse_dataset(dataset_dir, dataset_name, nr_tags, tags_to_use=None):
	""""
		Parsed files in 'train', 'dev' and 'test' subfolders in dataset_dir.
		Takes nr_tags most occuring labels of documents in train set and filters out all others.
		Stores *.pickle files in dataset_dir.
	"""

	label_occs = defaultdict(int)
	docs = load_dataset(dataset_dir)
	for doc in docs:
		for tag in doc.tags:
			label_occs[tag] += 1

	if tags_to_use is None:
		sorted_tags = sorted(label_occs.items(), key=operator.itemgetter(1))
		tags_to_use = [x[0] for x in sorted_tags[:nr_tags]]

	for doc in docs:
		doc.tags = [tag for tag in doc.tags if tag in tags_to_use]

	with open(os.path.join(dataset_dir, dataset_name + '.pkl'), 'wb') as f:
		pickle.dump(docs, f)

	# for name in ['dev', 'test']:
	# 	docs = load_dataset(name)
	#
	# 	for doc in docs:
	# 		doc.tags = [tag for tag in doc.tags if tag in tags_to_use]
	#
	# 	with open(os.path.join(DATA_SET_DIR, DATASET_NAME + '.pkl'), 'wb') as f:
	# 		pickle.dump(docs, f)

# Parse dataset from [Documents]

def _convert_word_to_idx(word, word_to_idx):

	try:
		return word_to_idx[word]
	except KeyError:
		idx = len(word_to_idx)
		word_to_idx[word] = idx
		return idx


def process_dataset(docs, label_to_idx, word_to_idx=None):
	""""
		Process list of docs into Pytorch-ready dataset
	"""
	n_labels = len(label_to_idx)
	dset = []
	if word_to_idx is None:
		word_to_idx = OrderedDict()

	for doc in docs:
		sample = {}

		tags = [int(label_to_idx[tag]) for tag in doc.tags]
		sents = [torch.LongTensor([_convert_word_to_idx(w, word_to_idx) for w in sent]) for sent in doc.sentences]
		# sen_lens = [len(sen) for sen in sents]
		# tokens = [tok for sent in sents for tok in sent] # flatten list

		# convert to tensors
		sample['tags'] = np.zeros(n_labels)
		sample['tags'][tags] = 1
		sample['tags'] = torch.FloatTensor(sample['tags']) # One Hot Encoded target
		sample['sents'], _ = stack_and_pad_tensors(sents)
		# sample['sen_lens'] = torch.LongTensor(sen_lens)
		# sample['tokens'] = torch.LongTensor(tokens)

		dset.append(sample)

	return Dataset(dset), word_to_idx



# Collate function

def collate_fn(batch):

	test = [sent for doc in batch for sent in doc['sents']]
	sents_batch, sents_len_batch = stack_and_pad_tensors([sent for doc in batch for sent in doc['sents']])
	doc_lens_batch = [len(doc['sents']) for doc in batch]


	# tokens_batch, _ = stack_and_pad_tensors([doc['tokens'] for doc in batch])
	tags_batch, _ = stack_and_pad_tensors([doc['tags'] for doc in batch])
	# sents_len_batch = stack_and_pad_tensors([doc['sen_lens'] for doc in batch])
	# word_len_batch, _ = stack_and_pad_tensors([seq['word_len'] for seq in batch])

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	sents_batch = sents_batch.to(device)
	sents_len_batch = sents_len_batch.to(device)
	# doc_lens_batch = doc_lens_batch.to(device)
	tags_batch = tags_batch.to(device)

	# PyTorch RNN requires batches to be transposed for speed and integration with CUDA
	transpose = (lambda b: b.t_().squeeze(0).contiguous())

	# return (word_ids_batch, seq_len_batch, label_batch)
	return (transpose(sents_batch), sents_len_batch, doc_lens_batch, tags_batch)



# Get dataloader
def get_data_loader(data, batch_size, drop_last, collate_fn=collate_fn):
    sampler = BucketBatchSampler(data,
                                 batch_size,
                                 drop_last=drop_last,
                                 sort_key=lambda row: -max([len(sent) for sent in row['sents']]))

    loader = DataLoader(data,
                        batch_sampler=sampler,
                        collate_fn=collate_fn)

    return loader