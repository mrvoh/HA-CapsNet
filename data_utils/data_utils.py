# from torchnlp.word_to_vector import FastText
from torchnlp.datasets import Dataset
from torchnlp.samplers import BucketBatchSampler
from torchnlp.encoders.text import stack_and_pad_tensors, pad_tensor
from torch.utils.data import DataLoader, SequentialSampler
import torch
# import nltk
import numpy as np
from collections import OrderedDict
import glob
import os
import tqdm
from collections import defaultdict
from data_utils.json_loader import JSONLoader
import operator
import pickle
from collections import Counter
try:
	import fasttext
except:
	print('WARNING: Fasttext module not loaded.')

# UNK = '<UNK>'
# PAD = '<PAD>'
def doc_to_fasttext(in_path, out_path):
	# Read in docs
	with open(in_path, 'rb') as f:
		docs = pickle.load(f)

	with open(out_path, 'w') as f:
		for doc in docs:
			labels = " ".join(["__label__"+tag for tag in doc.tags])
			text = ' '.join([' '.join([word for word in sen]) for sen in doc.sentences])
			f.write(labels + ' ' + text+'\n')


def embeddings_from_docs(in_path, out_path, fasttext_path=None, word_vec_dim = 300, min_count= 5, n_epoch = 20, minn = 3, maxn = 5, lr = 0.05, ):
	# Read in docs
	with open(in_path, 'rb') as f:
		docs = pickle.load(f)

	# Write docs to temporary *.txt file for fasttext to train on
	with open('tmp.txt', 'w', encoding='utf-8') as f:
		for doc in docs:
			f.write('\n'.join([' '.join([word for word in sen]) for sen in doc.sentences]))

	# Train word embeddings
	model = fasttext.train_unsupervised('tmp.txt', dim=word_vec_dim, minCount=min_count, epoch=n_epoch, minn=minn, maxn = maxn, lr = lr )

	model.save_model(out_path)




def get_embedding(vecs, word_to_idx, embed_size):
	# Retrieves relevant word vectors and returns as matrix
	embed_table = [vecs[key] if key in word_to_idx.keys() else np.random.rand(embed_size) for key in sorted(word_to_idx.keys())]

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

# if (word_counter and min_freq): # return if enough occurences when using preloaded word_to_idx
# 			return word_to_idx[word] if word_counter[word] >= min_freq else word_to_idx[unk]

def _convert_word_to_idx(word, word_to_idx, word_counter=None, min_freq=None, unk='<UNK>'):

	try:
		return word_to_idx[word]
	except KeyError:
		if word_counter: # word counter only given for training set
			if word_counter[word] >= min_freq:
				idx = len(word_to_idx)
				word_to_idx[word] = idx
				return idx
		# map to <UNK>
		return word_to_idx[unk]

def doc_to_sample(doc, label_to_idx, word_to_idx= None, word_counter=None, min_freq_word=50, unk='<UNK>', stoi = None, tag_counter = None, label_value = 1.0, binary_class = True):

	n_labels = len(label_to_idx)
	sample = {}
	# collect tags
	tags = [int(label_to_idx[tag]) for tag in doc.tags]
	if tag_counter: # Keep track of #occurrences of each tag
		tag_counter.update(tags)
	# convert sentences to indices of words
	if stoi:  # preloaded word to idx, no need to update it
		sents = [torch.LongTensor([stoi.get(w, stoi[unk]) for w in sent]) for sent in doc.sentences]
	else:
		sents = [
			torch.LongTensor([_convert_word_to_idx(w, word_to_idx, word_counter, min_freq_word, unk) for w in sent]) for
			sent in doc.sentences]

	# convert to tensors
	if binary_class:
		sample['tags'] = np.zeros(n_labels)
		sample['tags'][tags] = 1.0
		sample['tags'] = sample['tags'] * label_value + (1.-label_value)/n_labels

		sample['tags'] = torch.FloatTensor(sample['tags']) #One Hot Encoded target
	else:
		sample['tags'] = torch.LongTensor(tags)

	sample['sents'] = sents

	if doc.encoding is not None: sample['encoding'] = torch.FloatTensor(doc.encoding)

	return sample, tag_counter


def process_dataset(docs, label_to_idx, word_to_idx=None, word_counter=None,unk='<UNK>',pad='<PAD>', pad_idx=0, unk_idx=1, min_freq_word=50, label_value = 1.0, binary_class = True):
	""""
		Process list of docs into Pytorch-ready dataset
	"""
	dset = []
	tag_counter = Counter()
	stoi = None

	if min_freq_word:
		word_counter = Counter([w for doc in docs for sent in doc.sentences for w in sent])

	if word_to_idx is None:
		word_to_idx = OrderedDict()
		word_to_idx[pad] = pad_idx
		word_to_idx[unk] = unk_idx
	elif min_freq_word:
		stoi = {k:v for k,v in word_to_idx.items() if (word_counter[k] >= min_freq_word) or (k in [pad, unk])}

	print('Loading and converting docs to PyTorch backend...')
	for doc in docs:
		sample, tag_counter = doc_to_sample(doc, label_to_idx, word_to_idx, word_counter, stoi=stoi, min_freq_word= min_freq_word,
											unk=unk, tag_counter=tag_counter, label_value=label_value, binary_class = binary_class)

		dset.append(sample)

	return Dataset(dset), word_to_idx, tag_counter

# Collate function
def collate_fn_rnn(batch):
	# # PyTorch RNN requires batches to be transposed for speed and integration with CUDA
	transpose = (lambda b: b.t_().squeeze(0).contiguous())

	# Shape tensors in right format
	sents_len_batch = [[len(sent) for sent in doc['sents']] for doc in batch]
	max_sent_len = max([max(s) for s in sents_len_batch])
	sents_batch, doc_lens_batch = stack_and_pad_tensors(
		[torch.stack([pad_tensor(sent, max_sent_len) for sent in doc['sents']]) for doc in batch])
	tags_batch, _ = stack_and_pad_tensors([doc['tags'] for doc in batch])

	# Move to device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	sents_batch = sents_batch.to(device)
	tags_batch = tags_batch.to(device)

	if 'encoding' in batch[0].keys():  # add doc encoding if applicable
		encoding_batch = torch.stack([doc['encoding'] for doc in batch]).to(device)
		return (sents_batch, tags_batch, encoding_batch)

	# return (word_ids_batch, seq_len_batch, label_batch)
	return (sents_batch, tags_batch, None)

def collate_fn_transformer1(batch):

	# test = [sent for doc in batch for sent in doc['sents']]
	sents_batch, sents_len_batch = stack_and_pad_tensors([sent for doc in batch for sent in doc['sents']])
	doc_lens_batch = [len(doc['sents']) for doc in batch]


	# tokens_batch, _ = stack_and_pad_tensors([doc['tokens'] for doc in batch])
	tags_batch, _ = stack_and_pad_tensors([doc['tags'] for doc in batch])
	# sents_len_batch = stack_and_pad_tensors([doc['sen_lens'] for doc in batch])
	# word_len_batch, _ = stack_and_pad_tensors([seq['word_len'] for seq in batch])

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	sents_batch = sents_batch.to(device)
	sents_len_batch = sents_len_batch.to(device)
	# doc_lens_batch = doc_lens_batch.to(device)
	tags_batch = tags_batch.to(device)

	if 'encoding' in batch[0].keys():
		encoding_batch = torch.stack([doc['encoding'] for doc in batch]).to(device)
		return (sents_batch, sents_len_batch, doc_lens_batch, tags_batch, encoding_batch)

	# return (word_ids_batch, seq_len_batch, label_batch)
	return (sents_batch, sents_len_batch, doc_lens_batch, tags_batch)



def collate_fn_transformer(batch): #multigpu implementation

	# Shape tensors in right format
	sents_len_batch = [[len(sent) for sent in doc['sents']] for doc in batch]
	max_sent_len = max([max(s) for s in sents_len_batch])
	sents_batch, doc_lens_batch = stack_and_pad_tensors([torch.stack([pad_tensor(sent, max_sent_len) for sent in doc['sents']]) for doc in batch])
	tags_batch, _ = stack_and_pad_tensors([doc['tags'] for doc in batch])

	# Move to device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	sents_batch = sents_batch.to(device)
	tags_batch = tags_batch.to(device)

	if 'encoding' in batch[0].keys(): # add doc encoding if applicable
		encoding_batch = torch.stack([doc['encoding'] for doc in batch]).to(device)
		return (sents_batch, tags_batch, encoding_batch)

	# return (word_ids_batch, seq_len_batch, label_batch)
	return (sents_batch, tags_batch, None)


# Get dataloader
def get_data_loader(data, batch_size, drop_last, use_rnn, is_train=False):

	# if is_train:
	sampler = BucketBatchSampler(data,
								 batch_size,
								 drop_last=drop_last,
								 sort_key=lambda row: -len(row['sents']))
	# else:
	# 	sampler = SequentialSampler(data)

	collate_fn = collate_fn_rnn if use_rnn else collate_fn_transformer

	loader = DataLoader(data,
						batch_sampler=sampler,
						collate_fn=collate_fn,
						pin_memory=False)
						# shuffle=True,
						# num_workers=1)
	return loader