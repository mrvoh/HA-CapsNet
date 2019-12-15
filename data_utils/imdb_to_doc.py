from torchnlp.datasets import imdb_dataset
from document_model import Document, TextPreprocessor
from random import shuffle
import os
import pickle
from tqdm import tqdm
import json

def parse(out_dir, percentage_train, restructure_doc = True, max_seq_len = 50, use_ulmfit=False):

	assert 0 < percentage_train <= 1, "The percentage of docs to be used for training should be between 0 and 1."

	# Make sure the output dir exists
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	# retrieve data and create splits
	train_full = imdb_dataset(train=True)
	shuffle(train_full)
	train = train_full[:int(percentage_train*len(train_full))]
	dev = train_full[int(percentage_train*len(train_full)):]
	del train_full
	test = imdb_dataset(test=True)
	# initiate text preprocessor

	text_preprocessor = TextPreprocessor(use_ulmfit)
	# convert each file to a Document
	for dataset, name in [(train, 'train'), (dev, 'dev'), (test, 'test')]:
		if len(dataset) == 0:
			continue
		docs = [Document(text=doc['text'],
						 text_preprocessor = text_preprocessor,
						 filename = 'test',
						 tags=doc['sentiment'],
						 restructure_doc=restructure_doc,
						 split_size_long_seqs=max_seq_len)
				for doc in tqdm(dataset)]

		with open(os.path.join(out_dir, name+'.pkl'), 'wb') as f:
			pickle.dump(docs, f)


	# Get label_mapping
	label_to_idx = {'neg':0, 'pos':1}

	label_to_idx_path = os.path.join(out_dir, 'label_to_idx.json')
	with open(label_to_idx_path, 'w', encoding='utf-8') as f:
		json.dump(label_to_idx, f)



if __name__ == '__main__':
	None
