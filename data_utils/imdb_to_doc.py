from torchnlp.datasets import imdb_dataset

from document_model import Document, TextPreprocessor
from random import shuffle
import os
import pickle
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from data_utils.csv_to_documents import df_to_docs


def imdb_to_df(is_train, label_to_idx):

	dset = imdb_dataset(train=is_train, test=not is_train)

	# create one hot encoding of labels
	num_labels = len(label_to_idx)
	all_labels = np.zeros((len(dset.rows), num_labels))
	all_label_indices = [[label_to_idx[row['sentiment']]] for row in dset.rows]

	for i, labs in enumerate(all_label_indices):
		# binary encode the labels
		all_labels[i][labs] = 1
	all_labels = all_labels.astype(int)

	cols = ['text']
	label_cols = ['topic_{}'.format(lab) for lab in label_to_idx.keys()]
	cols.extend(label_cols)
	df = pd.DataFrame(columns=cols)
	df['text'] = [row['text'] for row in dset.rows]

	df[label_cols] = all_labels

	return df


def parse(out_dir, percentage_train, restructure_doc = True, max_seq_len = 50, use_ulmfit=False):

	assert 0 < percentage_train <= 1, "The percentage of docs to be used for training should be between 0 and 1."

	# Make sure the output dir exists
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	label_to_idx = {'pos':0, 'neg':1}

	# convert to dataframes
	train_df = imdb_to_df(True, label_to_idx)
	test_df = imdb_to_df(False, label_to_idx)

	# split and process data to Documents
	df_to_docs(train_df, label_to_idx, out_dir, do_split=True, dev_percentage=1 - percentage_train, store_df=True,
			   set_name='train', restructure_doc=restructure_doc, max_seq_len=max_seq_len, use_ulmfit=use_ulmfit)
	df_to_docs(test_df, label_to_idx, out_dir, do_split=False, dev_percentage=.5, store_df=True,
			   set_name='test', restructure_doc=restructure_doc, max_seq_len=max_seq_len, use_ulmfit=use_ulmfit)

	label_to_idx_path = os.path.join(out_dir, 'label_to_idx.json')
	with open(label_to_idx_path, 'w', encoding='utf-8') as f:
		json.dump(label_to_idx, f)


if __name__ == '__main__':
	pass
