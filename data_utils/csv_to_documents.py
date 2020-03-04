import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split
from imblearn.over_sampling import RandomOverSampler
from document_model import Document, TextPreprocessor
import os
import pickle
from tqdm import tqdm
import json
import numpy as np

def docs_to_sheet(in_path, out_path, label_to_idx_path, use_excel = False, delimiter=';', encoding='utf-8', binary_class=True):

	# read in data
	with open(in_path, 'rb') as f:
		docs = pickle.load(f)

	with open(label_to_idx_path, 'r') as f:
		label_to_idx = json.load(f)

	idx_to_label = {v:k for k,v in label_to_idx.items()}

	# extract text and tags
	text = [doc.text for doc in docs]
	tags = [[label_to_idx[t] for t in doc.tags if t != ''] for doc in docs]
	num_tags = max([max(tag, default=0) for tag in tags]) +1
	# convert to one-hot-encoding
	if binary_class:
		ohe_tags = np.zeros((len(docs), num_tags))
		for i, tag in enumerate(tags):
			ohe_tags[i,tag] = 1
		# ohe_tags = np.eye(num_tags)[tags.reshape(-1)]
		tags = ohe_tags
		tag_cols = [idx_to_label[i] for i in range(num_tags)]
	else:
		tag_cols = ['tags']
	# Create df
	df = pd.DataFrame(text, columns=['text'])

	# for tag_col in tag
	for i, tag_col in enumerate(tag_cols):
		test = tags[:,i]
		df[tag_col] = test
	# df.loc[tag_cols] = tags

	if use_excel:
		df.to_excel(out_path, encoding=encoding)
	else:
		df.to_csv(out_path, sep=delimiter, encoding=encoding)

	return df

def df_to_docs(df, label_to_idx, out_dir, do_split, dev_percentage, store_df, set_name, restructure_doc=True, max_seq_len=100,
			  use_ulmfit=False, delimiter='\t', encoding='utf-8', use_excel=True, text_cols='text', target_prefix='topic_',
			  binary_class=True):
	assert 0 < dev_percentage < 1, "the percentage of data to be used for dev should be between 0 and 1."
	# Make sure the output dir exists
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	idx_to_label = {v:k for k,v in label_to_idx.items()}
	target_cols = [col for col in df.columns if target_prefix in col]

	dfs = [(df, set_name)]

	if do_split:
		cols = df.columns
		y = df[target_cols].astype(int).values
		train, _, dev, _ = iterative_train_test_split(df.values, y, test_size=dev_percentage)

		df_train = pd.DataFrame(train, columns=cols)
		df_dev = pd.DataFrame(dev, columns=cols)
		dfs = [(df_train, 'train'), (df_dev, 'dev')]

	if store_df:
		for df, name in dfs:
			out_path = os.path.join(out_dir, name)
			if use_excel:
				df.to_excel(out_path+'.xlsx', encoding=encoding, engine='xlsxwriter')
			else:
				df.to_csv(out_path+'.csv', sep=delimiter, encoding=encoding)

	# Convert rows to Documents
	text_preprocessor = TextPreprocessor(use_ulmfit)

	for df, name in dfs:
		docs = [Document(text=row.text,
						 text_preprocessor=text_preprocessor,
						 filename='test',
						 tags=[idx_to_label[int(t)] for t in np.argwhere(row[target_cols].values == 1)],
						 restructure_doc=restructure_doc,
						 split_size_long_seqs=max_seq_len)
				for ix, row in tqdm(df.iterrows(), desc='Converting to docs')]

		with open(os.path.join(out_dir, name + '.pkl'), 'wb') as f:
			pickle.dump(docs, f)


def sheet_to_docs(in_path, out_dir, dev_percentage, test_percentage, restructure_doc=True, max_seq_len=50,
				  use_ulmfit=False, delimiter=',', encoding='utf-8', use_excel=True, text_cols='text', target_prefix='topic_',
				  binary_class=True, split_val='_', balance_dataset = True):
	assert 0 < dev_percentage < 1, "the percentage of data to be used for dev should be between 0 and 1."
	assert 0 < test_percentage < 1, "the percentage of data to be used for dev should be between 0 and 1."
	assert dev_percentage + test_percentage < 1, "the percentage used for dev and test should be less than 1."

	# Make sure the output dir exists
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	# read in data
	if use_excel:
		df = pd.read_excel(in_path, encoding=encoding)
	else:
		df = pd.read_csv(in_path, delimiter=delimiter, encoding=encoding)

	# Extract label mapping from cols
	label_to_idx = {col: ix for ix, col in enumerate(df[[col for col in df.columns if col.startswith(target_prefix)]].columns) }
	idx_to_label = {v:k for k,v in label_to_idx.items()}
	label_to_idx_path = os.path.join(out_dir, 'label_to_idx.json')
	with open(label_to_idx_path, 'w', encoding='utf-8') as f:
		json.dump(label_to_idx, f)


	# aggregate data into text and target column
	if isinstance(text_cols, list):
		df['text'] = df.apply(lambda x: ' '.join([str(x[col]) for col in text_cols]), axis=1)
	else:
		df.loc['text', :] = df[text_cols]

	if binary_class:
		# convert one-hot-encoding
		target = df[[col for col in df.columns if target_prefix in col]].apply(np.nonzero, axis=1).values
		target = [list(t[0]) for t in target]
		df['target'] = [split_val.join([str(val) for val in t]) for t in target]

		# hacky way to make use of SkMultiLearn
		X = df[['text', 'target']].values
		y = df[[col for col in df.columns if target_prefix in col]].values

		if balance_dataset:
			ros = RandomOverSampler(random_state=42)
			X, y = ros.fit_resample(X, y)

		X, y, X_test, y_test = iterative_train_test_split(X, y, test_size=test_percentage)

		discounted_dev_percentage = dev_percentage / (1. - test_percentage)

		X_train, y_train, X_dev, y_dev = iterative_train_test_split(X, y, test_size=discounted_dev_percentage)

	# Convert rows to Documents
	text_preprocessor = TextPreprocessor(use_ulmfit)

	for dset, name in [(X_train, "train"), (X_dev, "dev"), (X_test, "test")]:
		docs = [Document(text=text,
						 text_preprocessor=text_preprocessor,
						 filename='test',
						 tags=[idx_to_label[int(t)] for t in target.split(split_val) if t != ''],
						 restructure_doc=restructure_doc,
						 split_size_long_seqs=max_seq_len)
				for text, target in tqdm(dset)]

		with open(os.path.join(out_dir, name + '.pkl'), 'wb') as f:
			pickle.dump(docs, f)


