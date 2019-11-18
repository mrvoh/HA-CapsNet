from nltk.corpus import reuters
from document_model import Document, TextPreprocessor
from random import shuffle
import os
import pickle
from tqdm import tqdm
import json

def parse(out_dir, percentage_train, restructure_doc = True, split_size_long_seqs = 50, use_ulmfit=False):

	assert 0 < percentage_train <= 1, "The percentage of docs to be used for training should be between 0 and 1."

	# retrieve data and create splits
	train_full = [x for x in reuters.fileids() if 'train' in x]
	shuffle(train_full)
	train_ids = train_full[:int(percentage_train*len(train_full))]
	dev_ids = train_full[int(percentage_train*len(train_full)):]
	del train_full
	test_ids = [x for x in reuters.fileids() if 'test' in x]
	# initiate text preprocessor

	text_preprocessor = TextPreprocessor(use_ulmfit)
	# convert each file to a Document
	for dataset, name in [(train_ids, 'train'), (dev_ids, 'dev'), (test_ids, 'test')]:

		docs = [Document(text='',
						 text_preprocessor = text_preprocessor,
						 filename = doc_id,
						 sentences=[' '.join(sen) for sen in reuters.sents(doc_id)],
						 tags=reuters.categories(doc_id),
						 restructure_doc=restructure_doc,
						 split_size_long_seqs=split_size_long_seqs)
				for doc_id in tqdm(dataset)]

		with open(os.path.join(out_dir, name+'.pkl'), 'wb') as f:
			pickle.dump(docs, f)


	# Get label_mapping
	label_to_idx = {lab:i for i,lab in enumerate(reuters.categories())}

	label_to_idx_path = os.path.join(out_dir, 'label_to_idx.json')
	with open(label_to_idx_path, 'w', encoding='utf-8') as f:
		json.dump(label_to_idx, f)



if __name__ == '__main__':
	None
	# parse('dataset\\reuters', 0.9)