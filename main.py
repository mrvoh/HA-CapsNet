""""
TODO: Data pre-processing
	Own word vecs
	LASER sentences embeddings
	Embed label descriptions?
"""
import os
import pickle
import json
import argparse
import gc
import cProfile, pstats, io
from pstats import SortKey

from data_utils import process_dataset, get_data_loader, embeddings_from_docs
from text_class_learner import MultiLabelTextClassifier
from eur_lex57k_to_doc import parse as eur_lex_parse


if __name__ == '__main__':

	try:
		from apex import amp
		APEX_AVAILABLE = True
	except ModuleNotFoundError:
		APEX_AVAILABLE = False

	#########################################################################################
	# ARGUMENT PARSING
	#########################################################################################

	parser = argparse.ArgumentParser()

	#  MAIN ARGUMENTS
	parser.add_argument("--do_train",
						action='store_false',
						help="Whether to run training.")
	parser.add_argument("--do_eval",
						action='store_false',
						help="Whether to run eval on the dev set.")
	parser.add_argument("--pretrained_path",
						default=None,
						type=str,
						required=False,
						help="The path from where the class-names are to be retrieved.")

	#  TRAIN/EVAL ARGS
	parser.add_argument("--train_batch_size",
						default=32,
						type=int,
						help="Total batch size for training.")
	parser.add_argument("--eval_batch_size",
						default=32,
						type=int,
						help="Total batch size for eval.")
	parser.add_argument("--learning_rate",
						default=1e-3,
						type=float,
						help="The initial learning rate for Adam.")
	parser.add_argument("--dropout",
						default=0.15,
						type=float,
						help="The initial learning rate for Adam.")
	parser.add_argument("--num_train_epochs",
						default=10,
						type=int,
						help="Total number of training epochs to perform.")
	parser.add_argument("--eval_every",
						default=150,
						type=int,
						help="Nr of training updates before evaluating model")
	parser.add_argument("--K",
						default=1,
						type=int,
						help="Cut-off value for evaluation metrics")
	parser.add_argument("--weight_decay",
						default=5e-4,
						type=float,
						help="L2 regularization term.")

	#  MODEL ARGS
	parser.add_argument("--model_name",
						default='han',#'HCapsNetMultiHeadAtt',
						type=str,
						required=False,
						help="Model to use. Options: HAN, HGRULWAN, HCapsNet & HCapsNetMultiHeadAtt")
	parser.add_argument("--word_encoder",
					   default='gru',
					   type=str,
					   required=False,
					   help="The path from where the class-names are to be retrieved.")
	parser.add_argument("--sent_encoder",
						default='gru',
						type=str,
						required=False,
						help="The path from where the class-names are to be retrieved.")
	parser.add_argument("--embed_dim",
						default=300,
						type=int,
						help="Nr of dimensions of used word vectors")
	parser.add_argument("--word_hidden",
						default=256,
						type=int,
						help="Nr of hidden units word encoder. If GRU is used as encoder, the nr of hidden units is used for BOTH forward and backward pass, resulting in double resulting size.")
	parser.add_argument("--sent_hidden",
						default=256,
						type=int,
						help="Nr of hidden units sentence encoder. If GRU is used as encoder, the nr of hidden units is used for BOTH forward and backward pass, resulting in double resulting size.")

		# caps net options
	parser.add_argument("--dim_caps",
						default=16,
						type=int,
						help="Nr of dimensions of a capsule")
	parser.add_argument("--num_caps",
						default=48,
						type=int,
						help="Number of capsules in Primary Capsule Layer")
	parser.add_argument("--num_compressed_caps",
						default=200,
						type=int,
						help="Number of compressed capsules")

	#	DATA & PRE-PROCESSING ARGS
	parser.add_argument("--preprocess_all",
						action='store_true',
						help="Whether pre-process the dataset from a set of *.json files to a loadable dataset.")
	parser.add_argument("--raw_data_dir",
						default='dataset',
						type=str,
						required=False,
						help="Dir from where to parse the data.")
	parser.add_argument("--num_tags",
						default=100,
						type=int,
						help="Number of labels to use from the data (filters top N occurring)")
	parser.add_argument("--dataset_name",
						default='reuters',
						type=str,
						required=False,
						help="Name of the dataset.")
	parser.add_argument("--write_data_dir",
						default='dataset',
						type=str,
						required=False,
						help="Where to write the parsed data to.")
	parser.add_argument("--num_backtranslations",
						default=1,
						type=int,
						help="Number of times to backtranslate each document as means of data augmentation. Set to None for no backtranslation.")


	parser.add_argument("--train_path",
						default=os.path.join('dataset', 'reuters', 'train.pkl'),
						type=str,
						required=False,
						help="The path from where the train dataset is to be retrieved.")
	parser.add_argument("--dev_path",
						default=os.path.join('dataset', 'reuters', 'dev.pkl'),
						type=str,
						required=False,
						help="The path from where the dev dataset is to be retrieved.")
	parser.add_argument("--test_path",
						default=os.path.join('dataset', 'reuters', 'test.pkl'),
						type=str,
						required=False,
						help="The path from where the dev dataset is to be retrieved.")
	parser.add_argument("--create_wordvecs",
						action='store_false',
						help="Whether to create custom word vectors using FastText.")
	parser.add_argument("--word_vec_path",
						default='dataset\\reuters\\fasttext.model',
						# "D:\\UvA\\Statistical Methods For Natural Language Semantics\\Assignments\\2\\LASERWordEmbedder\\src\\.word_vectors_cache",
						type=str,
						required=False,
						help="Folder where vector caches are stored.")
	parser.add_argument("--preload_word_to_idx",
						action='store_true',
						help="Whether to use an existing word to idx mapping.")
	parser.add_argument("--word_to_idx_path",
						default=os.path.join('dataset', 'word_to_idx_500.pkl'),
						type=str,
						required=False,
						help="The path from where the word mapping is to be retrieved.")
	parser.add_argument("--label_to_idx_path",
						default=os.path.join('dataset', 'reuters', 'label_to_idx.json'),
						type=str,
						required=False,
						help="The path from where the label mapping is to be retrieved.")
	parser.add_argument("--label_map_path",
						default=None, #os.path.join('dataset', 'label_map.json'),
						type=str,
						required=False,
						help="The path from where the mapping from label id to label description is loaded.")
	parser.add_argument("--min_freq_word",
						default=5,
						type=int,
						help="Minimum nr of occurrences before being assigned a word vector")

	#	OTHER ARGS
	parser.add_argument("--log_path",
						default='log.txt',
						type=str,
						required=False,
						help="The path where to dump logging.")
	parser.add_argument("--tensorboard_dir",
						default='runs',
						type=str,
						required=False,
						help="The path where to dump logging.")
	parser.add_argument("--save_dir",
						default='models',
						type=str,
						required=False,
						help="Folder where to save models when training.")
	args = parser.parse_args()

	use_rnn = args.word_encoder == 'gru'
	train_path, dev_path, test_path = (args.train_path, args.dev_path, args.test_path)
	label_to_idx_path = args.label_to_idx_path
	word_vec_path = args.word_vec_path
	label_map = None
	# pr = cProfile.Profile()
	# pr.enable()

	###########################################################################
	# SANITY CHECKS
	###########################################################################

	assert (args.do_train or args.do_eval), "Either do_train and/or do_eval must be chosen"

	if (args.preload_word_to_idx or args.pretrained_path):
		assert args.word_to_idx_path, "When either --preload_word_to_idx or --pretrained_path is given, its respectice --word_to_idx_path must also be given"

	if args.do_eval:
		assert args.test_path, "when --do_eval is set, --test_path must also be set"

	if args.preprocess_all:
		assert args.raw_data_dir, "When --preprocess_all is set, --raw_data_dir must also be set"
		assert args.write_data_dir, "When --preprocess_all is set, --write_data_dir must also be set"

	###########################################################################
	# DATA PRE-PROCESSING
	###########################################################################

	if args.preprocess_all:
		label_to_idx_path, train_path, dev_path, test_path = \
			eur_lex_parse(args.raw_data_dir, args.write_data_dir, args.dataset_name, args.num_tags, args.num_backtranslations)

	if args.create_wordvecs: # Create word vectors from train documents
		print('Creating word vectors')
		embeddings_from_docs(train_path, word_vec_path, word_vec_dim=args.embed_dim)

	###########################################################################
	# DATA LOADING
	###########################################################################
	with open(label_to_idx_path, 'r') as f:
		# idx_to_label = json.load(f)
		label_to_idx = json.load(f)

	if args.label_map_path:
		with open(args.label_map_path, 'r') as f:
			# idx_to_label = json.load(f)
			label_map = json.load(f)
	if args.preload_word_to_idx:
		with open(args.word_to_idx_path, 'r') as f:
			# idx_to_label = json.load(f)
			word_to_idx = json.load(f)
	else:
		word_to_idx = None

	# load docs into memory
	if args.do_train:
		with open(train_path, 'rb') as f:
			train_docs = pickle.load(f)

		with open(dev_path, 'rb') as f:
			dev_docs = pickle.load(f)



		# get dataloader
		train_dataset, word_to_idx, tag_counter_train = process_dataset(train_docs, word_to_idx=word_to_idx, label_to_idx=label_to_idx, min_freq_word=args.min_freq_word)
		pos_weight = [v/len(train_dataset) for k,v in tag_counter_train.items()]
		dataloader_train = get_data_loader(train_dataset, args.train_batch_size, True, use_rnn)

		# Save word_mapping
		with open(args.word_to_idx_path, 'w') as f:
			json.dump(word_to_idx, f)

		# Free some memory
		del train_dataset
		del train_docs

		dev_dataset, word_to_idx, tag_counter_dev = process_dataset(dev_docs, word_to_idx=word_to_idx, label_to_idx=label_to_idx, min_freq_word=args.min_freq_word)
		dataloader_dev = get_data_loader(dev_dataset, args.eval_batch_size, False, use_rnn)
		# Free some memory
		del dev_dataset
		del dev_docs
		gc.collect()

	###########################################################################
	# TRAIN MODEL
	###########################################################################
		# Init model
		if args.pretrained_path:
			TextClassifier = MultiLabelTextClassifier.load(args.pretrained_path)
		else:
			TextClassifier = MultiLabelTextClassifier(args.model_name, word_to_idx, label_to_idx, label_map, path_log = args.log_path,
													  save_dir=args.save_dir, tensorboard_dir=args.tensorboard_dir,
													  min_freq_word=args.min_freq_word, word_to_idx_path=args.word_to_idx_path,
													  B_train=args.train_batch_size, word_encoder = args.word_encoder,
													  B_eval=args.eval_batch_size, weight_decay=args.weight_decay, lr=args.learning_rate)

			TextClassifier.init_model(args.embed_dim, args.word_hidden, args.sent_hidden, args.dropout, args.vector_cache,
									  word_encoder=args.word_encoder, sent_encoder=args.sent_encoder, pos_weight=pos_weight)

		# Train
		TextClassifier.train(dataloader_train, dataloader_dev, pos_weight,
							 num_epochs=args.num_train_epochs, eval_every=args.eval_every)

	###########################################################################
	# EVAL MODEL
	###########################################################################
	if args.do_eval:
		if args.do_train: # Load best model obtained during training
			TextClassifier = MultiLabelTextClassifier.load(TextClassifier.pretrained_path)
		else: # Use model checkpoint
			TextClassifier = MultiLabelTextClassifier.load(args.pretrained_path)


		# Load test dataset
		with open(test_path, 'rb') as f:
			test_docs = pickle.load(f)

		# get dataloader
		test_dataset, word_to_idx, _ = process_dataset(test_docs, word_to_idx=word_to_idx,
																		label_to_idx=label_to_idx,
																		min_freq_word=args.min_freq_word)

		dataloader_test = get_data_loader(test_dataset, args.eval_batch_size, True, use_rnn)

		TextClassifier.eval_dataset(dataloader_test)

