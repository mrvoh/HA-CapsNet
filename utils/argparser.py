import configargparse
import os

def get_parser():
	parser = configargparse.ArgParser(default_config_files=['parameters.ini'])
	#  MAIN ARGUMENTS
	parser.add_argument('-c', '--my-config',
						required=False,
						is_config_file=True,
						help='config file path')
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
						help="The path from where the pretrained model is to be retrieved.")

	#  TRAIN/EVAL ARGS
	parser.add_argument("--train_batch_size",
						required=True,
						type=int,
						help="Total batch size for training.")
	parser.add_argument("--eval_batch_size",
						required=True,
						type=int,
						help="Total batch size for eval.")
	parser.add_argument("--label_value",
						required=True,
						type=float,
						help="Ground truth label value against which to compute loss.")
	parser.add_argument("--learning_rate",
						required=True,
						type=float,
						help="The initial learning rate for Adam.")
	parser.add_argument("--dropout",
						required=True,
						type=float,
						help="The initial learning rate for RAdam.")
	parser.add_argument("--num_train_epochs",
						required=True,
						type=int,
						help="Total number of training epochs to perform.")
	parser.add_argument("--eval_every",
						required=True,
						type=int,
						help="Nr of training updates before evaluating model")
	parser.add_argument("--K",
						required=True,
						type=int,
						help="Cut-off value for evaluation metrics")
	parser.add_argument("--weight_decay",
						required=True,
						type=float,
						help="L2 regularization term.")

	#  MODEL ARGS
	parser.add_argument("--model_name",
						type=str,
						required=True,
						help="Model to use. Options: HAN, HGRULWAN, HCapsNet & HCapsNetMultiHeadAtt")
	parser.add_argument("--binary_class",
						action='store_false',
						help="Whether model dataset as multi-class classification(cross-entropy based) or multi-label classification (multiple binary classification).")
	parser.add_argument("--use_glove",
						action='store_true',
						help="Whether to utilize additional GloVe embeddings next to FastText.")
	parser.add_argument("--word_encoder",
						type=str,
						required=True,
						help="The path from where the class-names are to be retrieved.")
	parser.add_argument("--sent_encoder",
						type=str,
						required=True,
						help="The path from where the class-names are to be retrieved.")
	parser.add_argument("--embed_dim",
						type=int,
						help="Nr of dimensions of used word vectors")
	parser.add_argument("--word_hidden",
						type=int,
						help="Nr of hidden units word encoder. If GRU is used as encoder, the nr of hidden units is used for BOTH forward and backward pass, resulting in double resulting size.")
	parser.add_argument("--sent_hidden",
						type=int,
						help="Nr of hidden units sentence encoder. If GRU is used as encoder, the nr of hidden units is used for BOTH forward and backward pass, resulting in double resulting size.")

	# ulmfit options
	parser.add_argument("--ulmfit_pretrained_path",
						default=None,
						type=str,
						required=False,
						help="The path from where the pretrained language model is to be retrieved.")
	parser.add_argument("--dropout_factor",
						type=float,
						help="Multiplier for standard dropout values in ULMFiT.")
	parser.add_argument("--gradual_unfreeze",
						action='store_true',
						help="Unfreeze ULMFiT one layer per epoch.")
	parser.add_argument("--keep_frozen",
						action='store_true',
						help="Keep ULMFiT weights static over course of training.")
	# caps net options
	parser.add_argument("--dim_caps",
						type=int,
						help="Nr of dimensions of a capsule")
	parser.add_argument("--num_head_doc",
						default=5,
						type=int,
						help="Nr of dimensions of a capsule")
	parser.add_argument("--num_caps",
						type=int,
						help="Number of capsules in Primary Capsule Layer")
	parser.add_argument("--num_compressed_caps",
						type=int,
						help="Number of compressed capsules")
	parser.add_argument("--dropout_caps",
						type=float,
						help="Dropout to apply within CapsNet")
	parser.add_argument("--lambda_reg_caps",
						type=float,
						help="Penalty weight for reconstruction loss")
	#	DATA & PRE-PROCESSING ARGS
	parser.add_argument("--preprocess_all",
						action='store_true',
						help="Whether pre-process the dataset from a set of *.json files to a loadable dataset.")
	parser.add_argument("--create_doc_encodings",
						action='store_true',
						help="Whether to enrich the documents with an encoding for reconstruction.")

	parser.add_argument("--raw_data_dir",
						default=r'C:\Users\nvanderheijden\Documents\Regminer\regminer-topic-modelling\modeling\train_filtered.xlsx',
						type=str,
						required=False,
						help="Dir from where to parse the data.")
	parser.add_argument("--num_tags",
						default=100,
						type=int,
						help="Number of labels to use from the data (filters top N occurring)")
	parser.add_argument("--max_seq_len",
						default=100,
						type=int,
						help="Number of labels to use from the data (filters top N occurring)")
	parser.add_argument("--restructure_docs",
						action='store_false',
						help="Whether to restructure docs such that sentences are split/combined to evenly spread words over sequences.")
	parser.add_argument("--dataset_name",
						default='imdb',
						type=str,
						required=False,
						help="Name of the dataset.")
	parser.add_argument("--percentage_train",
						default=1.0,
						type=float,
						help="Percentage of train set to actually use for training when no train/dev/test split is given in data.")
	parser.add_argument("--percentage_dev",
						default=0.0,
						type=float,
						help="Percentage of train set to actually use for training when no train/dev/test split is given in data.")
	parser.add_argument("--write_data_dir",
						default='dataset\\imdb-full',
						type=str,
						required=False,
						help="Where to write the parsed data to.")
	parser.add_argument("--num_backtranslations",
						default=1,
						type=int,
						help="Number of times to backtranslate each document as means of data augmentation. Set to None for no backtranslation.")

	parser.add_argument("--train_path",
						type=str,
						required=True,
						help="The path from where the train dataset is to be retrieved.")
	parser.add_argument("--dev_path",
						required=True,
						type=str,
						help="The path from where the dev dataset is to be retrieved.")
	parser.add_argument("--test_path",
						type=str,
						required=False,
						help="The path from where the dev dataset is to be retrieved.")
	parser.add_argument("--create_wordvecs",
						action='store_true',
						help="Whether to create custom word vectors using FastText.")
	parser.add_argument("--word_vec_path",
						type=str,
						required=True,
						help="Folder where vector caches are stored.")
	parser.add_argument("--preload_word_to_idx",
						action='store_true',
						help="Whether to use an existing word to idx mapping.")
	parser.add_argument("--word_to_idx_path",
						type=str,
						required=True,
						help="The path from where the word mapping is to be retrieved.")
	parser.add_argument("--label_to_idx_path",
						default=os.path.join('dataset', 'reuters', 'label_to_idx.json'),
						type=str,
						required=True,
						help="The path from where the label mapping is to be retrieved.")
	parser.add_argument("--label_map_path",
						type=str,
						required=False,
						help="The path from where the mapping from label id to label description is loaded.")
	parser.add_argument("--min_freq_word",
						type=int,
						help="Minimum nr of occurrences before being assigned a word vector")

	#	OTHER ARGS
	parser.add_argument("--use_fasttext_baseline",
						action='store_true',
						help="Whether to use a FastText model for baseline purposes or not.")
	parser.add_argument("--autotune_time_fasttext",
						default=None,
						type=int,
						help="How many seconds to optimize fasttext hyperparams. Set to None to not perform autotuning")
	parser.add_argument("--fasttext_save_path",
						default=os.path.join('models', 'fasttext.model'),
						type=str,
						required=False,
						help="The path where to dump logging.")

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
	return args, parser