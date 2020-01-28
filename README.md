
##  Hierarchical Capsule Networks with Attention for Document Classification
Intro

## Contents
| Section | Description |
|-|-|
| [Theoretical background](#theoretical-background) | Theoretical building blocks of this repo |
| [Setup](#setup) | How to setup a working environment |

# Theoretical background
- Towards Scalable and reliable Capsule Networks
- Hierarchical Attention Networks
- ULMFiT
- Dynamic Routing Between Capsules

# Setup

[1] Install anaconda:
Instructions here: https://www.anaconda.com/download/

[2] Create virtual environment:
```
conda env create --name hcapsnet python=3.7
source activate hcapsnet
```
[3]
Install PyTorch (>1.1). Please refer to the [PyTorch installation page](https://pytorch.org/get-started/locally/) for the specifics for your platform.

[4] Clone the repository:
```
git clone https://github.com/mrvoh/HCapsNet.git
cd HCapsNet
```
[5] Install the requirements:
```
pip install -r requirements.txt
```

# How to use
Below all parameters/settings are discussed per respective topic. The total of parameters and a set of example values can be found in the config file, ```parameters.ini```. All parameters can be altered by either adjusting its value in the config file or by passing a value on the command line, e.g. ```python main.py --train_batch_size 32```.

## Data and preprocessing
### Parameters
```
[Data]
train_path = dataset/trec/train.pkl 				# Path with preprocessed documents for training
dev_path = dataset/trec/dev.pkl 					# Path with preprocessed documents for development
test_path = dataset/trec/test.pkl 					# Path with preprocessed documents for testing
preload_word_to_idx = false 						# Load or create new word mapping
word_to_idx_path = dataset/trec/stoi1.json			# Where to load/store word mapping
label_to_idx_path = dataset/trec/label_to_idx.json  # Where to load/store label mapping



[Preprocessing]
preprocess_all = false 			# Create new dataset
dataset_name = trec 			# Name of dataset to create
write_data_dir = dataset/trec 	    # Where to store created dataset
restructure_docs = true 		# Restructure text to be more equal in sequence lengths within doc
balance_dataset = true 			# only for parsing from sheet (csv/xlsx)
max_seq_len = 100 		  # cut-off value for restructuring docs
min_freq_word = 2 		# Minimal frequency for a word to be considered
percentage_train = 0.8	 		# Percentage of data to use for training
percentage_dev = 0.0 			# Percentage of data to use for development (sheet parsing only)
```

### Document model

### Adding custom datasets
## Training/evaluating
### Parameters
```
[Training]
do_train = true 				# Whether to train the model
do_eval = true 					# Whether to evaluate on test set
binary_class = false 			# Whether current classification problem is binary
train_batch_size = 16 			# Batch size to train on
eval_batch_size = 32 			# Batch size for evaluation
learning_rate = 0.0025 			# Learning rate
dropout = 0.3163139487152957 	# Dropout between sent encoder and doc encoder
num_train_epochs = 30 			# Number of epochs to train for
eval_every = 272 				# Evaluate on dev set after eval_every training updates
K = 1 							# DEPRECATED
weight_decay = 0.001 			# L2 weight regularization weight
label_value = 0.95 				# Label value for computing loss (binary classification only)

[Model]
model_name = Hcapsnet 	# Name of the model to use
word_encoder = gru 		# Name of word encoder to use
sent_encoder = gru 		# Name of sentence encoder to use
use_glove = false 		# DEPRECATED(?)
embed_dim = 300 		# Embedding dimension of word vectors
word_hidden = 100 		# Hidden size of word encoder
sent_hidden = 100 		# Hidden size of sentence encoder
```
### Logging
log file
class report
tensorboard


# Using FastText
- pretrained/custom word vecs

# Using ULMFiT (optional)
- Train custom language model
- Usage
- Document encodings

# Extending to other datasets
- Document model
- Custom parser

# Hyperparameter optimization
- Hyperopt
- How to use

# Logging
- Tensorboard
- Classification reports


