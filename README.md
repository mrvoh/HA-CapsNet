
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
[DataAndPreprocessing]
preprocess_all = false
train_path = dataset/trec/train.pkl
dev_path = dataset/trec/dev.pkl
test_path = dataset/trec/test.pkl
write_data_dir = dataset/trec
word_vec_path = word vectors/wiki.en/wiki.en.bin
preload_word_to_idx = false
word_to_idx_path = dataset/trec/stoi1.json
label_to_idx_path = dataset/trec/label_to_idx.json
min_freq_word = 2
dataset_name = trec
max_seq_len = 100
percentage_train = 0.8
percentage_dev = 0.0
```

### Document model

### Adding custom datasets
## Training/evaluating
### Parameters
```
[Training]
binary_class = false
train_batch_size = 16
eval_batch_size = 32
learning_rate = 0.0025
dropout = 0.3163139487152957
num_train_epochs = 30
eval_every = 272
K = 1 # DEPRECATED
weight_decay = 2.43624658089963e-05
label_value = 0.9680114815754679

[Model]
model_name = Hcapsnet
word_encoder = gru
sent_encoder = gru
use_glove = false
embed_dim = 300
word_hidden = 100
sent_hidden = 100
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


