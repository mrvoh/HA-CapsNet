
##  Hierarchical Capsule Networks with Attention for Document Classification
Intro

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

# How to use (main.py)
- Data & Preprocessing
- Training
- Evaluation
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


