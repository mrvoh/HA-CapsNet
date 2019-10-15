
##  Hierarchical Capsule Networks with Attention for Document Classification
TODO:
* Refactor document_model
* Refactor hyperparameters to configuration file
* FastText autotune + support model HCapsNet
* Proper documentation

### Notebook

The notebook contains an example of trained model on IMDB movie review dataset. I could not get the original IMDB dataset that the paper referred to, so I have used [this data](http://ir.hit.edu.cn/~dytang/paper/acl2015/dataset.7z)

The preprocessed data is available [here](https://drive.google.com/file/d/0B1RmSd_tWx4CUnFQaXdvV2R1NFk/view?usp=sharing)

The best accuracy that I got was around ~ 0.35. This dataset has only 84919 samples and 10 classes. Here is the training loss for the dataset. 

![alt text](imdb_data_attn.png "Document Classification")
