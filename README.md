# Test Bert on CLOTH Dataset

This repository test [the Bert model](https://github.com/google-research/bert) on the CLOTH Dataset. The framework of implementation is provided by [the Pytorch Pretrained Bert repo](https://github.com/huggingface/pytorch-pretrained-BERT). 

The finetuned Bert model reach 86.0% accuracy on the test set, which is the same as the performance of Amazon Turkers. [Leaderboard](http://www.qizhexie.com/data/CLOTH_leaderboard)

## Experiment Environment

The code is tested with Python3.6 and Pytorch 0.4.0

## Code Usage

### Download Dataset

You can require the dataset from [our website](http://www.cs.cmu.edu/~glai1/data/cloth/), and decompress it in the root of this repo.

### Preprocessing Data

First step is to preprocess data by running

```
python data_util.py
```

You should assign the name of the Bert model that you want to use in the file. It is "bert-large-uncased" by default. 

### Finetune Model 

You can run the model with the best hyper-parameter setting by

```
./run.sh
```