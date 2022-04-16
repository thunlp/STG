# STG
Official code and data of the Findings of ACL 2022 paper ["Going 'Deeper': Structured Sememe Prediction via Transformer with Tree Attention"](https://openreview.net/pdf?id=Zs5o6hwEhFE)


## Overview
    
Sememe knowledge bases (SKBs), which annotate words with the smallest semantic units (i.e., sememes), have proven beneficial to many NLP tasks. Building an SKB is very timeconsuming and labor-intensive. Therefore, some studies have tried to automate the building process by predicting sememes for the unannotated words. However, all existing sememe prediction studies ignore the hierarchical structures of sememes, which are important in the sememe based semantic description system. In this work, we tackle the structured sememe prediction problem for the first time, which is aimed at predicting a sememe tree with hierarchical structures rather than a set of sememes. We design a sememe tree generation model based on Transformer with an adjusted attention mechanism, which shows its superiority over the baseline methods in experiments. We also conduct a series of quantitative and qualitative analyses of the effectiveness of our model.

## Data

- we use snetence-BERT for cosine similarity computaion in  NSTG, and it is avaliable in [here](https://github.com/UKPLab/sentence-transformers).

- we use pretrained BERT-base as our NL-reader encoder, and it is avaliable [here](https://github.com/google-research/bert).

- we use the pretrained sememe embeddings in SE-WRL, which is avaliable [here](https://github.com/thunlp/SE-WRL-SAT).

- we use BabelNet as our dataset.


## Dependenices

the code is done in python. To run the training and testing process, you need the following modules:
```python
anytree==2.8.0
demjson==2.2.4
matplotlib==3.3.4
nltk==3.5
numpy==1.20.1
scikit_image==0.16.2
sentence_transformers==2.2.0
skimage==0.0
torch==1.8.1
torchvision==0.9.1
tqdm==4.56.0
transformers==3.0.2
treelib==1.6.1
```

You also need to put the pretrianed BERT-base in `BERT_PATH`

##Usage

You can take a easy tour of our repo in `./look_project.ipynb`, which shows some data structure and methods of our project. 

#### preprocess

To preprocess the data from BabelNet, first you need to make structured-sememe-synset-dataset avaliable, which you can see part in `./data/synsetDef.txt` and `./data/synsetStructed.txt`.
And then you can run the following code:

```sh

python create_data.py
python preprocess.py
```
#### train the model
to train a new model, you can run `train.sh`, where we provide to train in normal mode, NSTG mode and TaSTG mode.
```sh
sh train.sh NSTG
sh train.sh TSTG
sh train.sh TaSTG
```

Also, you can train in your settings, use `train.py`

```sh
usage: train.py [-h] [--learning_rate LEARNING_RATE] [--max_epoch MAX_EPOCH]
                [--pretrained PRETRAINED] [--mask MASK] [--tree_attention TREE_ATTENTION]
                [--depth_method DEPTH_METHOD] [--bias_method BIAS_METHOD] [--sequence SEQUENCE]
                [--train_set_path TRAIN_SET_PATH] [--test_set_path TEST_SET_PATH]
                [--valid_set_path VALID_SET_PATH] [--model_save_path MODEL_SAVE_PATH]
                [--model_name MODEL_NAME]

args is defined bellow:

optional arguments:
  -h, --help            show this help message and exit
  --learning_rate LEARNING_RATE
                        learning rate
  --max_epoch MAX_EPOCH
                        max training epochs
  --pretrained PRETRAINED
                        use pretrained sememe embeddings
  --mask MASK           use candidate mask
  --tree_attention TREE_ATTENTION
                        use tree attention method
  --depth_method DEPTH_METHOD
  --bias_method BIAS_METHOD
  --sequence SEQUENCE   use sequence encoding result
  --train_set_path TRAIN_SET_PATH
  --test_set_path TEST_SET_PATH
  --valid_set_path VALID_SET_PATH
  --model_save_path MODEL_SAVE_PATH
  --model_name MODEL_NAME
                        how to name your model
```


#### predict the model

After training, you can test your model:

```bash
python test.py
```

## Citation
If these data and codes help you, please cite this paper.


