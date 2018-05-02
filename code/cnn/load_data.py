# coding = utf-8
import os
import pandas as pd
import numpy as np
import re
import random
import tarfile
from torchtext import data
from datetime import datetime
import traceback  
import torchtext.datasets as datasets
import pickle
from gensim.models import Word2Vec
import jieba

jieba.load_userdict('../../data/special_word.txt')

data_path      = '../../data/atec_nlp_sim_train.csv'
train_path     = '../../data/train.tsv'
dev_path       = '../../data/dev.tsv'
w2v_model_path = '../../data/w2v_train.save'

def preprocess(data_path):
    '''
    convert Chinese sentences to word lists
    '''
    df = pd.read_csv(data_path, sep='\t', names=['id', 'q1', 'q2', 'label'])
    df['q1_list'] = df['q1'].apply(lambda x: [i for i in jieba.cut(x, cut_all=True, HMM=False) if len(i)])
    df['q2_list'] = df['q2'].apply(lambda x: [i for i in jieba.cut(x, cut_all=True, HMM=False) if len(i)])
    return df

def train_word2vec_model(df):
    '''
    basic w2v model trained by sentences
    '''
    corpus = []
    for i, r in df.iterrows():
        corpus.append(r['q1_list'])
        corpus.append(r['q2_list'])
    word2vec_model = Word2Vec(corpus, size=100, window=3, min_count=3, sg=1)
    return word2vec_model


def load_glove_as_dict(filepath):
    word_vec = {}
    with open(filepath) as fr:
        for line in fr:
            line = line.split()
            word = line[0]
            vec = line[1:]
            word_vec[word] = vec
    return word_vec

def gen_iter(path, text_field, label_field, args):
    '''
        Load TabularDataset from path,
        then convert it into a iterator
        return TabularDataset and iterator
    '''
    tmp_data = data.TabularDataset(
                            path=path,
                            format='csv',
                            skip_header=True,
                            fields=[
                                    ('question1', text_field),
                                    ('question2', text_field),
                                    ('label', label_field)
                                    ])

    tmp_iter = data.BucketIterator(
                    tmp_data,
                    batch_size=args.batch_size,
                    sort_key=lambda x: len(x.question1) + len(x.question2),
                    device=0, # 0 for GPU, -1 for CPU
                    repeat=False)
    return tmp_data, tmp_iter

def load_data(args):
    '''
        1. train the w2v_model
        2. split the data as 9:1(train:dev)
        3. load the data
        load as pairs
    '''
    df = preprocess(data_path)
    word2vec_model = train_word2vec_model(df)
    word2vec_model.save(w2v_model_path)

    df       = df[['q1_list', 'q2_list', 'label']]
    df['q1_list'] = df['q1_list'].apply(lambda x: ' '.join(x))
    df['q2_list'] = df['q2_list'].apply(lambda x: ' '.join(x))
    train_df = df.head(int(len(df)*0.9))
    dev_df   = df.tail(int(len(df)*0.1))
    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)

    
    text_field    = data.Field(sequential=True, use_vocab=True, batch_first=True, lower=True)
    label_field   = data.Field(sequential=False, use_vocab=False)
    
    train_data, train_iter = gen_iter(train_path, text_field, label_field, args)
    dev_data, dev_iter     = gen_iter(dev_path, text_field, label_field, args)
    
    text_field.build_vocab(train_data, dev_data)

    return text_field, label_field, \
        train_data, train_iter,\
        dev_data, dev_iter
          
