# coding = utf-8
import os
import pandas as pd
import numpy as np
import re
import random
from env.torchtext import data
from datetime import datetime
import traceback  
import env.torchtext.datasets as datasets
import pickle
from gensim.models import Word2Vec
import jieba

jieba.load_userdict('data/special_word.txt')
w2v_model_path = 'data/w2v_train.save'

def train_word2vec_model(df):
    '''
    basic w2v model trained by sentences
    '''
    corpus = []
    for i, r in df.iterrows():
        try:
            corpus.append(jieba.lcut(r['ques1']))
            # print jieba.lcut(r['ques1'])
            corpus.append(jieba.lcut(r['ques2']))
        except:
            print 'Exception: ', r['ques1']
    word2vec_model = Word2Vec(corpus, size=300, window=3, min_count=1, sg=0, iter=100)
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
                            format='tsv',
                            skip_header=False,
                            fields=[
                                    ('id', None),
                                    ('question1', text_field),
                                    ('question2', text_field),
                                    ('label', label_field)
                                    ])

    tmp_iter = data.BucketIterator(
                    dataset=tmp_data,
                    batch_size=args.batch_size,
                    device=-1, # 0 for GPU, -1 for CPU
                    sort_key=lambda x: len(x.question1) + len(x.question2),
                    shuffle=False,
                    repeat=False)
    return tmp_data, tmp_iter


def gen_iter_test(path, text_field, label_field, args):
    '''
        Load TabularDataset from path,
        then convert it into a iterator
        return TabularDataset and iterator
    '''
    tmp_data = data.TabularDataset(
                            path=path,
                            format='tsv',
                            skip_header=False,
                            fields=[
                                    ('id', label_field),
                                    ('question1', text_field),
                                    ('question2', text_field)
                                    ])

    tmp_iter = data.Iterator(
                    dataset=tmp_data,
                    batch_size=args.batch_size,
                    device=-1, # 0 for GPU, -1 for CPU
                    shuffle=False,
                    repeat=False)
    return tmp_data, tmp_iter


def load_data(args):
    # train woed2vec model
    # *****************
    # df_train = pd.read_csv('data/train_3000.tsv', names=['id', 'ques1', 'ques2', 'label'], sep='\t')
    # word2vec_model = train_word2vec_model(df_train)
    # word2vec_model.save(w2v_model_path)
    # *****************
    
    text_field    = data.Field(sequential=True, use_vocab=True, 
                    batch_first=True, eos_token='<EOS>', init_token='<BOS>', pad_token='<PAD>', tokenize=jieba.lcut)
    label_field   = data.Field(sequential=False, use_vocab=False, batch_first=True)
    
    train_data, train_iter = gen_iter('data/train_3000.tsv', text_field, label_field, args)
    dev_data, dev_iter     = gen_iter('data/valid_3000.tsv', text_field, label_field, args)
    test_data, test_iter   = gen_iter_test(args.test_path, text_field, label_field, args)
    text_field.build_vocab(train_data, dev_data)

    return text_field, label_field, \
        train_data, train_iter,\
        dev_data, dev_iter,\
        test_data, test_iter
          
