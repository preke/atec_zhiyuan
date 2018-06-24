# coding = utf-8
import os
import pandas as pd
import numpy as np
import torch
import torch.autograd as autograd
import re
import random
from env.torchtext import data
from datetime import datetime
import traceback  
import env.torchtext.datasets as datasets
import pickle
from gensim.models import Word2Vec
import jieba
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer 

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


def get_tfidf_weighted_embedding(df, word2vec_model, mode):
    corpus = []
    for i, r in df.iterrows():
        try:
            corpus.append(' '.join(jieba.lcut(r['ques1'])))
            corpus.append(' '.join(jieba.lcut(r['ques2'])))
        except:
            print 'Exception: ', r['ques1']
    
    vectorizer  = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf       = transformer.fit_transform(vectorizer.fit_transform(corpus))
    word        = vectorizer.get_feature_names()
    weight      = tfidf.toarray()
    
    tfidf_weighted_embeddings = []
    for i in range(len(weight)): 
        temp_embedding = np.array([0.0]*300)
        for j in range(len(word)):  
            if word[j] in corpus[i]:
                temp_embedding += np.array(word2vec_model[word[j]]) * weight[i][j]
        tfidf_weighted_embeddings.append(temp_embedding)
    indexs = list(df['id'])
    if mode != 'test':
        labels = list(df['label'])
    tfidf_weighted_embeddings_pairs = []
    cnt = 0
    for i in range(len(corpus)):
        if i % 2 == 0:
            temp_list = []
            temp_list.append(indexs[cnt])
            temp_list.append(tfidf_weighted_embeddings[i])
            temp_list.append(tfidf_weighted_embeddings[i+1])
            if mode != 'test':
                temp_list.append(labels[cnt])
            cnt += 1
            tfidf_weighted_embeddings_pairs.append(temp_list)
    
    if mode != 'test':
        tfidf_weighted_embeddings_pairs = pd.DataFrame(tfidf_weighted_embeddings_pairs, columns=['id', 'ebd1', 'ebd2', 'label'])
    else:
        tfidf_weighted_embeddings_pairs = pd.DataFrame(tfidf_weighted_embeddings_pairs, columns=['id', 'ebd1', 'ebd2'])

    # print tfidf_weighted_embeddings_pairs.head(1)
    tfidf_weighted_embeddings_pairs.to_csv('data/'+ mode + '_tfidf_weighted_embeddings_pairs.tsv', header=None, sep='\t', index=False)

def load_glove_as_dict(filepath):
    word_vec = {}
    with open(filepath) as fr:
        for line in fr:
            line = line.split()
            word = line[0]
            vec = line[1:]
            word_vec[word] = vec
    return word_vec

def preprocess(_string):
    _string = re.sub('\n', ' ', _string)
    _string = _string[1:-1]
    _string = [float(i) for i in _string.split()]
    _string = torch.cuda.FloatTensor(_string)
    _string = autograd.Variable(_string)
    return _string



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
    # sprint tmp_data
    tmp_iter = data.BucketIterator(
                    dataset=tmp_data,
                    batch_size=args.batch_size,
                    device=0, # 0 for GPU, -1 for CPU
                    # sort_key=lambda x: len(x.question1) + len(x.question2),
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
                    device=1, # 0 for GPU, -1 for CPU
                    shuffle=False,
                    repeat=False)
    return tmp_data, tmp_iter


def load_data(args):
    # train woed2vec model
    # *****************
    # df = pd.read_csv('data/atec_nlp_sim_train.tsv', names=['id', 'ques1', 'ques2', 'label'], sep='\t')
    # word2vec_model = train_word2vec_model(df)
    # df_train = pd.read_csv('data/train_3000.tsv', names=['id', 'ques1', 'ques2', 'label'], sep='\t')
    # get_tfidf_weighted_embedding(df_train, word2vec_model, mode='train')

    # df_dev = pd.read_csv('data/valid_3000.tsv', names=['id', 'ques1', 'ques2', 'label'], sep='\t')
    # get_tfidf_weighted_embedding(df_dev, word2vec_model, mode='valid')

    # df_test = pd.read_csv('data/test_3000.tsv', names=['id', 'ques1', 'ques2'], sep='\t')
    # get_tfidf_weighted_embedding(df_test, word2vec_model, mode='test')
    
    # word2vec_model.save(w2v_model_path)

    # *****************
    
    text_field   = data.Field(sequential=False, use_vocab=False, batch_first=True)
    text_field.preprocessing = data.Pipeline(preprocess)
    label_field   = data.Field(sequential=False, use_vocab=False, batch_first=True)
    
    train_data, train_iter = gen_iter(args.train_path, text_field, label_field, args)
    dev_data, dev_iter     = gen_iter(args.dev_path, text_field, label_field, args)
    test_data, test_iter   = gen_iter_test(args.test_path, text_field, label_field, args)
    

    return text_field, label_field, \
        train_data, train_iter,\
        dev_data, dev_iter,\
        test_data, test_iter
          
