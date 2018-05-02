import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import pickle

DATA_PATH = '../data/split_result.csv'

def train_word2vec_model(DATA_PATH):
    '''
    basic w2v model trained by sentences
    '''
    df = pd.read_csv(DATA_PATH)
    df['wordList1'] = df['wordList1'].apply(lambda x: x.split('#'))
    df['wordList2'] = df['wordList2'].apply(lambda x: x.split('#'))

    corpus = []
    for i, r in df.iterrows():
        corpus.append(r['wordList1'])
        corpus.append(r['wordList2'])
    word2vec_model = Word2Vec(corpus, size=100, window=3, min_count=1, sg=1)
    return word2vec_model
    