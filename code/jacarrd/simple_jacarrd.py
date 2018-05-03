# coding=utf-8
import pandas as pd
import sys
import jieba

def Jacarrd(vec1, vec2):
    '''
    vec1 and vec2 are vectors of splited sentences.
    '''
    up = float(len(set(vec1) & set(vec2)))
    down = float(len(set(vec1) | set(vec2)))
    return up/down


if __name__ == '__main__':
    data_path = sys.argv[1]
    res_path  = sys.argv[2]
    df = pd.read_csv(data_path, sep='\t', names=['id', 'q1', 'q2'])
    # df_test = df[['id', 'q1', 'q2']]
    # df_test.to_csv('../../data/test.csv', index=False, header=None, sep='\t')
    # print df.head()
    df['q1_list'] = df['q1'].apply(lambda x: [i for i in jieba.cut(x, cut_all=True, HMM=False) if len(i)])
    df['q2_list'] = df['q2'].apply(lambda x: [i for i in jieba.cut(x, cut_all=True, HMM=False) if len(i)])
    df['Jacarrd_res'] = df.apply(lambda x: Jacarrd(x['q1_list'], x['q2_list']), axis=1)
    df['predict'] = df['Jacarrd_res'].apply(lambda x: 1 if x >=0.6 else 0)
    df = df[['id', 'predict']]
    df.to_csv(res_path, index=False, header=None, sep='\t')
