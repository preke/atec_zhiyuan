# coding=utf-8
import pandas as pd
import numpy as np



'''
results:

Threshold is 0.1, Accuracy is 0.233950083871
Threshold is 0.2, Accuracy is 0.352767752758
Threshold is 0.3, Accuracy is 0.607126518579
Threshold is 0.4, Accuracy is 0.750851420729
Threshold is 0.5, Accuracy is 0.785391145224
Threshold is 0.6, Accuracy is 0.789737203274
Threshold is 0.7, Accuracy is 0.786585675799
Threshold is 0.8, Accuracy is 0.784094952473
Threshold is 0.9, Accuracy is 0.782824175266

'''

DATA_PATH = '../data/split_result.csv'

def Jacarrd(vec1, vec2):
    '''
    vec1 and vec2 are vectors of splited sentences.
    '''
    up = float(len(set(vec1) & set(vec2)))
    down = float(len(set(vec1) | set(vec2)))
    return up/down


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)
    df['wordList1'] = df['wordList1'].apply(lambda x: x.split('#'))
    df['wordList2'] = df['wordList2'].apply(lambda x: x.split('#'))

    df['Jacarrd_res'] = df.apply(lambda x: Jacarrd(x['wordList1'], x['wordList2']), axis=1)
    df['wordList1'] = df['wordList1'].apply(lambda x: [str(i).encode('utf-8') for i in x])
    df['wordList2'] = df['wordList2'].apply(lambda x: [str(i).encode('utf-8') for i in x])
    df.to_csv('../data/jacarrd_res.csv', index=False)
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    length = len(df)
    for threshold in thresholds:
        tmp_acc = 0
        for i, r in df.iterrows():
            if r['is_duplicated'] == 1 and r['Jacarrd_res'] >= threshold:
                tmp_acc += 1
            elif r['is_duplicated'] == 0 and r['Jacarrd_res'] < threshold:
                tmp_acc += 1
        print('Threshold is %s, Accuracy is %s' %(str(threshold), str(float(tmp_acc)/length)))
