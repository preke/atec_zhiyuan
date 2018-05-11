import pandas as pd

train_df = pd.read_csv('atec_nlp_sim_train.csv', sep='\t', names=['id', 'q1', 'q2', 'label'])
test_df = train_df[['id', 'q1', 'q2']]
test_df.to_csv('test.csv', index=False, header=None, sep='\t')
