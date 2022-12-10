from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import pandas as pd
import json

# only for single table for now

if __name__ == '__main__':
  devfile = 'dev_em_2'

  # get BERT output
  # num_parts = 3

  # dev_outputs = [np.load(f'./data/dev/{args.devfile}_output_{i}_ranking.npy') for i in range(num_parts)]

  # for i in range(num_parts):
  #   output = np.hstack(dev_outputs)

  output = np.load(f'./data/dev/{devfile}_output_-1_ranking.npy')
  
  output = list(output)

  # get dev file
  dev_output_df = pd.read_csv(f'./data/dev/{devfile}_ranking.csv')

  assert(len(output) == dev_output_df.shape[0])

  dev_output_df = dev_output_df[pd.Series(output).astype(bool)]

  print(dev_output_df)

  pred = []
  label = []

  with open('../spider_data/dev_new.json') as f:
    qs = json.load(f)

  with open('../spider_data/tables.json') as f:
    dbs = json.load(f)

  # get all tables and generate matches
  # db_id, tbl_index

  idx = 0
  for q in qs:
    if len(q['table_labels']) >= 2:
      continue

    _output = dev_output_df.iloc[idx]

    # print(_output)

    for db in dbs:
      for t in range(len(db['table_names'])):
        if db['db_id'] == q['db_id'] and t in q['table_labels']:
          label.append(1)
          
        else:
          label.append(0)
        
        if _output['db_id'] == db['db_id'] and _output['table_index'] in q['table_labels'] and t == _output['table_index']:
          pred.append(1)
        else:
          pred.append(0)

    idx += 1
  
  print(len(pred))
  assert(len(pred) == len(label))
  
  # compare with F1
  print(f'precision: {precision_score(label, pred):.3f}')
  print(f'recall: {recall_score(label, pred):.3f}')
  print(f'f1: {f1_score(label, pred):.3f}')
  print(f'accuracy: {accuracy_score(label, pred):.3f}')