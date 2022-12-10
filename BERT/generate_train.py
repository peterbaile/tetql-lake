# 1. data leakage: generate train, valid such that they all come from different databases (dev is just dev.json, no test.json)
# split ratio: 60-40
# 2-1. constrastive learning: use the top candidates from tf-idf as negative samples
# 2-2. more positive samples from combinations

import json
import argparse
from itertools import combinations
import pandas as pd

def generate_samples(q_idx_ls, q_data, neg_data, dbs_tables, dbs_table_names, dbs):
  all_data = []

  max_len = 0

  for q_idx in q_idx_ls:
    q = q_data[q_idx]

    q_question = q['question']
    q_db_id = q['db_id'].lower()
    q_table_labels = q['table_labels']
    q_column_labels = q['column_labels']

    # generate positive samples
    positive_samples = []
    positive_cnt = 0

    for q_table_idx in q_table_labels:
      # fetch all columns that belong to the matched table
      matched_columns = set()
      min_columns = set()
      for idx, column in enumerate(dbs_tables[q_db_id]['column_names']):
        if column[0] == q_table_idx:
          matched_columns.add(column[1])
        
        if idx != 0 and idx in q_column_labels:
          min_columns.add(column[1])
        
      difference_columns = list(matched_columns.difference(min_columns))
      difference_columns = difference_columns[:7]
      min_columns = list(min_columns)

      for l in range(len(difference_columns) + 1):
        for diff_col in list(combinations(difference_columns, l)):
          actual_cols = []
          actual_cols += diff_col
          actual_cols += min_columns

          text = f"{q_question} [SEP] {dbs_table_names[q_db_id][q_table_idx]} {' '.join(actual_cols)}"
          positive_samples.append([text, 1])
          positive_cnt += 1

          if len(text.split(' ')) > max_len:
            max_len = len(text.split(' '))
    
    # generate negative samples (same num as positive samples)
    negative_samples = []
    negative_cnt = 0

    for (db_id, tbl_idx) in neg_data[q_idx][:positive_cnt]:
      cols = ' '.join(dbs[(db_id, tbl_idx)])
      text = f'{q_question} [SEP] {dbs_table_names[db_id][tbl_idx]} {cols}'
      negative_samples.append([text, 0])

      if len(text.split(' ')) > max_len:
        max_len = len(text.split(' '))
    
    all_data += positive_samples + negative_samples
  
  print(f'max len is {max_len}')
  return all_data

if __name__ == '__main__':
  # parser = argparse.ArgumentParser()
  # parser.add_argument('file', type=str)
  # args = parser.parse_args()
  # print(f'source file: {args.file}')

  with open('../spider_data/tables.json') as f:
    db_data = json.load(f)
  
  dbs_tables = {}
  dbs_table_names = {}
  dbs = {}
  for db in db_data:
    db_id = db['db_id'].lower()
    dbs_tables[db_id] = db
    dbs_table_names[db_id] = db['table_names']

    for i in range(len(db['table_names'])):
      dbs[(db_id, i)] = []

    for col in db['column_names']:
      if col[0] >= 0:
        dbs[(db_id, col[0])].append(col[1])

  with open('../spider_data/train_spider_new.json', 'r') as f:
    q_data = json.load(f)
  
  with open('./data/train_spider/train_spider_idf_200.json', 'r') as f:
    raw_neg_data = json.load(f) #(db_id, table_index)
  
  neg_data = []
  for idx, cands in enumerate(raw_neg_data):
    neg_data.append([])
    for cand in cands:
      if not (cand[0] == q_data[idx]['db_id'].lower() and cand[1] in q_data[idx]['table_labels']):
        neg_data[idx].append(cand)
  
  # get count of how many questions per database and sort and split
  db_q_count = {}
  db_q = {}

  for idx, q in enumerate(q_data):
    if len(q['table_labels']) >= 2:
      continue
  
    q_db_id = q['db_id'].lower()
    if q_db_id not in db_q_count:
      db_q_count[q_db_id] = 1
      db_q[q_db_id] = [idx]
    else:
      db_q_count[q_db_id] += 1
      db_q[q_db_id].append(idx)

  db_q_count = dict(sorted(db_q_count.items(), key = lambda x: x[1], reverse = True))

  num_q = sum(db_q_count.values())

  train_len = int(0.6 * num_q)
  valid_len = num_q - train_len

  print(f'train_len, valid_len: {train_len, valid_len}')

  train_count = 0
  train_q = []
  valid_count = 0
  valid_q = []
  for db_id in db_q_count:
    if train_count < train_len:
      train_count += db_q_count[db_id]
      train_q += db_q[db_id]
    else:
      valid_count += db_q_count[db_id]
      valid_q += db_q[db_id]
  
  print(len(train_q), len(valid_q))

  train_data = []

  train_data = generate_samples(train_q, q_data, neg_data, dbs_tables, dbs_table_names, dbs)
  valid_data = generate_samples(valid_q, q_data, neg_data, dbs_tables, dbs_table_names, dbs)

  print(len(train_data))
  print(len(valid_data))

  # train_df = pd.DataFrame(train_data, columns=['text', 'label'])
  # valid_df = pd.DataFrame(valid_data, columns=['text', 'label'])
  # train_df.to_csv('./data/train_spider/train.csv', index=False)
  # valid_df.to_csv('./data/train_spider/valid.csv', index=False)
