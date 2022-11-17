# 1. data leakage: generate train, valid, test, dev such that they all come from different databases
# split ratio: 60-20-20
# 2-1. constrastive learning: use the top candidates from tf-idf as negative samples
# 2-2. more positive samples from combinations

import json
import argparse

if __name__ == '__main__':
  # parser = argparse.ArgumentParser()
  # parser.add_argument('file', type=str)
  # args = parser.parse_args()
  # print(f'source file: {args.file}')  

  with open('../spider_data/train_spider_new.json', 'r') as f:
    q_data = json.load(f)
  
  # get count of how many questions per database and sort and split
  db_q_count = {}
  db_q = {}

  for idx, q in enumerate(q_data):
    q_db_id = q['db_id']
    if q_db_id not in db_q_count:
      db_q_count[q_db_id] = 1
      db_q[q_db_id] = [idx]
    else:
      db_q_count[q_db_id] += 1
      db_q[q_db_id].append(idx)

  db_q_count = dict(sorted(db_q_count.items(), key = lambda x: x[1], reverse = True))

  num_q = len(q_data)

  train_len = int(0.6 * num_q)
  valid_len = int(0.2 * num_q)
  test_len = num_q - train_len - valid_len

  print(train_len, valid_len, test_len)

  train_count = 0
  train_q = []
  valid_count = 0
  valid_q = []
  test_q = []
  for db_id in db_q_count:
    if train_count < train_len:
      train_count += db_q_count[db_id]
      train_q += db_q[db_id]
    elif valid_count < valid_len:
      valid_count += db_q_count[db_id]
      valid_q += db_q[db_id]
    else:
      test_q += db_q[db_id]
  
  print(len(train_q), len(valid_q), len(test_q))

  
  # print(db_q_count)

  # with open('../spider_data/tables.json') as f:
  #   db_data = json.load(f)
  
  # for db in db_data:
  # db_id = db['db_id']
  # num_tables = len(db['table_names'])

  # dbs_table_names[db_id] = db['table_names']

  # dbs_tables[db_id] = db

  # table_cols = {}

  # for i in range(num_tables):
  #   table_cols[i] = []

  # for col in db['column_names']:
  #   if col[0] >= 0:
  #     table_cols[col[0]].append(col[1])
    
  # for table_index in table_cols:
  #   dbs.append((db_id, db['table_names'][table_index], table_cols[table_index]))
