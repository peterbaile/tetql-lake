import json
import argparse
from itertools import combinations
import pandas as pd

def generate_samples(q_idx_ls, q_data, neg_data, dbs_tables, dbs_table_names, dbs, add_negative):
  all_data = []

  for q_idx in q_idx_ls:
    q = q_data[q_idx]

    q_question = q['question']
    q_db_id = q['db_id'].lower()
    q_table_labels = q['table_labels']
    q_column_labels = q['column_labels']

    tbl_texts = []

    for q_table_idx in q_table_labels:
      cols = ','.join(dbs[(q_db_id, q_table_idx)])
      tbl_text = f'{q_db_id},{dbs_table_names[q_db_id][q_table_idx]},{cols}'

      tbl_texts.append(tbl_text)

    if add_negative:
      neg_key = tuple(neg_data[q_idx][0])
      cols = ','.join(dbs[neg_key])
      tbl_text = f'{dbs_table_names[neg_key[0]][neg_key[1]]},{cols}'

      tbl_texts.append(tbl_text)
    
    all_data.append([q_question, '#sep#'.join(tbl_texts)])
  
  return all_data

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--negative', type=bool, default=False)
  args = parser.parse_args()
  print(f'add negative: {args.negative}')

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
  
  print(f'actual train_len, valid_len: {len(train_q), len(valid_q)}')

  train_data = []

  train_data = generate_samples(train_q, q_data, neg_data, dbs_tables, dbs_table_names, dbs, add_negative=args.negative)
  valid_data = generate_samples(valid_q, q_data, neg_data, dbs_tables, dbs_table_names, dbs, add_negative=args.negative)

  print(len(train_data))
  print(len(valid_data))
 
  train_df = pd.DataFrame(train_data, columns=['question', 'table'])
  valid_df = pd.DataFrame(valid_data, columns=['question', 'table'])

  train_df.to_csv('./data/train_spider/train_ranking.csv', index=False)
  valid_df.to_csv('./data/train_spider/valid_ranking.csv', index=False)