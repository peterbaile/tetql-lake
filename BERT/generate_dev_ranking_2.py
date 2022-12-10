import json
from tqdm import tqdm
import pandas as pd

import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--mode', type=str)
  parser.add_argument('--filename', type=str)

  args = parser.parse_args()

  with open('../spider_data/dev_new.json', 'r') as f:
    q_data = json.load(f)
  
  dev_db_ids = set()

  for q in q_data:
    dev_db_ids.add(q['db_id'])

  with open('../spider_data/tables.json') as f:
    db_data = json.load(f)

  if args.mode == 'em':
    with open('./data/dev/dev_idf_100.json') as f:
      dev_cands_data = json.load(f)

  t_data = []

  tables = {} # key is (db_id, db_id_index)
  tables_index = {}

  for db in db_data:
    if db['db_id'] in dev_db_ids:
      db_id = db['db_id'].lower()
      num_tables = len(db['table_names'])

      for i in range(num_tables):
        tables[(db_id, i)] = []
        tables_index[(db_id, i)] = len(tables_index)

      for col in db['column_names']:
        if col[0] >= 0:
          tables[db_id, col[0]].append(col[1])
      
      for i in range(num_tables):
        t_data.append(f"{db_id},{db['table_names'][i]},{','.join(tables[(db_id, i)])}")

  q_data2 = []

  max_len = 0
  count = 0
  max_join = 0

  for q_idx, q in enumerate(tqdm(q_data)):
    q_question = q['question']
    q_db_id = q['db_id'].lower()
    q_table_labels = q['table_labels']

    if len(q_table_labels) != 1:
      continue

    max_join = max(max_join, len(q_table_labels))
    
    count += 1

    # table_idx = table_labels[0]
    # correct_key = _key(db_id, table_idx)

    # include all tables in the label

    # dbs_actual = dbs
    
    # if args.mode == 'em':
    #   dbs_actual = dev_cands_data[q_idx][:2]
    q_table_labels_global = [tables_index[(q_db_id, i)] for i in q_table_labels]
    q_data2.append([q_question, json.dumps(q_table_labels_global)])

      

  # print(len(dev_data))
  print(f'{len(q_data2)} queries')
  print(f'total number of table {len(t_data)}')
  print(f'max join table {max_join}')
  q_df = pd.DataFrame(q_data2, columns=['question', 'label'])
  t_df = pd.DataFrame(t_data, columns=['table'])

  q_df.to_csv(f'./data/dev/{args.filename}_q_ranking.csv', index=False)
  t_df.to_csv(f'./data/dev/{args.filename}_t_ranking.csv', index=False)