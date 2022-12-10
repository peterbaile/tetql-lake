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
  
  db_id_set = set()

  for q in q_data:
    db_id_set.add(q['db_id'])

  with open('../spider_data/tables.json') as f:
    db_data = json.load(f)

  if args.mode == 'em':
    with open('./data/dev/dev_idf_100.json') as f:
      dev_cands_data = json.load(f)

  dbs = {}
  dbs_table_names = {}
  for db in db_data:
    if db['db_id'] in db_id_set:
      db_id = db['db_id'].lower()
      num_tables = len(db['table_names'])

      dbs_table_names[db_id] = db['table_names']

      for i in range(num_tables):
        dbs[(db_id, i)] = []

      for col in db['column_names']:
        if col[0] >= 0:
          dbs[db_id, col[0]].append(col[1])

  all_data = []

  max_len = 0
  count = 0
  max_join = 0

  special_tokens = False

  for q_idx, q in enumerate(tqdm(q_data)):
    q_question = q['question']
    q_db_id = q['db_id'].lower()
    q_table_labels = q['table_labels']

    # if len(q_table_labels) != 4:
    #   continue

    max_join = max(max_join, len(q_table_labels))
    
    count += 1

    # table_idx = table_labels[0]
    # correct_key = _key(db_id, table_idx)

    # include all tables in the label

    dbs_actual = dbs
    
    if args.mode == 'em':
      dbs_actual = dev_cands_data[q_idx][:2]

    for key in dbs_actual:
      if special_tokens:
        cols = ' [COL] '.join(tuple(dbs[key]))
      else:
        cols = ','.join(dbs[tuple(key)])

      _db_id, idx = key
      # print(key)
      # print(idx)
      # print(type(int(idx)))
      _table_name = dbs_table_names[_db_id][idx]
      if special_tokens:
        text = f'{q_question} [SEP] [TBL] {_table_name} [COL] {cols}'
      else:
        text = f'{q_question} [SEP] {_db_id},{_table_name},{cols}'

      if len(text) > max_len:
        max_len = len(text)

      # if key == correct_key:
      if key[0] == q_db_id and key[1] in q_table_labels:  
        all_data.append([text, q_db_id, idx, 1])
      else:
        all_data.append([text, q_db_id, idx, 0])

  # print(len(dev_data))
  print(f'{count} queries')
  print(f'total number of table {len(dbs)}')
  print(f'max join table {max_join}')
  # print(max_len)
  print(len(all_data))
  # print(all_data[0])
  df = pd.DataFrame(all_data, columns=['text', 'db_id', 'table_index', 'label'])

  df.to_csv(f'./data/dev/{args.filename}_ranking.csv', index=False)