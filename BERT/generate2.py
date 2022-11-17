# try to generate better training data that's more balanced

import json
from tqdm import tqdm
import pandas as pd
import random

random.seed(123)

def _key(db, table_idx, dbs_table_names):
  return f'{db}#sep#{dbs_table_names[db][table_idx]}'

with open('../spider_data/train_spider_new.json', 'r') as f:
  q_data = json.load(f)

with open('../spider_data/tables.json') as f:
  db_data = json.load(f)

dbs_table_names = {}

# [(db_id, table_name, [cols])]
dbs = []

for db in db_data:
  db_id = db['db_id']
  num_tables = len(db['table_names'])

  dbs_table_names[db_id] = db['table_names']

  table_cols = {}

  for i in range(num_tables):
    table_cols[i] = []

  for col in db['column_names']:
    if col[0] >= 0:
      table_cols[col[0]].append(col[1])
    
  for table_index in table_cols:
    dbs.append((db_id, db['table_names'][table_index], table_cols[table_index]))

all_data = []

max_len = 0
count = 0

NUM_TABLES_PER_Q = 15

special_tokens = False

for q in tqdm(q_data):
  question = q['question']
  db_id = q['db_id']
  table_labels = q['table_labels']

  if len(table_labels) >= 2:
    continue
  
  count += 1

  # table_idx = table_labels[0]
  # correct_key = _key(db_id, table_idx)

  # include all tables in the label
  correct_keys = [_key(db_id, ii, dbs_table_names) for ii in table_labels]

  train_tables = []

  tables_from_same_db = list(filter(lambda x: x[0] == db_id, dbs))

  train_tables += tables_from_same_db

  tables_not_from_same_db = list(filter(lambda x: x[0] != db_id, dbs))

  if len(tables_from_same_db) < NUM_TABLES_PER_Q:
    more_tables = random.choices(tables_not_from_same_db, k = NUM_TABLES_PER_Q - len(tables_from_same_db))
    train_tables += more_tables
  
  for table in train_tables:
    if special_tokens:
      cols = ' [COL] '.join(table[2])
    else:
      cols = ' '.join(table[2])

    if special_tokens:
      text = f'{question} [SEP] [TBL] {table[1]} [COL] {cols}'
    else:
      text = f'{question} [SEP] {table[1]} {cols}'
    
    # print(text)
    
    if f'{table[0]}#sep#{table[1]}' in correct_keys:  
      all_data.append([text, 1])
    else:
      all_data.append([text, 0])

print(f'number of queries {len(q_data)}')
print(f'actual number of queries processed {count}')
print(f'number of table {len(dbs)}')
# print(max_len)
print(f'number of data generated {len(all_data)}')
# print(all_data[0])
df = pd.DataFrame(all_data, columns=['text', 'label'])
df.to_csv('./data/train_spider/no_join.csv', index=False)