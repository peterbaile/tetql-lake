# for each used table, remove columns until the minimum set (those in the column labels)

import json
from tqdm import tqdm
import pandas as pd
import random
from itertools import combinations

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

dbs_tables = {}

for db in db_data:
  db_id = db['db_id']
  num_tables = len(db['table_names'])

  dbs_table_names[db_id] = db['table_names']

  dbs_tables[db_id] = db

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

special_tokens = False

for q in tqdm(q_data):
  question = q['question']
  db_id = q['db_id']
  table_labels = q['table_labels']
  column_labels = q['column_labels']

  if len(table_labels) >= 2:
    continue
  
  count += 1

  # include all tables in the label
  correct_keys = [_key(db_id, ii, dbs_table_names) for ii in table_labels]

  train_tables = []

  tables_from_same_db = list(filter(lambda x: x[0] == db_id, dbs))

  train_tables += tables_from_same_db

  correct_cnt = 0
  incorrect_cnt = 0
  
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
      # fetch all columns that belong to the matched table
      matched_columns = set()
      min_columns = set()
      for idx, column in enumerate(dbs_tables[db_id]['column_names']):
        if column[0] == table_labels[0]:
          matched_columns.add(column[1])
        
        if idx != 0 and idx in column_labels:
          min_columns.add(column[1])
        
      difference_columns = list(matched_columns.difference(min_columns))
      difference_columns = difference_columns[:7]
      min_columns = list(min_columns)

      for l in range(len(difference_columns) + 1):
        for diff_col in list(combinations(difference_columns, l)):
          actual_cols = []
          actual_cols += diff_col
          actual_cols += min_columns

          text = f"{question} [SEP] {table[1]} {' '.join(actual_cols)}"
          all_data.append([text, 1])
          correct_cnt += 1
    else:
      all_data.append([text, 0])
      incorrect_cnt += 1
    
  if incorrect_cnt < correct_cnt:
    tables_not_from_same_db = list(filter(lambda x: x[0] != db_id, dbs))

    more_tables = random.choices(tables_not_from_same_db, k = correct_cnt - incorrect_cnt)

    for table in more_tables:
      cols = ' '.join(table[2])
      text = f'{question} [SEP] {table[1]} {cols}'
      all_data.append([text, 0])
    

print(f'number of queries {len(q_data)}')
print(f'actual number of queries processed {count}')
print(f'number of table {len(dbs)}')
print(f'number of data generated {len(all_data)}')
# print(all_data[0])
df = pd.DataFrame(all_data, columns=['text', 'label'])
df.to_csv('./data/train_spider/no_join_v3_diff_7.csv', index=False)


# df_file = pd.read_csv('./data/train_spider/all_v3.csv')
# df = pd.concat(df_file, df)

