import json
from tqdm import tqdm
import pandas as pd

def _key(db, table_idx):
  return f'{db}#sep#{table_idx}'

with open('../spider_data/dev_new_new.json', 'r') as f:
  dev_data = json.load(f)

with open('../spider_data/tables.json') as f:
  db_data = json.load(f)

dbs = {}
dbs_table_names = {}
for db in db_data:
  db_id = db['db_id']
  num_tables = len(db['table_names'])

  dbs_table_names[db_id] = db['table_names']

  for i in range(num_tables):
    dbs[_key(db_id, i)] = []

  for col in db['column_names']:
    if col[0] >= 0:
      dbs[_key(db_id, col[0])].append(col[1])

all_data = []

max_len = 0

for q in tqdm(dev_data):
  question = q['question']
  db_id = q['db_id']
  table_labels = q['table_labels']

  if len(table_labels) >= 2:
    continue
  
  table_idx = table_labels[0]
  correct_key = _key(db_id, table_idx)
  
  for key in dbs:
    cols = ' [COL] '.join(dbs[key])
    _db_id, idx = key.split('#sep#')
    # print(key)
    # print(idx)
    # print(type(int(idx)))
    _table_name = dbs_table_names[_db_id][int(idx)]
    text = f'{question} [SEP] [TBL] {_table_name} [COL] {cols}'

    if len(text) > max_len:
      max_len = len(text)

    if key == correct_key:
      all_data.append([text, 1])
    else:
      all_data.append([text, 0])

print(max_len)
# print(all_data[0])
# df = pd.DataFrame(all_data, columns=['text', 'label'])
# df.to_csv('./data.csv', index=False)