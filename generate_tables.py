# each table label is an array of tables

import json
from sql_metadata import Parser

with open('./spider_data/tables.json') as f:
  tables_data = json.load(f)

tables = set()
columns = set()

for table in tables_data:
  names = table['table_names_original']
  names = [name.lower() for name in names]
  tables.update(names)

  col_names = table['column_names_original'][1:]
  col_names = [col[1].lower() for col in col_names]
  columns.update(col_names)

with open('./spider_data/dev.json', 'r') as f:
  dev_data = json.load(f)

dev_data_new = []

count = 0
for q in dev_data:
  q_sql = q['query']
  
  q_parse = Parser(q_sql)
  q_tables = q_parse.tables
  q_tables = [table.lower() for table in q_tables]

  
  q_cols_raw = q_parse.columns
  q_cols = set()

  print(q_sql)
  print(q_cols_raw)
  

  for col in q_cols_raw:
    if '.' in col:
      q_cols.add(col.lower().split('.')[1])
    else:
      q_cols.add(col.lower())
  
  q_cols = list(q_cols)

  # sanity check that this table is actually in table.json
  for table in q_tables:
    assert(table in tables)
  
  for col in q_cols:
    assert(col in columns)

  q['table_name'] = q_tables
  q['column_name'] = q_cols

  dev_data_new.append(q)

  count += 1

with open('./spider_data/dev_new_newp.json', 'w') as f:
  json.dump(dev_data_new, f, indent=4)

print(f'{count} / {len(dev_data)} queries')