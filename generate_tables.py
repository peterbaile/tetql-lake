# each table label is an array of tables

import json
from sql_metadata import Parser

with open('./spider_data/tables.json') as f:
  tables_data = json.load(f)

tables = set()

for table in tables_data:
  names = table['table_names_original']
  names = [name.lower() for name in names]
  tables.update(names)

with open('./spider_data/dev.json', 'r') as f:
  dev_data = json.load(f)

dev_data_new = []

count = 0
for q in dev_data:
  q_sql = q['query']
  
  q_tables = Parser(q_sql).tables
  q_tables = [table.lower() for table in q_tables]

  # sanity check that this table is actually in table.json
  for table in q_tables:
    assert(table in tables)

  q['table_name'] = q_tables

  dev_data_new.append(q)

  count += 1

with open('./spider_data/dev_new.json', 'w') as f:
  json.dump(dev_data_new, f, indent=4)

print(f'{count} / {len(dev_data)} queries')