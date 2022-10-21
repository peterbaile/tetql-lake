import json

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
  q_sql = q['query_toks']
  q_sql = [word.lower() for word in q_sql]

  # skip join for now
  if 'join' in q_sql:
    continue

  # search for the word that comes right after FROM
  # print(q_sql)
  q_table = q_sql[q_sql.index('from') + 1].lower()

  # sanity check that this table is actually in table.json
  assert(q_table in tables)

  q['table_name'] = q_table

  dev_data_new.append(q)

  count += 1

with open('./dev_new.json', 'w') as f:
  json.dump(dev_data_new, f, indent=4)

print(f'{count} / {len(dev_data)} queries')