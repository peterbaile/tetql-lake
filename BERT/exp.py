import json

# number of tables saved
def num_tables_saved(q_data, db_data):
  db_table_count = {}
  
  for db in db_data:
    db_table_count[db['db_id']] = len(db['table_names'])

  all_count = 0
  our_count = 0
  for q in q_data:
    # our_count += len(q['table_labels'])
    num_matching_tables = len(q['table_labels'])

    if num_matching_tables == 1:
      our_count += 1
    else:
      our_count += 4

    # our_count += 4

    all_count += db_table_count[q['db_id']]

  print(our_count, all_count, -100 * (our_count - all_count) / all_count)

if __name__ == '__main__':
  with open('../spider_data/dev_new.json', 'r') as f:
    q_data = json.load(f)
  
  dev_db_ids = set()

  for q in q_data:
    dev_db_ids.add(q['db_id'])

  with open('../spider_data/tables.json') as f:
    db_data = json.load(f)
  
  db_data = list(filter(lambda x : x['db_id'] in dev_db_ids, db_data))
  print(f'#db in dev: {len(db_data)}')
  
  num_tables_saved(q_data, db_data)