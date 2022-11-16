# each table label is an array of tables

import json

with open('./spider_data/tables.json') as f:
  tables_data = json.load(f)

tables = set()
columns = set()

for table in tables_data:
  table_names = table['table_names_original']
  table_names = [name.lower() for name in table_names]
  tables.update(table_names)

  col_names = table['column_names_original'][1:]
  col_names = [col[1].lower() for col in col_names]
  columns.update(col_names)

def find_values(id, json_repr):
  results = []

  def _decode_dict(a_dict):
      try:
          results.append(a_dict[id])
      except KeyError:
          pass
      return a_dict

  json.loads(json_repr, object_hook=_decode_dict) # Return value ignored.
  return results

with open('./spider_data/train_spider.json', 'r') as f:
  dev_data = json.load(f)
  # dev_data = json.loads(f.read(), object_hook=lambda _dict : _dict['sql'])
  # dev_data = find_values('table_units', f.read())

# roughly fetches all tables
# for q in dev_data:
#   print(find_values('table_units', json.dumps(q)))

# fetch columns

# select
# conds
# where
# groupBy
# orderBy

def col_unit_to_col(unit):
  return unit[1]

def val_unit_to_col(unit):
  cols = set()

  cols.add(col_unit_to_col(unit[1]))

  if unit[2] is not None:
    cols.add(col_unit_to_col(unit[2]))
  
  return cols

def cond_to_col(cond):
  cols = set()
  for idx, _cond in enumerate(cond):
    if idx % 2 == 0:
      val_unit = _cond[2]
      cols = cols.union(val_unit_to_col(val_unit))

      if _cond[3] is not None and type(_cond[3]) is list and len(_cond[3]) == 3:
        cols.add(col_unit_to_col(_cond[3]))

      if _cond[4] is not None and type(_cond[4]) is list and len(_cond[4]) == 3:
        cols.add(col_unit_to_col(_cond[4]))
  return cols

def select_to_col(select):
  cols = set()
  for _select in select[1]:
    val_unit = _select[1]
    cols = cols.union(val_unit_to_col(val_unit))
  
  return cols

def order_by_to_col(order_by):
  cols = set()

  for val_unit in order_by[1]:
    cols = cols.union(val_unit_to_col(val_unit))
  
  return cols

def group_by_to_col(group_by):
  cols = set()

  for col_unit in group_by:
    cols.add(col_unit_to_col(col_unit))
  
  return cols

dev_data_new = []

for idx, q in enumerate(dev_data):
  cols = set()
  tables = set()

  _select_ls = find_values('select', json.dumps(q))
  # print(idx, _select_ls)
  for _select in _select_ls:
    if len(_select) != 0:
      cols = cols.union(select_to_col(_select))
  # print(cols)

  _cond_ls = find_values('conds', json.dumps(q))
  # print(idx, _cond_ls)
  for _cond in _cond_ls:
    if len(_cond) != 0:
      cols = cols.union(cond_to_col(_cond))
  # print(cols)
  
  _where_ls = find_values('where', json.dumps(q))
  # print(idx, _where_ls)
  for _where in _where_ls:
    if len(_where) != 0:
      cols = cols.union(cond_to_col(_where))
  # print(cols)

  _groupby_ls = find_values('groupBy', json.dumps(q))
  # print(_groupby_ls)
  for _groupby in _groupby_ls:
    if len(_groupby) != 0:
      cols = cols.union(group_by_to_col(_groupby))
  # print(cols)

  _orderby_ls = find_values('orderBy', json.dumps(q))
  # print(_orderby_ls)
  for _orderby in _orderby_ls:
    if len(_orderby) != 0:
      cols = cols.union(order_by_to_col(_orderby))
  # print(cols)

  _having_ls = find_values('having', json.dumps(q))
  # print(idx, _having_ls)
  for _having in _having_ls:
    if len(_having) != 0:
      cols = cols.union(cond_to_col(_having))
  # print(idx, cols)

  _tables = find_values('table_units', json.dumps(q))
  print(idx, _tables)
  for _table_unit in _tables:
    if len(_table_unit) != 0:
      for _table in _table_unit:
        if type(_table) is list and _table[0] == 'table_unit':
          tables.add(_table[1])
  print(tables)

  q['table_labels'] = list(tables)
  q['column_labels'] = list(cols)

  dev_data_new.append(q)

with open('./spider_data/train_spider_new.json', 'w') as f:
  json.dump(dev_data_new, f, indent=4)

print(f'{len(dev_data_new)} / {len(dev_data)} queries')