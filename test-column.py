import nltk
from nltk.corpus import stopwords
import json
import string
from tqdm import tqdm
from nltk.stem.porter import *

# nltk.download('stopwords')
stopwords_en = stopwords.words('english')
stemmer = PorterStemmer()

with open('./spider_data/tables.json') as f:
  tables_data = json.load(f)

tables = []

for db in tqdm(tables_data):
  for idx, table in enumerate(db['table_names_original']):
    label = f"{db['db_id']}-{table.lower()}"

    columns = set()

    for column in db['column_names']:
      if column[0] == idx:
        _cols = column[1].lower().split(' ')
        _cols = [stemmer.stem(col) for col in _cols]
        columns.update(_cols)
    
    columns.add(stemmer.stem(db['table_names'][idx].lower()))

    tables.append((columns, label))

# print(len(tables))

with open('./spider_data/dev_new.json', 'r') as f:
  dev_data = json.load(f)

qs = []

k = 1

for q in tqdm(dev_data):
  if len(q['table_name']) > 1:
    continue

  keywords = q['question_toks']
  keywords = [word.lower() for word in keywords]
  keywords = [stemmer.stem(word) for word in keywords if word not in stopwords_en and word not in string.punctuation]
  # print(q_removed)
  keywords = set(keywords)

  cands = []

  for table in tables:
    hit = 0
    for keyword in keywords:
      if keyword in table[0]:
        hit += 1
    
    if hit >= 1:
      cands.append((table[1], hit))
  
  cands.sort(key = lambda x: x[1], reverse=True)

  # print(q)
  q_label = f"{q['db_id']}-{q['table_name'][0].lower()}"
  qs.append((cands[:k], q_label))

  # TODO: fetch columns from tables

  # first determine the keyword that looks like table and remove it
  

  for q_keyword in keywords:
    for column in cands:


precision = 0
recall = 0

for result in qs:
  cands = result[0]
  label = result[1]

  cands_label = [x[0] for x in cands]

  # print(cands_label)
  # print(l)

  if label in cands_label:
    recall += 1

    precision += (1/len(cands))

precision = precision/len(qs)
recall = recall/len(qs)
print(f'overall precision@{k} is {precision}')
print(f'overall recall@{k} is {recall}')
print(f'overall f-1 is {2*(precision * recall)/ (precision + recall)}')

# print(qs[:2])
# print(qs_words[:2])