import nltk
from nltk.corpus import stopwords
import json
import string
from tqdm import tqdm

# nltk.download('stopwords')
stopwords_en = stopwords.words('english')


with open('./spider_data/tables.json') as f:
  tables_data = json.load(f)

tables = []

for table in tqdm(tables_data):
  columns = set()
  for column in table['column_names'][1:]:
    columns.update(column[1].split(' '))
  label = table['db_id']
  tables.append((columns, label))

with open('./spider_data/dev.json', 'r') as f:
  dev_data = json.load(f)

qs = []
qs_words = []

for q in tqdm(dev_data):
  q_removed = q['question_toks']
  keywords = [word.lower() for word in q_removed if word.lower() not in stopwords_en and word.lower() not in string.punctuation]
  # print(q_removed)
  keywords = set(keywords)

  cands_hit = []
  cands_name = []
  words = []
  cands = []

  for table in tables:
    hit = 0
    for keyword in keywords:
      if keyword in table[0]:
        hit += 1
    
    if hit >= 1:
      cands_hit.append(hit)
      cands_name.append(table[1])
  
  if len(cands_hit) != 0:
    max_hit = max(cands_hit)

    for idx, table_name in enumerate(cands_name):
      if cands_hit[idx] == max_hit:
        cands.append(table_name)

  
  qs.append((cands, q['db_id']))
  # qs_words.append(words)

precision = 0
recall = 0

for result in qs:
  cands = result[0]
  label = result[1]

  if label in cands:
    recall += 1

    precision += (1/len(cands))

precision = precision/len(qs)
recall = recall/len(qs)
print(f'overall precision is {precision}')
print(f'overall recall is {recall}')
print(f'overall f-1 is {2*(precision * recall)/ (precision + recall)}')

# print(qs[:2])
# print(qs_words[:2])