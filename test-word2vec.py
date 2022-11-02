import nltk
from nltk.corpus import stopwords
import json
import string
from tqdm import tqdm
from nltk.stem.porter import *
from gensim.test.utils import datapath
import gensim

# nltk.download('stopwords')
stopwords_en = stopwords.words('english')
stemmer = PorterStemmer()

model = gensim.models.fasttext.load_facebook_model(datapath('/Everything/Github/tetql-lake/content/wiki.en.bin'))

with open('./spider_data/tables.json') as f:
  tables_data = json.load(f)

tables = []

for db in tqdm(tables_data):
  for idx, table in enumerate(db['table_names_original']):
    label = f"{db['db_id']}-{table.lower()}"

    keywords = set()

    for column in db['column_names']:
      if column[0] == idx:
        _cols = column[1].lower().split(' ')
        _cols = [stemmer.stem(col) for col in _cols]
        keywords.update(_cols)
    
    keywords.add(stemmer.stem(db['table_names'][idx].lower()))

    tables.append((keywords, label))

with open('./spider_data/dev_new.json', 'r') as f:
  dev_data = json.load(f)

k = 1

for threshold in [0.7, 0.8, 0.9]:
  print(f'threshold is {threshold}')

  qs = []

  for q in tqdm(dev_data):
    if len(q['table_name']) > 1:
      continue

    keywords = q['question_toks']
    keywords = [word.lower() for word in keywords]
    keywords = [stemmer.stem(word) for word in keywords if word not in stopwords_en and word not in string.punctuation]
    # print(q_removed)
    q_keywords = set(keywords)
    
    cands = []

    for table in tables:
      t_keywords = table[0]
      t_label = table[1]

      count = 0

      for q_key in q_keywords:
        for t_key in t_keywords:
          # sim = 1 - scipy.spatial.distance.consine(embed[q_key], embed[t_key])
          count += model.wv.similarity(q_key, t_key) >= threshold

      if count >= 1:
        cands.append((t_label, count))
        # cands_count.append(count)
        # cands_name.append(t_label)
    
    cands.sort(key = lambda x: x[1], reverse=True)

    # print(q)
    q_label = f"{q['db_id']}-{q['table_name'][0].lower()}"
    qs.append((cands[:k], q_label))


  precision = 0
  recall = 0

  for result in qs:
    cands = result[0]
    label = result[1]

    cands_label = [x[0] for x in cands]

    if label in cands_label:
      recall += 1

      precision += (1/len(cands))

  precision = precision/len(qs)
  recall = recall/len(qs)
  print(f'overall precision@{k} is {precision}')
  print(f'overall recall@{k} is {recall}')
  print(f'overall f-1 is {2*(precision * recall)/ (precision + recall)}')