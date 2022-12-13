from typing import Tuple, List

import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import json


def load_picard_model(name="tscholak-3vnuv1vf"):
  tokenizer = AutoTokenizer.from_pretrained(name)
  model = AutoModelForSeq2SeqLM.from_pretrained(name)
  return tokenizer, model


# def tokenize(tokenizer, question: str, db_id: str, tables: List[Tuple[str, List[str]]]) -> Tensor:
#   input_string = f"{question} | {db_id}"
#   for t in tables:
#     input_string += " | "
#     input_string += t[0]
#     input_string += " : "
#     for i, c in enumerate(t[1]):
#       if i > 0:
#         input_string += ", "
#       input_string += c

#   input_ids = tokenizer(input_string, max_length=512, return_tensors="pt").input_ids

#   return input_string, input_ids

def tokenize(tokenizer, question):
  return tokenizer(input_string, max_length=512, return_tensors="pt").input_ids

def generate_single(model, tokenizer, input_data: Tensor):
  input_data = input_data.to(model.device)
  outputs = model.generate(inputs=input_data, num_beams=10, top_k=10, max_length=512)
  result = tokenizer.decode(token_ids=outputs[0], skip_special_tokens=True)
  sql_query = "|".join(result.split("|")[1:])[1:]
  return result, sql_query

def generate_string(q, db_id, tables, DB_TO_COLS):
  tables_string_ls = [DB_TO_COLS[db_id, table] for table in tables]
  result = f"{q} | {db_id} | {' | '.join(tables_string)}"

  print(result)

  return result

def generate_queries(picard_cands_dict):
  device = 'cuda'
  print(f'loading model')
  tokenizer, model = load_picard_model()
  model = model.to(device)

  pred_sql_queries = []
  gold_sql_queries = []

  with open('../spider_data/tables.json') as f:
    db_data = json.load(f)

  DB_TO_COLS = {}
  db_data_dict = {}
  
  for db in db_data:
    db_id = db['db_id']
    db_data_dict[db_id] = db

    for col in db['column_names_original']:
      if col[0] == -1:
        continue
      
      if (db_id, col[0]) not in DB_TO_COLS:
        DB_TO_COLS[(db_id, col[0])] = [col[1]]
      else:
        DB_TO_COLS[(db_id, col[0])].append(col[1])
    
  for tbl in DB_TO_COLS:
    db_id, tbl_idx = tbl
    tbl_orig_name = db_data_dict[db_id]['table_names_original'][tbl_idx]
    cols = ', '.join(DB_TO_COLS[tbl])
    DB_TO_COLS[tbl] = f'{tbl_orig_name} : {cols}'

  for q in tqdm(picard_cands_dict):
    q_db_id, gold_sql, q_tbl_indices = q
    gold_sql_queries.append(gold_sql)

    input_data = tokenize(tokenizer, generate_string(q, q_db_id, q_tbl_indices, DB_TO_COLS))
    _, sql_query = generate_single(model, tokenizer, input_data)

    pred_sql_queries.append(sql_query)

  return pred_sql_queries, gold_sql_queries


def test():
  # device = (
  #     torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  # )
  device = torch.device("cpu")
  tokenizer, model = load_picard_model()
  # model = model.to(device)

  gt = "How many singers do we have? | concert_singer | stadium : stadium_id, location, name, capacity, highest, lowest, average | singer : singer_id, name, country, song_name, song_release_year, age, is_male | concert : concert_id, concert_name, theme, stadium_id, year | singer_in_concert : concert_id, singer_id"
  question = "How many singers do we have?"
  db_id = "concert_singer"
  tables = [
    (
      "stadium",
      [
        "stadium_id",
        "location",
        "name",
        "capacity",
        "highest",
        "lowest",
        "average",
      ],
    ),
    (
      "singer",
      [
        "singer_id",
        "name",
        "country",
        "song_name",
        "song_release_year",
        "age",
        "is_male",
      ],
    ),
    ("concert", ["concert_id", "concert_name", "theme", "stadium_id", "year"]),
    ("singer_in_concert", ["concert_id", "singer_id"]),
  ]

  input_string, input_data = tokenize(tokenizer, question, db_id, tables)
  assert input_string == gt, input_string
  result, sql_query = generate_single(model, tokenizer, input_data)
  print(result)
  print(sql_query)
  assert result == "concert_singer | select count(*) from singer", result


if __name__ == "__main__":
  test()