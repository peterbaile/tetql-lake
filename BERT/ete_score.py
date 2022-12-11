# obtain end-to-end performance on the T5 model

from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer, file_utils
from torch.utils import data
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import random
import argparse
import math
import json
from os.path import exists
import sys
# from picard import generate_queries

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

MODEL_TYPE = 'roberta-base'

EMBED_SIZE = {
  'bert-tiny': 128,
  'bert-base-uncased': 768,
  'roberta-base': 768,
  'roberta-large': 1024
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)

def tokenize(text):
  return tokenizer(text, padding='max_length', max_length=300, return_tensors='pt')

class BertClassifier(nn.Module):
  def __init__(self, dropout=0.1):
    super(BertClassifier, self).__init__()

    self.bert = AutoModel.from_pretrained(MODEL_TYPE)
    # self.bert.resize_token_embeddings(len(tokenizer))
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(EMBED_SIZE[MODEL_TYPE], 1)

  def forward(self, input_id, mask):
    _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
    dropout_output = self.dropout(pooled_output)
    linear_output = self.linear(dropout_output)

    return linear_output

class DevDataset(data.Dataset):
  def __init__(self, texts, labels):
    self.texts = [tokenize(text) for text in tqdm(texts)]
    self.labels = torch.tensor(labels.values)
  
  def __getitem__(self, index):
    return self.texts[index], self.labels[index]

  def __len__(self):
    return len(self.texts)

def suffix(base, args, connector, ext):
  new_base = base

  if args.join:
    new_base += f'{connector}join'
  
  if args.addnegative:
    new_base += f'{connector}negative'
  
  new_base += ext

  return new_base

def collate_fn(batch):
  # this is OK because batch_size = 1
  return batch[0]

device = 'cuda'

def evaluate(CANDS_PATH):
  cands_dev_df = pd.read_csv(CANDS_PATH)

  # TODO: generate cands_dev_df


  pred_queries, gold_queries = generate_queries(cands_dev_df)

  

  with open('./data/eval/pred.txt', 'w') as f:
    f.write('\n'.join(pred_queries))
  
  with open('./data/eval/gold.txt', 'w') as f:
    f.write('\n'.join(gold_queries))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--path', type=str)
  parser.add_argument('--devfile', type=str)
  parser.add_argument('--addnegative', type=bool, default=False)
  parser.add_argument('--join', type=bool, default=False)
  parser.add_argument('--topk', type=int)
  parser.add_argument('--rerank', action='store_true')
  parser.add_argument('--topnum', type=int, default=-1)

  args = parser.parse_args()

  CANDS_PATH = f'./data/dev/{args.devfile}_ranking_cands.csv'

  # if exists(CANDS_PATH):
  #   evaluate(CANDS_PATH)
  #   sys.exit(0)

  MODEL_PATH = suffix(f'./data/{args.path}/{MODEL_TYPE}-ranking', args, '-', '.pt')
  print(f'source path: {args.path}, {MODEL_TYPE}, {MODEL_PATH}, add negative: {args.addnegative}')

  print(f'dev file: {args.devfile}, topk: {args.topk}, re-rank: {args.rerank}, top #cands: {args.topnum}')
  model = torch.load(MODEL_PATH)
  dev_batch_size = 81 # this has to be the same as the number of candidates picked (81 no join, 100 idf)

  dev_df = pd.read_csv(f'./data/dev/{args.devfile}_ranking.csv')

  dev_X, dev_Y = dev_df.iloc[:, 0], dev_df.iloc[:, -1]
  dev_dataset = DevDataset(dev_X, dev_Y)
  dev_dataloader = data.DataLoader(dev_dataset, batch_size = dev_batch_size)

  total_output = None
  total_max_indices = None

  model.eval()
  with torch.no_grad():
    for i, (test_input, test_label) in enumerate(tqdm(dev_dataloader)):
      test_label = test_label.to(device)
      mask = test_input['attention_mask'].to(device)
      input_id = test_input['input_ids'].squeeze(1).to(device)

      raw_output = model(input_id, mask).squeeze(1)
      
      if args.rerank:
        _, max_indices = torch.topk(raw_output, args.topnum)
      else:
        _, max_indices = torch.topk(raw_output, args.topk)

      max_indices = max_indices.tolist()

      output = [0 for _ in range(dev_batch_size)]

      if args.rerank:
        max_idx = torch.argmax(raw_output).item()
        max_db_id = dev_df.iloc[i * dev_batch_size + max_idx]['db_id']
      
      # db_id_set = set()

      db_count = {}

      num_tables = 0
      for max_i in max_indices:
        if args.rerank:
          cand_db_id = dev_df.iloc[i * dev_batch_size + max_i]['db_id']
          # if cand_db_id not in db_count:
          #   db_count[cand_db_id] = [1, [max_i]]
          # else:
          #   db_count[cand_db_id][0] += 1
          #   db_count[cand_db_id][1].append(max_i)
          if num_tables < args.topk and cand_db_id == max_db_id:
            num_tables += 1
            output[max_i] = 1
        else:
          output[max_i] = 1
        
        # db_id_set.add(dev_df.iloc[i * dev_batch_size + max_i]['db_id'])
      
      # if args.rerank:
      #   _, max_indices_reranked = sorted(db_count.items(), key=lambda item: item[1][0], reverse=True)[0][1]
      #   max_indices_reranked = max_indices_reranked[:args.topk]

      #   for max_i in max_indices_reranked:
      #     output[max_i] = 1
      
      # assert(len(db_id_set) == 1)

      if total_output is None:
        total_output = output
        # total_output_prob = (raw_output.max()).cpu()
      else:
        total_output += output
        # total_output_prob = torch.vstack((total_output_prob, (raw_output.max()).cpu()))

      # if args.rerank:
      #   max_indices = [x + i * dev_batch_size for x in max_indices_reranked]
      # else:
      #   max_indices = [x + i * dev_batch_size for x in max_indices]

      # if total_max_indices is None:
      #   total_max_indices = max_indices
      # else:
      #   total_max_indices += max_indices
    
  print(f'accuracy: {100 * accuracy_score(dev_Y, total_output):.3f}%')
  print(f'precision: {100 * precision_score(dev_Y, total_output):.3f}%')
  print(f'recall: {100 * recall_score(dev_Y, total_output):.3f}%')
  print(f'f1: {100 * f1_score(dev_Y, total_output):.3f}%')
  
  cands_dev_df = dev_df.iloc[total_max_indices]
  cands_dev_df.to_csv(CANDS_PATH, index=False)
  
  num_q = int(args.topk * dev_df.shape[0]/dev_batch_size)
  print(f'expected size {num_q}, actual size {cands_dev_df.shape[0]}')

  assert(num_q == cands_dev_df.shape[0])

  print(f'#questions is {num_q}, cands shape {cands_dev_df.shape}')