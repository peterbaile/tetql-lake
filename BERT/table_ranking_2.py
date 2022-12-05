# two BERT models fed into a linear layer to get the final similarity score

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
    self.dropout = nn.Dropout(dropout)

  def forward(self, input_id, mask):
    _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
    dropout_output = self.dropout(pooled_output)

    return dropout_output

class TrainDataset(data.Dataset):
  def __init__(self, questions, tables, batch_instance_size, add_negative):
    self.questions = questions
    self.tables = tables
    self.num_instance = batch_instance_size
    self.add_negative = add_negative
  
  def __getitem__(self, index):
    # print(f'index {index}')
    start_idx = index * self.num_instance
    end_idx = (index + 1) * self.num_instance

    if end_idx <= len(self.questions):
      qs = self.questions[start_idx : end_idx]
      ts = self.tables[start_idx : end_idx]
    else:
      qs = self.questions[start_idx:]
      ts = self.tables[start_idx:]
    
    if self.add_negative:
      pos_ts = []
      neg_ts = []
      for t in ts:
        pos_t, neg_t = t.split('#sep#')
        pos_ts.append(pos_t)
        neg_ts.append(neg_t)
      
      ts = pos_ts + neg_ts
    
    q_batch_mask = None
    q_batch_input_id = None
    label = []
    for i, q in enumerate(qs):
      label.append(i)
      
      r = tokenize(q)

      if q_batch_mask is None:
        q_batch_mask = r['attention_mask'].unsqueeze(0)
        q_batch_input_id = r['input_ids']
      else:
        q_batch_mask = torch.vstack((q_batch_mask, r['attention_mask'].unsqueeze(0)))
        q_batch_input_id = torch.vstack((q_batch_input_id, r['input_ids']))
    

    t_batch_mask = None
    t_batch_input_id = None
    for i, t in enumerate(ts):
      r = tokenize(t)

      if t_batch_mask is None:
        t_batch_mask = r['attention_mask'].unsqueeze(0)
        t_batch_input_id = r['input_ids']
      else:
        t_batch_mask = torch.vstack((t_batch_mask, r['attention_mask'].unsqueeze(0)))
        t_batch_input_id = torch.vstack((t_batch_input_id, r['input_ids']))

    return q_batch_mask, q_batch_input_id, t_batch_mask, t_batch_input_id, torch.tensor(label)
  
  def __len__(self):
    return math.ceil(len(self.questions) / self.num_instance)

# TODO: need to be rewritten
# class DevDataset(data.Dataset):
#   def __init__(self, texts, labels):
#     print('tokenizing texts')
#     self.texts = [tokenize(text) for text in tqdm(texts)]
#     self.labels = torch.tensor(labels.values)
  
#   def __getitem__(self, index):
#     return self.texts[index], self.labels[index]

#   def __len__(self):
#     return len(self.texts)

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
max_epochs = 100

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--mode', type=str)
  parser.add_argument('--path', type=str)
  parser.add_argument('--devfile', type=str)
  parser.add_argument('--devpart', type=int)
  parser.add_argument('--addnegative', type=bool, default=False)
  parser.add_argument('--join', type=bool, default=False)
  parser.add_argument('--topk', type=int)

  args = parser.parse_args()

  print(f'mode: {args.mode}, source path: {args.path}, add negative: {args.addnegative}, join: {args.join}')
  learning_rate = 1e-6 # 1e-5 or 1e-6
  print(f'learning rate: {learning_rate}')

  Q_MODEL_PATH = suffix(f'./data/{args.path}/{MODEL_TYPE}-ranking-q', args, '-', '.pt')
  T_MODEL_PATH = suffix(f'./data/{args.path}/{MODEL_TYPE}-ranking-t', args, '-', '.pt')

  print(MODEL_TYPE, Q_MODEL_PATH, T_MODEL_PATH)

  if args.mode == 'train':
    train_df = pd.read_csv(suffix(f'./data/{args.path}/train_ranking', args, '_', '.csv'))
    valid_df = pd.read_csv(suffix(f'./data/{args.path}/valid_ranking', args, '_', '.csv'))
    
    train_X, train_Y = train_df.iloc[:, 0], train_df.iloc[:, 1]
    valid_X, valid_Y = valid_df.iloc[:, 0], valid_df.iloc[:, 1]

    train_batch_instance_size = 7
    valid_batch_instance_size = 10
    q_model = BertClassifier().to(device)
    t_model = BertClassifier().to(device)
    print('finished loading model')
    train_dataset = TrainDataset(train_X, train_Y, train_batch_instance_size, args.addnegative)
    valid_dataset = TrainDataset(valid_X, valid_Y, valid_batch_instance_size, args.addnegative)

    train_dataloader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    criterion = nn.NLLLoss(reduction='mean').to(device)
    m = nn.LogSoftmax(dim=1).to(device)
    optimizer = Adam(list(q_model.parameters()) + list(t_model.parameters()), lr=learning_rate) # back-propagate two models together

    train_losses = []
    valid_losses = []

    avg_train_losses = []
    avg_valid_losses = []
    best_valid_loss = None

    patience_cnt = 0
    patience = 5

    for epoch in range(max_epochs):
      torch.cuda.empty_cache()
      
      q_model.train()
      t_model.train()
      for q_batch_mask, q_batch_input_id, t_batch_mask, t_batch_input_id, train_labels in tqdm(train_dataloader):
        # print(train_labels)
        train_labels = train_labels.to(device)
        q_mask = q_batch_mask.to(device)
        q_input_id = q_batch_input_id.to(device)

        t_mask = t_batch_mask.to(device)
        t_input_id = t_batch_input_id.to(device)

        # print(mask.shape, input_id.shape, train_labels.shape)

        q_output = q_model(q_input_id, q_mask)
        t_output = t_model(t_input_id, t_mask)

        # compute dot product (cartesian product) --> essentially just matrix multiplication
        output = torch.matmul(q_output, t_output.T)

        optimizer.zero_grad()

        # print(output.shape)
        # print(train_labels.shape)
        # print(train_labels)

        loss = criterion(output, train_labels.long())
        
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
      
      q_model.eval()
      t_model.eval()
      with torch.no_grad():
        for q_batch_mask, q_batch_input_id, t_batch_mask, t_batch_input_id, valid_labels in tqdm(valid_dataloader):
          valid_labels = valid_labels.to(device)
          q_mask = q_batch_mask.to(device)
          q_input_id = q_batch_input_id.to(device)

          t_mask = t_batch_mask.to(device)
          t_input_id = t_batch_input_id.to(device)

          # print(mask.shape, input_id.shape, train_labels.shape)

          q_output = q_model(q_input_id, q_mask)
          t_output = t_model(t_input_id, t_mask)

          # compute dot product (cartesian product) --> essentially just matrix multiplication
          output = torch.matmul(q_output, t_output.T)

          loss = criterion(output, valid_labels.long())

          valid_losses.append(loss.item())
      
      train_loss = np.average(train_losses)
      valid_loss = np.average(valid_losses)

      avg_train_losses.append(train_loss)
      avg_valid_losses.append(valid_loss)

      print(f'\n epoch {epoch}: training loss {train_loss:.5f}, validation loss {valid_loss:.5f}')

      if best_valid_loss is None:
        torch.save(q_model, Q_MODEL_PATH)
        torch.save(t_model, T_MODEL_PATH)
        print(f'model saved')
        best_valid_loss = valid_loss
      elif valid_loss >= best_valid_loss:
        patience_cnt += 1
        print(f'early stopping {patience_cnt}/{patience}')

        if patience_cnt >= patience:
          print('early stopping')
          break
      else:
        torch.save(q_model, Q_MODEL_PATH)
        torch.save(t_model, T_MODEL_PATH)
        print(f'model saved')
        best_valid_loss = valid_loss
        patience_cnt = 0