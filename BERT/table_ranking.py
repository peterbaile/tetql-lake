# train with ranking loss

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, file_utils
from torch.utils import data
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import random
import argparse
import math

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

MODEL_TYPE = 'bert-tiny'

# tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)

def tokenize(text):
  return tokenizer(text, padding='max_length', max_length=300, return_tensors='pt')

class LogDataset(data.Dataset):
  def __init__(self, questions, tables, batch_instance_size):
    self.questions = questions
    self.tables = tables
    self.num_instance = batch_instance_size
  
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
    
    batch_mask = None
    batch_input_id = None
    label = []
    for i, q in enumerate(qs):
      single_mask = None
      single_input_id = None
      label.append(i)
      for t in ts:
        r = tokenize(f'{q} [SEP] {t}')

        if single_mask is None:
          single_mask = r['attention_mask'].unsqueeze(0)
          single_input_id = r['input_ids']
        else:
          single_mask = torch.vstack((single_mask, r['attention_mask'].unsqueeze(0)))
          single_input_id = torch.vstack((single_input_id, r['input_ids']))
      
      if batch_mask is None:
        batch_mask = single_mask
        batch_input_id = single_input_id
      else:
        batch_mask = torch.vstack((batch_mask, single_mask))
        batch_input_id = torch.vstack((batch_input_id, single_input_id))

    return batch_mask, batch_input_id, torch.tensor(label)
  
  def __len__(self):
    return math.ceil(len(self.questions) / self.num_instance)

class BertClassifier(nn.Module):
  def __init__(self, dropout=0.1):
    super(BertClassifier, self).__init__()

    # self.bert = BertModel.from_pretrained(MODEL_TYPE)
    self.bert = AutoModel.from_pretrained(MODEL_TYPE)
    self.bert.resize_token_embeddings(len(tokenizer))
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(128, 1)

  def forward(self, input_id, mask):
    _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
    dropout_output = self.dropout(pooled_output)
    linear_output = self.linear(dropout_output)

    return linear_output

device = 'cuda'

max_epochs = 50

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--mode', type=str)
  parser.add_argument('--path', type=str)
  parser.add_argument('--devfile', type=str)
  parser.add_argument('--devpart', type=int)

  args = parser.parse_args()

  print(f'mode: {args.mode}, source path: {args.path}')
  learning_rate = 1e-6 # 1e-5 or 1e-6
  print(f'learning rate: {learning_rate}')
  print(MODEL_TYPE)

  if args.mode == 'train':
    train_df = pd.read_csv(f'./data/{args.path}/train_ranking.csv')
    valid_df = pd.read_csv(f'./data/{args.path}/valid_ranking.csv')
    train_X, train_Y = train_df.iloc[:, 0], train_df.iloc[:, 1]
    valid_X, valid_Y = valid_df.iloc[:, 0], valid_df.iloc[:, 1]

    train_batch_instance_size = 7
    valid_batch_instance_size = 10
    model = BertClassifier().to(device)
    print('finished loading model')
    train_dataset = LogDataset(train_X, train_Y, train_batch_instance_size)
    valid_dataset = LogDataset(valid_X, valid_Y, valid_batch_instance_size)

    train_dataloader = data.DataLoader(train_dataset, batch_size = 1, shuffle = True)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size = 1, shuffle = True)

    criterion = nn.NLLLoss(reduction='mean').to(device)
    m = nn.LogSoftmax(dim=1).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    valid_losses = []

    avg_train_losses = []
    avg_valid_losses = []
    best_valid_loss = None

    patience_cnt = 0
    patience = 5

    for epoch in range(max_epochs):
      total_acc_train = 0
      
      model.train()
      for batch_mask, batch_input_id, train_labels in tqdm(train_dataloader):
        train_labels = train_labels.squeeze(0).to(device)
        mask = batch_mask.squeeze(0).to(device)
        input_id = batch_input_id.squeeze(0).to(device)

        # print(mask.shape, input_id.shape, train_labels.shape)

        output = model(input_id, mask)

        # print(output.shape)

        num_instance = train_labels.shape[0]
        output = output.reshape((num_instance, num_instance))
        output = m(output)

        # acc = (output.argmax(dim=1) == train_labels).sum().item()
        # total_acc_train += acc

        optimizer.zero_grad()

        # print(output.shape)
        # print(train_labels.shape)
        # print(train_labels)

        loss = criterion(output, train_labels.long())
        
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
      
      model.eval()
      with torch.no_grad():
        for batch_mask, batch_input_id, valid_labels in tqdm(valid_dataloader):
          valid_labels = valid_labels.squeeze(0).to(device)
          mask = batch_mask.squeeze(0).to(device)
          input_id = batch_input_id.squeeze(0).to(device)

          output = model(input_id, mask)
          num_instance = valid_labels.shape[0]
          output = output.reshape((num_instance, num_instance))
          output = m(output)

          loss = criterion(output, valid_labels.long())

          valid_losses.append(loss.item())
      
      train_loss = np.average(train_losses)
      valid_loss = np.average(valid_losses)

      avg_train_losses.append(train_loss)
      avg_valid_losses.append(valid_loss)

      print(f'\n epoch {epoch}: training loss {train_loss:.5f}, validation loss {valid_loss:.5f}')

      if best_valid_loss is None:
        torch.save(model, f'./data/{args.path}/{MODEL_TYPE}-ranking.pt')
        print(f'model saved')
        best_valid_loss = valid_loss
      elif valid_loss >= best_valid_loss:
        patience_cnt += 1
        print(f'early stopping {patience_cnt}/{patience}')

        if patience_cnt >= patience:
          print('early stopping')
          break
      else:
        torch.save(model, f'./data/{args.path}/{MODEL_TYPE}-ranking.pt')
        print(f'model saved')
        best_valid_loss = valid_loss
        patience_cnt = 0