# table pruning using BERT
# pairwise comparisons
# regression with dropout + linear layer with 1 output

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer
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

random.seed(0)

# tokenizer.add_special_tokens({'additional_special_tokens': ['[TBL]', '[COL]']})

BERT_MODEL_TYPE = './prajjwal1/bert-tiny'

# tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_TYPE)

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_TYPE)

def tokenize(texts):
  return [tokenizer(text, padding='max_length', max_length=512, return_tensors='pt') for text in tqdm(texts)]

# tokenized = tokenize(['How many singers do we have? [SEP] [TBL] perpetrator [COL] perpetrator id [COL] people id [COL] date [COL] year [COL] location [COL] country [COL] killed [COL] injured'])
# print(tokenized[0].input_ids)
# print(tokenize(['[COL]'])[0].input_ids)
# print(tokenizer.convert_ids_to_tokens(tokenized[0].input_ids[0]))

class LogDataset(data.Dataset):
  def __init__(self, features, labels):
    self.features = tokenize(features)
    self.labels = torch.tensor(labels.values)
  
  def __getitem__(self, index):
    return self.features[index], self.labels[index]
  
  def __len__(self):
    return len(self.features)

class BertClassifier(nn.Module):
  def __init__(self, dropout=0.5):
    super(BertClassifier, self).__init__()

    # self.bert = BertModel.from_pretrained(BERT_MODEL_TYPE)
    self.bert = AutoModel.from_pretrained(BERT_MODEL_TYPE)
    self.bert.resize_token_embeddings(len(tokenizer))
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(128, 2)

  def forward(self, input_id, mask):
    _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
    dropout_output = self.dropout(pooled_output)
    linear_output = self.linear(dropout_output)

    return linear_output

device = 'cuda'

max_epochs = 50

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('mode', type=str)
  parser.add_argument('path', type=str)

  args = parser.parse_args()

  print(args.mode)
  print(f'source path: {args.path}')  

  if args.mode == 'train':
    train_df = pd.read_csv(f'./data/{args.path}/train.csv')
    valid_df = pd.read_csv(f'./data/{args.path}/valid.csv')
    train_X, train_Y = train_df.iloc[:, 0], train_df.iloc[:, 1]
    valid_X, valid_Y = valid_df.iloc[:, 0], valid_df.iloc[:, 1]

    train_batch_size = 20
    valid_batch_size = 400
    model = BertClassifier().to(device)
    print('finished downloading')
    train_dataset = LogDataset(train_X, train_Y)
    valid_dataset = LogDataset(valid_X, valid_Y)

    train_dataloader = data.DataLoader(train_dataset, batch_size = train_batch_size, shuffle = True)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size = valid_batch_size, shuffle = True)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=1e-5)

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
      for train_input, train_labels in tqdm(train_dataloader):
        train_labels = train_labels.to(device)
        mask = train_input['attention_mask'].to(device)
        input_id = train_input['input_ids'].squeeze(1).to(device)

        output = model(input_id, mask)

        acc = (output.argmax(dim=1) == train_labels).sum().item()
        total_acc_train += acc

        optimizer.zero_grad()
        loss = criterion(output, train_labels.long())
        
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
      
      model.eval()
      with torch.no_grad():
        for valid_input, valid_labels in tqdm(valid_dataloader):
          valid_labels = valid_labels.to(device)
          mask = valid_input['attention_mask'].to(device)
          input_id = valid_input['input_ids'].squeeze(1).to(device)

          output = model(input_id, mask)

          loss = criterion(output, valid_labels.long())

          valid_losses.append(loss.item())
      
      train_loss = np.average(train_losses)
      valid_loss = np.average(valid_losses)

      avg_train_losses.append(train_loss)
      avg_valid_losses.append(valid_loss)

      print(f'\n epoch {epoch}: training loss {train_loss:.5f}, validation loss {valid_loss:.5f}')

      if best_valid_loss is None:
        torch.save(model, f'./data/{args.path}/bert.pt')
        best_valid_loss = valid_loss
      elif valid_loss >= best_valid_loss:
        patience_cnt += 1
        print(f'early stopping {patience_cnt}/{patience}')

        if patience_cnt >= patience:
          print('early stopping')
          break
      else:
        torch.save(model, f'./data/{args.path}/bert.pt')
        best_valid_loss = valid_loss
        patience_cnt = 0

  elif args.mode == 'dev':
    model = torch.load('./bert.pt')

    test_X, test_Y = df.iloc[:, 0], df.iloc[:, 1]
    test_dataset = LogDataset(test_X, test_Y)
    test_dataloader = data.DataLoader(test_dataset, batch_size = 500)

    total_output = None
    total_acc_test = 0
    total_output_prob = None

    with torch.no_grad():
      for test_input, test_label in tqdm(test_dataloader):
        test_label = test_label.to(device)
        mask = test_input['attention_mask'].to(device)
        input_id = test_input['input_ids'].squeeze(1).to(device)

        raw_output = model(input_id, mask)
        output = raw_output.argmax(dim=1)

        if total_output is None:
          total_output = output.cpu()
          # total_output_prob = (raw_output.max()).cpu()
        else:
          total_output = torch.cat((total_output, output.cpu()), 0)
          # total_output_prob = torch.vstack((total_output_prob, (raw_output.max()).cpu()))

        acc = (output == test_label).sum().item()
        total_acc_test += acc

    print(total_acc_test)
    print(f'precision: {precision_score(test_Y, total_output)}')
    print(f'recall: {recall_score(test_Y, total_output)}')
    print(f'f1: {f1_score(test_Y, total_output)}')
    # print(f1_score(total_output, test_Y, average='micro'))
    # print(f1_score(total_output, test_Y, average='weighted'))

    
    # total_output_prob = total_output_prob.squeeze(1)
    # print(total_output_prob)

    # pd.set_option('display.max_colwidth', None)
    # print(test_X[torch.argsort(total_output_prob, descending=True)[:3].tolist()])