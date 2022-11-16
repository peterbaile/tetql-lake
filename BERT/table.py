# table pruning using BERT
# pairwise comparisons
# regression with dropout + linear layer with 1 output

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
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

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# tokenizer.add_special_tokens({'additional_special_tokens': ['[TBL]', '[COL]']})

def tokenize(texts):
  return [tokenizer(text, padding='max_length', return_tensors='pt') for text in tqdm(texts)]

# tokenized = tokenize(['How many singers do we have? [SEP] [TBL] perpetrator [COL] perpetrator id [COL] people id [COL] date [COL] year [COL] location [COL] country [COL] killed [COL] injured'])
# print(tokenized[0].input_ids)
# print(tokenize(['[COL]'])[0].input_ids)
# print(tokenizer.convert_ids_to_tokens(tokenized[0].input_ids[0]))

df = pd.read_csv('./data/dev/data_join_no_special.csv')
train, test = train_test_split(df, test_size = 0.6, random_state = 123, shuffle = True) # TODO: change test size = 0.3
train_X, train_Y = train.iloc[:, 0], train.iloc[:, 1]
test_X, test_Y = test.iloc[:, 0], test.iloc[:, 1]

# test_X, test_Y = df.iloc[:, 0], df.iloc[:, 1]

# model = BertModel.from_pretrained('bert-base-uncased')

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

    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.bert.resize_token_embeddings(len(tokenizer))
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(768, 2)

  def forward(self, input_id, mask):
    _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
    dropout_output = self.dropout(pooled_output)
    linear_output = self.linear(dropout_output)

    return linear_output

device = 'cuda'

max_epochs = 3

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('mode', type=str)

  args = parser.parse_args()

  print(args.mode)

  if args.mode == 'train':
    train_batch_size = 40
    model = BertClassifier().to(device)
    print('finished downloading')
    dataset = LogDataset(train_X, train_Y)
    dataloader = data.DataLoader(dataset, batch_size = train_batch_size, shuffle = True)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=1e-5)

    for epoch in range(max_epochs):
      total_acc_train = 0

      for train_input, train_labels in tqdm(dataloader):
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
      
      print(f'\nTrain accuracy: {total_acc_train} / {train_X.shape[0]}')

    torch.save(model, './bert.pt')
  elif args.mode == 'test':
    model = torch.load('./bert.pt')
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