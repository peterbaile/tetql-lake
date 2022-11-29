# train with ranking loss

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, file_utils
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

tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)

def tokenize(text):
  return tokenizer(text, padding='max_length', max_length=300, return_tensors='pt')

EMBED_SIZE = {
  'bert-tiny': 128,
  'bert-base-uncased': 768,
  'roberta-base': 768,
  'roberta-large': 1024
}

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

class DevDataset(data.Dataset):
  def __init__(self, texts, labels):
    self.texts = [tokenize(text) for text in tqdm(texts)]
    self.labels = torch.tensor(labels.values)
  
  def __getitem__(self, index):
    return self.texts[index], self.labels[index]

  def __len__(self):
    return len(self.texts)

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

device = 'cuda'

max_epochs = 100

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--mode', type=str)
  parser.add_argument('--path', type=str)
  parser.add_argument('--devfile', type=str)
  parser.add_argument('--devpart', type=int)
  parser.add_argument('--addnegative', type=bool)
  parser.add_argument('--topk', type=int)

  args = parser.parse_args()

  print(f'mode: {args.mode}, source path: {args.path}, add negative: {args.addnegative}')
  learning_rate = 1e-6 # 1e-5 or 1e-6
  print(f'learning rate: {learning_rate}')
  
  if args.addnegative:
    MODEL_PATH = f'./data/{args.path}/{MODEL_TYPE}-ranking-negative.pt'
  else:
    MODEL_PATH = f'./data/{args.path}/{MODEL_TYPE}-ranking.pt'

  print(MODEL_TYPE, MODEL_PATH)

  if args.mode == 'train':
    if args.addnegative:
      train_df = pd.read_csv(f'./data/{args.path}/train_ranking_negative.csv')
      valid_df = pd.read_csv(f'./data/{args.path}/valid_ranking_negative.csv')
    else:
      train_df = pd.read_csv(f'./data/{args.path}/train_ranking.csv')
      valid_df = pd.read_csv(f'./data/{args.path}/valid_ranking.csv')
    
    train_X, train_Y = train_df.iloc[:, 0], train_df.iloc[:, 1]
    valid_X, valid_Y = valid_df.iloc[:, 0], valid_df.iloc[:, 1]

    train_batch_instance_size = 6
    valid_batch_instance_size = 10
    model = BertClassifier().to(device)
    print('finished loading model')
    train_dataset = TrainDataset(train_X, train_Y, train_batch_instance_size, args.addnegative)
    valid_dataset = TrainDataset(valid_X, valid_Y, valid_batch_instance_size, args.addnegative)

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

      torch.cuda.empty_cache()
      
      model.train()
      for batch_mask, batch_input_id, train_labels in tqdm(train_dataloader):
        train_labels = train_labels.squeeze(0).to(device)
        mask = batch_mask.squeeze(0).to(device)
        input_id = batch_input_id.squeeze(0).to(device)

        # print(mask.shape, input_id.shape, train_labels.shape)

        output = model(input_id, mask)

        # print(output.shape)

        num_instance = train_labels.shape[0]

        if args.addnegative:
          output = output.reshape((num_instance, num_instance * 2))
        else:
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

      del train_labels
      del mask
      del input_id
      
      model.eval()
      with torch.no_grad():
        for batch_mask, batch_input_id, valid_labels in tqdm(valid_dataloader):
          valid_labels = valid_labels.squeeze(0).to(device)
          mask = batch_mask.squeeze(0).to(device)
          input_id = batch_input_id.squeeze(0).to(device)

          output = model(input_id, mask)
          num_instance = valid_labels.shape[0]

          if args.addnegative:
            output = output.reshape((num_instance, num_instance * 2))
          else:
            output = output.reshape((num_instance, num_instance))

          loss = torch.logsumexp(output, dim=1) - torch.logsumexp(output[valid_labels], dim=1)

          valid_losses.append(loss.item())

      del valid_labels
      del mask
      del input_id
      
      train_loss = np.average(train_losses)
      valid_loss = np.average(valid_losses)

      avg_train_losses.append(train_loss)
      avg_valid_losses.append(valid_loss)

      print(f'\n epoch {epoch}: training loss {train_loss:.5f}, validation loss {valid_loss:.5f}')

      if best_valid_loss is None:
        torch.save(model, MODEL_PATH)
        print(f'model saved')
        best_valid_loss = valid_loss
      elif valid_loss >= best_valid_loss:
        patience_cnt += 1
        print(f'early stopping {patience_cnt}/{patience}')

        if patience_cnt >= patience:
          print('early stopping')
          break
      else:
        torch.save(model, MODEL_PATH)
        print(f'model saved')
        best_valid_loss = valid_loss
        patience_cnt = 0

  elif args.mode == 'dev':
    print(f'dev partition: {args.devpart}, dev file: {args.devfile}, topk: {args.topk}')
    model = torch.load(MODEL_PATH)
    dev_batch_size = 876 # this has to be the same as the number of candidates picked (876 no join, 100 idf)

    dev_df = pd.read_csv(f'./data/dev/{args.devfile}_ranking.csv')

    if args.devpart != -1:
      part_percent = 0.3
      cut = int(part_percent * (dev_df.shape[0] / dev_batch_size))
      if args.devpart == 0:
        dev_df = dev_df[:cut * dev_batch_size]
      elif args.devpart == 1:
        dev_df = dev_df[cut * dev_batch_size: 2*cut*dev_batch_size]
      else:
        dev_df = dev_df[2*cut*dev_batch_size:]

    dev_X, dev_Y = dev_df.iloc[:, 0], dev_df.iloc[:, 1]
    dev_dataset = DevDataset(dev_X, dev_Y)
    dev_dataloader = data.DataLoader(dev_dataset, batch_size = dev_batch_size)

    total_output = None
    total_acc_test = 0
    total_output_prob = None

    m = nn.LogSoftmax(dim=1).to(device)

    model.eval()
    with torch.no_grad():
      for test_input, test_label in tqdm(dev_dataloader):
        test_label = test_label.to(device)
        mask = test_input['attention_mask'].to(device)
        input_id = test_input['input_ids'].squeeze(1).to(device)

        raw_output = model(input_id, mask).squeeze(1)
        
        _, max_indices = torch.topk(raw_output, args.topk)
        max_indices = max_indices.tolist()
        # print(max_idx)

        output = [0 for _ in range(dev_batch_size)]

        for max_idx in max_indices:
          output[max_idx] = 1

        if total_output is None:
          total_output = output
          # total_output_prob = (raw_output.max()).cpu()
        else:
          total_output += output
          # total_output_prob = torch.vstack((total_output_prob, (raw_output.max()).cpu()))

    print(f'accuracy: {accuracy_score(dev_Y, total_output):.5f}')
    print(f'precision: {precision_score(dev_Y, total_output):.5f}')
    print(f'recall: {recall_score(dev_Y, total_output):.5f}')
    print(f'f1: {f1_score(dev_Y, total_output):.5f}')

    # np.save automatically add .npy extension
    np.save(f'./data/dev/{args.devfile}_label_{args.devpart}_ranking', dev_Y)
    np.save(f'./data/dev/{args.devfile}_output_{args.devpart}_ranking', total_output)
    print(f'output saved')
  
  elif args.mode == 'score':
    # print(file_utils.default_cache_path)

    num_parts = 3

    dev_outputs = [np.load(f'./data/dev/{args.devfile}_output_{i}_ranking.npy') for i in range(num_parts)]
    dev_labels = [np.load(f'./data/dev/{args.devfile}_label_{i}_ranking.npy') for i in range(num_parts)]

    for i in range(num_parts):
      output = np.hstack(dev_outputs)
      label = np.hstack(dev_labels)

    print(f'precision: {precision_score(label, output):.3f}')
    print(f'recall: {recall_score(label, output):.3f}')
    print(f'f1: {f1_score(label, output):.3f}')