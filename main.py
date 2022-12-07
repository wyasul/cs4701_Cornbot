from collections import Counter, namedtuple
from itertools import chain
import json
import math
import os
from pathlib import Path
from tqdm.notebook import tqdm, trange
from typing import List, Tuple, Dict, Set, Union
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

# load json data
# Opening JSON file
f = open('cornbot_train.json')
train = json.load(f) # returns JSON object as a dictionary

f = open('cornbot_validation.json')
val = json.load(f) # returns JSON object as a dictionary

f = open('cornbot_test.json')
test = json.load(f) # returns JSON object as a dictionary

# TOEKNIZE
text = []
text_str = []
text_untokenized = []
tags = []
flag = False
for intent in train["intents"]:
    for pattern in intent["patterns"]:
        text.append(pattern.split())
        for word in pattern.split():
            text_str.append(word)
        tags.append(intent["tag"])
        text_untokenized.append(pattern)

val_text = []
val_text_untokenized = []
val_tags = []
for intent in val["intents"]:
    for pattern in intent["patterns"]:
        val_text.append(pattern.split())
        val_tags.append(intent["tag"])
        val_text_untokenized.append(pattern)

test_text = []
test_text_untokenized = []
test_tags = []
for intent in test["intents"]:
    for pattern in intent["patterns"]:
        test_text.append(pattern.split())
        test_tags.append(intent["tag"])
        test_text_untokenized.append(pattern)
#print(test_text_untokenized)

# CLASS VOCAB
UNK = "<UNK>"
PAD = "<PAD>"

class Vocab(object):
    def __init__(self, word2id=None):
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id[PAD] = 0   # Pad Token
            self.word2id[UNK] = 1   # Unknown Token
        self.unk_id = self.word2id[UNK]
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def word_from_id(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    @staticmethod
    def from_corpus(corpus): 
        vocab_entry = Vocab()
        top_words = set(corpus)
        for word in top_words:
            vocab_entry.add(word.translate(str.maketrans('', '', '.?,:;!')))
        return vocab_entry

def tokenize_sentences(sentences, vocab):
    tokenized_sents = []
    for sentence in sentences:
        tokenized_sentence = [vocab[w] for w in sentence]
        tokenized_sents.append(tokenized_sentence)

    output = []

    for i in range(len(tokenized_sents)):
        output.append([float(0) for v in range(vocab.__len__())])
        for j in range(len(tokenized_sents[i])):
            output[i][tokenized_sents[i][j]] = float(1)

    return output

# BERT
from sentence_transformers import SentenceTransformer
def bert_sentences(sentences):
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return bert_model.encode(sentences)

# PADDING
max_len = 0

for sentence in text:
  if len(sentence) > max_len:
    max_len = len(sentence)

def pad_sents(sents, pad_token, max_len):
    for sent in sents:
      for i in range(max_len - len(sent)):
        sent.append(pad_token)

    return sents

# LABELS
category_map = {}
counter = 0
for tag in tags:
    if tag not in category_map:
        category_map[tag] = counter
        counter += 1

print(category_map)

def encode_tag(tag_data):
  label_ints = []

  for tag in tag_data:
    label_ints.append(category_map[tag])

  #return label_ints

# INITIALIZE TRAIN VOCAB
torch.manual_seed(123)
print('initialize train vocabulary ..')
vocab = Vocab.from_corpus(text_str)

# PROCESS
from zmq.constants import PROTOCOL_ERROR_ZMTP_INVALID_SEQUENCE

processed_train = dict()
processed_train['tags'] = encode_tag(tags)
processed_train['text'] = pad_sents(text, PAD, max_len)
processed_train['text-tokenized'] = tokenize_sentences(processed_train['text'], vocab)

processed_val = dict()
processed_val['tags'] = encode_tag(val_tags)
processed_val['text'] = pad_sents(val_text, PAD, max_len)
processed_val['text-tokenized'] = tokenize_sentences(processed_val['text'], vocab)

processed_train_bert = dict()
processed_train_bert['tags'] = processed_train['tags']
processed_train_bert['text-tokenized'] = bert_sentences(text_untokenized)

processed_val_bert = dict()
processed_val_bert['tags'] = processed_val['tags']
processed_val_bert['text-tokenized'] = bert_sentences(val_text_untokenized)

processed_test = dict()
processed_test['tags'] = encode_tag(test_tags)
processed_test['text'] = pad_sents(test_text, PAD, max_len)
processed_test['text-tokenized'] = tokenize_sentences(processed_test['text'], vocab)

processed_test_bert = dict()
processed_test_bert['tags'] = processed_test['tags']
processed_test_bert['text-tokenized'] = bert_sentences(test_text_untokenized)

# LANG DATA SET
class LanguageDataset(Dataset):
    def __init__(self, text, tag):
        self.text = torch.tensor(text)
        self.tag = torch.tensor(tag)
        self.len = len(text)
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if not self.tag.numel():
             return self.text[index], []
        else:
            return self.text[index], self.tag[index]

# LOAD DATA
def get_data_loaders(preprocessed_data, batch_size=1, shuffle=False):
    dataset = LanguageDataset(preprocessed_data['text-tokenized'], preprocessed_data['tags'])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

# SET VALUES
torch.manual_seed(123)

batch_size = 200
train_loader = get_data_loaders(processed_train, batch_size=batch_size)
val_loader = get_data_loaders(processed_val, batch_size=batch_size)
train_loader_bert = get_data_loaders(processed_train_bert, batch_size=batch_size)
val_loader_bert = get_data_loaders(processed_val_bert, batch_size=batch_size)
test_loader = get_data_loaders(processed_test, batch_size=batch_size)
test_loader_bert = get_data_loaders(processed_test_bert, batch_size=batch_size)

# FFNN
# Lambda to switch to GPU if available
get_device = lambda : "cuda:0" if torch.cuda.is_available() else "cpu"

# Setting seed ***DO NOT MODIFY***
torch.manual_seed(123)

class FFNN(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, vocab_size, nhidden = 1):
		super(FFNN, self).__init__()
		self.W1 = torch.nn.Linear(input_dim, hidden_dim)
		self.W2 = torch.nn.Linear(hidden_dim, hidden_dim)
		self.W3 = torch.nn.Linear(hidden_dim, output_dim)
		self.softmax = torch.nn.LogSoftmax(-1)
		self.loss = torch.nn.CrossEntropyLoss()
		self.nhidden = nhidden
  
	def compute_Loss(self, predicted_vector, gold_label):
		return self.loss(predicted_vector, gold_label)

	def forward(self, input_vector):
		z1 = self.W1(input_vector)
		z2 = self.W2(z1)
		for i in range(self.nhidden - 1):
			z2 = self.W2(z2)
		z3 = self.W3(z2)
		predicted_vector = self.softmax(z3)
		return predicted_vector

	def load_model(self, save_path, is_state_dict=False):
		if not is_state_dict:
			saved_model = torch.load(save_path)
			self.load_state_dict(saved_model.state_dict())
		else:
			self.load_state_dict(torch.load(save_path))

	def save_model(self, save_path, is_state_dict=False):
		if is_state_dict:
			torch.save(self.state_dict(), save_path)
		else:
			torch.save(self, save_path)

# EPOCH
torch.manual_seed(123)
def train_epoch(model, train_loader, optimizer):
  model.train()
  total = 0
  batch = 0
  total_loss = 0
  correct = 0
  for (input_batch, expected_out) in tqdm(train_loader, leave=False, desc="Training Batches"):
    optimizer.zero_grad()
    batch += 1
    output = model(input_batch.to(get_device())).to(get_device())
    loss = model.compute_Loss(output, expected_out)
    total += expected_out.size(dim=0)
    _, predicted = torch.max(output, -1)
    correct += (expected_out.to("cpu") == predicted.to("cpu")).cpu().numpy().sum()
    total_loss += loss.item()
    loss.backward()

    optimizer.step()
  print("Loss: " + str(total_loss/batch))
  print("Training Accuracy: " + str(correct/total))
  return total_loss/batch

# Lambda to switch to GPU if available
get_device = lambda : "cuda:0" if torch.cuda.is_available() else "cpu"
device = get_device()

# EVALUATION
torch.manual_seed(123)
def evaluation(model, val_loader, optimizer, test = False):
  model.eval()
  loss = 0
  correct = 0
  total = 0
  dataset = "Validation"
  if test:
      dataset = "Test"

  for (input_batch, expected_out) in tqdm(val_loader, leave=False, desc=dataset + " Batches"):
    output = model(input_batch.to(get_device())).to(get_device())
    total += expected_out.shape[0]
    _, predicted = torch.max(output, -1)
    correct += (expected_out.to("cpu") == predicted.to("cpu")).cpu().numpy().sum()
    loss += model.compute_Loss(output, expected_out)
  loss /= len(val_loader)
  print(dataset + " Loss: " + str(loss.item()))
  print(dataset + " Accuracy: " + str(correct/total))
  print()
  return loss.item()

# TRAIN AND EVALUATE
torch.manual_seed(123)
def train_and_evaluate(number_of_epochs, model, train_loader, val_loader, min_loss=0, lr=.001):
  optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
  loss_values = [[],[]]
  for epoch in trange(number_of_epochs, desc="Epochs"):
    cur_loss = train_epoch(model, train_loader, optimizer)
    loss_values[0].append(cur_loss)
    cur_loss_val = evaluation(model, val_loader, optimizer)
    loss_values[1].append(cur_loss_val)
    if cur_loss <= min_loss: return loss_values
  return loss_values

# EPOCH
torch.manual_seed(123)

vocab_size = len(vocab)
inp_dim = vocab_size # Don't change
h = 500 # Hidden dimension: feel free to change
out_dim = len(category_map) # Don't change
nhidden = 1 # Number of hidden layers: feel free to change
lr = 0.008 # Learning rate: feel free to change
model = FFNN(inp_dim, h, out_dim, vocab_size, nhidden)

nepochs = 10
trained_model = train_and_evaluate(nepochs, model, train_loader, val_loader, lr = lr)

# TRAIN BERT
embed_dim = 384
h_bert = 200
lr_bert = 0.010
nhidden_bert = 1
nepochs_bert = 20
model2 = FFNN(embed_dim, h_bert, out_dim, vocab_size, nhidden_bert)
trained_model_bert = train_and_evaluate(nepochs_bert, model2, train_loader_bert, val_loader_bert, lr = lr_bert)

# ACTUAL CHAT BOT
import random
print("Welcome! Enter a message for Cornbot or type \"quit\" to exit.")
while True:
    inp = input("You: ")
    if inp == 'quit':
        break
    padded_inp = pad_sents([inp.split()], PAD, max_len)
    vectorized_input = bert_sentences(padded_inp)
    model_output = model2(torch.Tensor(vectorized_input))
    _, tag_int = torch.max(model_output, -1)
    tag_str = ''
    for key in category_map:
        if category_map[key] == tag_int:
            tag_str = key
    output = ''
    for intent in train["intents"]:
        if intent["tag"] == tag_str:
            index = random.randint(0, len(intent["responses"]) - 1)
            output = intent["responses"][index]
    print("Cornbot:", output)








