import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.utils.data as data
import re
from collections import Counter

with open('./transformers/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)


# Simple tokenizer without subword encoding
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# print(encode('hello'))
# print(decode(encode('hello')))

data = torch.tensor(encode(text), dtype=torch.long)

n = int(len(data) * 0.9) #90% train, 10% test
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
    #generate small batch
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs')
print(xb)
print('targets')
print(yb)
print('----------')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} target is {target}")