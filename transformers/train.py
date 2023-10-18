import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.nn import functional as F
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
# ----------------------------
vocab_size = len(chars)
batch_size = 4
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rage = 1e-2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 200
n_embed = 32
# ----------------------------
# Simple tokenizer without subword encoding
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# print(encode('hello'))
# print(decode(encode('hello')))

data = torch.tensor(encode(text), dtype=torch.long)

n = int(len(data) * 0.9)  # 90% train, 10% test
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)


def get_batch(split):
    # generate small batch
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        model.eval()
        for k in range(eval_iters):
            X, y = get_batch(split)
            logits, loss = model(X, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

xb, yb = get_batch('train')

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly read off the logits of the next token in the sequence
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # BTC
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # T,C
        x = tok_emb + pos_emb  # BTC + T,C -> BTC
        logits = self.lm_head(x)

        if targets is None:
            loss = None
            return logits, loss
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B*T) # B*T is same as -1, but just in case
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1 , :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx


model = BigramLanguageModel()
m = model.to(device)
logits, loss = m(xb, yb)
# print(loss)
# g = decode(m.generate(torch.zeros((1,1),dtype=torch.long), 1000)[0].tolist())

optimizer = torch.optim.Adam(m.parameters(), lr=learning_rage)

batch_size = 32
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'Iter {iter} train loss: {losses["train"]:.4f} val loss: {losses["val"]:.4f}')

    xb,yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long,device=device)
g = decode(m.generate(context, 500)[0].tolist())
print(g)



















