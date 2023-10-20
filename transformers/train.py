import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.utils.data as data
from collections import Counter

with open('./transformers/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
# ----------------------------
vocab_size = len(chars)
batch_size = 64
block_size =256
max_iters = 10000
eval_interval = 500
n_layer = 6
n_head = 6
# Dropout added since model started overfitting
dropout = 0.2
learning_rage = 3e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 200
n_embed = 384
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


class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size,bias=False)
        self.query = nn.Linear(n_embed, head_size,bias=False)
        self.value = nn.Linear(n_embed, head_size,bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # B,T,C
        q = self.query(x) # B,T,C
        wei = q @ k.transpose(-2,-1) *C**-0.5  # B,T,C @ B,C,T -> B,T,T
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # B,T,T
        wei = self.dropout(wei)
        v = self.value(x) # B,T,C
        out = wei @ v # B,T,T @ B,T,C -> B,T,C

        return out



class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # B,T,C
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
            )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self,x):
        # Residual connection
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly read off the logits of the next token in the sequence
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        # self.sa_heads = MultiHeadAttention(4, n_embed//4) # 4 heads, 8/4=2
        # self.ffwd = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # BTC
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # T,C
        x = tok_emb + pos_emb  # BTC + T,C -> BTC
        # x = self.sa_heads(x)
        x = self.blocks(x) # BTC 
        x = self.ln_f(x) # BTC
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
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
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
g = decode(m.generate(context, 1000)[0].tolist())
print(g)



















