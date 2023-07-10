from pathlib import Path

import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
context_window = (
    256  # aka T or block_size, what is the maximum context length for predictions?
)
max_iters = 5000
eval_interval = 500
embed_dim = 384  # aka C
num_heads = 6
num_blocks = 6
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
text = Path("data/tinyshakespeare.txt").read_text()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - context_window, (batch_size,))
    x = torch.stack([data[i : i + context_window] for i in ix])
    y = torch.stack([data[i + 1 : i + context_window + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones((context_window, context_window)))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x.shape == (B, T, embed_dim)"""
        B, T, embed_dim = x.shape
        k = self.key(x)  # (B, T, embed_dim) -> (B, T, head_size)
        q = self.query(x)  # (B, T, embed_dim) -> (B, T, head_size)
        v = self.value(x)  # (B, T, embed_dim) -> (B, T, head_size)

        k_transp = k.transpose(1, 2)  # (B, head_size, T)
        wei = q @ k_transp  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = wei / embed_dim**0.5  # scale by sqrt(embed_dim), so that variance is ~1

        wei = self.dropout(wei)

        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x.shape == (B, T, embed_dim)"""
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa_heads = MultiHeadAttention(num_heads, head_size=head_size)
        self.ff = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))  # (B, T, embed_dim)
        x = x + self.ff(self.ln2(x))  # (B, T, embed_dim)
        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(
            vocab_size, embed_dim
        )  # (vocab_size, embed_dim)
        self.pos_embedding_table = nn.Embedding(
            context_window, embed_dim
        )  # (T, embed_dim)

        self.blocks = nn.Sequential(
            *[Block(embed_dim, num_heads) for _ in range(num_blocks)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_embeddings = self.token_embedding_table(idx)  # (B, T) -> (B, T, embed_dim)
        pos_embeddings = self.pos_embedding_table(
            torch.arange(T, device=device)
        )  # (T,) -> (T, embed_dim)
        x = (
            tok_embeddings + pos_embeddings
        )  # (B, T, embed_dim) + (T, embed_dim) -> (B, T, embed_dim)
        x = self.blocks(x)  # (B, T, embed_dim) -> (B, T, embed_dim)
        x = self.ln_f(x)  # (B, T, embed_dim) -> (B, T, embed_dim)
        logits = self.lm_head(x)  # (B,T,embed_dim) -> (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            c_window = idx[:, -context_window:]  # don't take more than context window
            # get the predictions
            logits, loss = self(c_window)  # (B, T) -> (B, T, vocab_size)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, vocab_size)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in tqdm.tqdm(range(max_iters)):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
