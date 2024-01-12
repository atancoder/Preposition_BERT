import torch
from datasets import load_dataset
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cpu"
BATCH_SIZE = 64
import numpy as np

torch.manual_seed(1337)

EMBEDDING_SIZE = 512  # C aka channels
BATCH_SIZE = 16  # B
CONTEXT_LENGTH = 100  # T aka Time Steps
NUM_HEADS = 4
NUM_TRANSFORMER_BLOCKS = 2


class AttentionHead(nn.Module):
    """
    Uses K,Q,V Attention
    """

    def __init__(self, head_size: int):
        super().__init__()
        self.K = nn.Linear(EMBEDDING_SIZE, head_size, bias=False)
        self.Q = nn.Linear(EMBEDDING_SIZE, head_size, bias=False)
        self.V = nn.Linear(EMBEDDING_SIZE, head_size, bias=False)

    def forward(self, x):
        """
        x is the embedding
        """
        B, T, C = x.shape
        k = self.K(x)  # (B,T,C)
        q = self.Q(x)  # (B,T,C)
        # compute attention scores ("affinities")
        # k needs to be transposed to (B,C,T)
        att_scores = q @ k.transpose(-2, -1) * C**-0.5
        att_scores = functional.softmax(att_scores, dim=-1)  # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.V(x)  # (B,T,C)
        out = att_scores @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.linear = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # concat column wise
        out = self.linear(out)
        return out


class FeedFoward(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 4 * input_size),  # 4 is the scalar used in the paper
            nn.ReLU(),
            nn.Linear(4 * input_size, input_size),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        if n_embd % n_head != 0:
            raise Exception("Embedding size must be divisible by number of heads")
        head_size = n_embd // n_head
        self.multi_head_attention = MultiHeadAttention(n_head, head_size)
        self.feed_forward = FeedFoward(n_embd)
        self.ln_pre_attention = nn.LayerNorm(n_embd)
        self.ln_pre_feed_forward = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Use residual connections
        Layer norm prior the input prior to attention and feed forward
        """
        x = x + self.multi_head_attention(self.ln_pre_attention(x))
        x = x + self.feed_forward(self.ln_pre_feed_forward(x))
        return x


class BERTModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, EMBEDDING_SIZE)
        self.position_embedding_table = nn.Embedding(CONTEXT_LENGTH, EMBEDDING_SIZE)
        self.blocks = nn.Sequential(
            *[
                Block(EMBEDDING_SIZE, n_head=NUM_HEADS)
                for _ in range(NUM_TRANSFORMER_BLOCKS)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(EMBEDDING_SIZE)  # final layer norm
        self.final_linear = nn.Linear(EMBEDDING_SIZE, vocab_size)

    def forward(self, vocab_idx, target_vocab_indices=None):
        B, T = vocab_idx.shape

        # vocab_idx and target_vocab_indices are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(vocab_idx)  # (B,T,C)

        # Convert all position integers into an embedding
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.final_layer_norm(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if target_vocab_indices is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target_vocab_indices = target_vocab_indices.view(B * T)
            loss = functional.cross_entropy(logits, target_vocab_indices)

        return logits, loss


def get_books_dataloader():
    # Load the BookCorpus dataset
    bookcorpus = load_dataset("bookcorpus")

    # Access the train split
    train_dataset = bookcorpus["train"]
    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


def main():
    books_dataloader = get_books_dataloader()


if __name__ == "__main__":
    main()
