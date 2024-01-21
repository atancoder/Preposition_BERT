# type: ignore
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from utils import flatten_indices

EMBEDDING_SIZE = 768  # C aka channels
NUM_HEADS = 4
NUM_TRANSFORMER_BLOCKS = 4
MAX_CONTEXT_LENGTH = 128


class BERTModel(nn.Module):
    def __init__(self, vocab_size, context_length, device):
        super().__init__()
        self.context_length = context_length
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, EMBEDDING_SIZE)
        self.position_embedding_table = nn.Embedding(MAX_CONTEXT_LENGTH, EMBEDDING_SIZE)
        self.blocks = nn.Sequential(
            *[
                Block(EMBEDDING_SIZE, n_head=NUM_HEADS)
                for _ in range(NUM_TRANSFORMER_BLOCKS)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(EMBEDDING_SIZE)  # final layer norm
        self.final_linear = nn.Linear(EMBEDDING_SIZE, vocab_size)
        self.device = device

    def forward(self, sentence_tokens, target_info: Optional[Tuple] = None):
        B, T = sentence_tokens.shape
        token_emb = self.token_embedding_table(sentence_tokens)  # (B,T,C)

        # Convert all position integers into an embedding
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)
        x = token_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.final_layer_norm(x)  # (B,T,C)
        logits = self.final_linear(x)  # (B,T,vocab_size)

        if target_info is None:
            loss = None
        else:
            masked_indices, target_sentence_tokens = target_info
            flat_indices = flatten_indices(masked_indices, self.context_length)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)[flat_indices]
            target_sentence_tokens = target_sentence_tokens.view(B * T)[flat_indices]
            # Subset logits only to timesteps we care about
            loss = functional.cross_entropy(logits, target_sentence_tokens)

        return logits, loss


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


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.linear = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # concat column wise
        out = self.linear(out)
        return out


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
