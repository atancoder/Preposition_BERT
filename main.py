# type: ignore
import copy
import random

import torch
from datasets import load_dataset
from torch.nn import functional
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from model import CONTEXT_LENGTH, DEVICE, BERTModel
from prepositions import PREPOSITIONS_LIST

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
torch.manual_seed(1337)

import numpy as np


def get_books_dataloader():
    # Load the BookCorpus dataset
    bookcorpus = load_dataset("bookcorpus")

    # Access the train split
    train_dataset = bookcorpus["train"]
    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)


def mask_out(token_ids, tokenizer, preposition_tokens_ids, vocab_token_ids):
    """
    Masks out prepositions
    Returns the sentence token_ids with some [MASK] token_ids and 2D array of token_ids masked
    """
    token_ids = copy.deepcopy(token_ids)
    masked_indices = []
    for token_batch in token_ids:
        batch_masked_indicies = []
        for i in range(len(token_batch)):
            token_id = token_batch[i].item()
            if token_id == tokenizer.pad_token_id:
                break
            if token_id in preposition_tokens_ids:
                rand = random.random()
                if rand <= 0.5:
                    # mask out
                    token_batch[i] = tokenizer.mask_token_id
                elif rand <= 0.8:
                    random_token = random.choice(vocab_token_ids)
                    token_batch[i] = random_token
                else:
                    pass
                batch_masked_indicies.append(i)
        masked_indices.append(batch_masked_indicies)
    return token_ids, masked_indices


def right_pad(token_ids, pad_token_id):
    num_to_pad = CONTEXT_LENGTH - len(token_ids)
    token_ids += [pad_token_id] * num_to_pad
    return token_ids


def get_preposition_token_ids(tokenizer):
    return set(tokenizer.convert_tokens_to_ids(PREPOSITIONS_LIST))


def get_vocab_token_ids(tokenizer):
    return tokenizer.convert_tokens_to_ids(tokenizer.vocab)


def batch_to_token_ids(batch_sentences, tokenizer):
    preposition_tokens_ids = get_preposition_token_ids(tokenizer)
    vocab_token_ids = get_vocab_token_ids(tokenizer)
    orig_sentences = batch_sentences["text"]
    orig_sentences_tokens = [
        right_pad(tokenizer.encode(sentence), tokenizer.pad_token_id)
        for sentence in orig_sentences
    ]
    orig_sentences_tokens = torch.tensor(orig_sentences_tokens)
    masked_sentences_tokens, masked_indices = mask_out(
        orig_sentences_tokens, tokenizer, preposition_tokens_ids, vocab_token_ids
    )
    return (
        orig_sentences_tokens.to(DEVICE),
        masked_sentences_tokens.to(DEVICE),
        masked_indices,
    )


def predict(model, batch, tokenizer):
    (
        orig_sentences_tokens,
        masked_sentences_tokens,
        masked_indices,
    ) = batch_to_token_ids(batch, tokenizer)

    model.eval()
    with torch.no_grad():
        logits, _ = model(masked_sentences_tokens)  # (B,T,C)
        B, T, C = logits.shape
        probabilities = functional.softmax(logits, dim=-1)
        probabilities = probabilities.view(B * T, C)
        predicted_tokens = torch.multinomial(probabilities, num_samples=1)
        predicted_tokens = predicted_tokens.view(B, T)
        sentences = tokenizer.batch_decode(predicted_tokens)
        return sentences


def train(model, dataloader, tokenizer):
    size = len(dataloader.dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for batch_id, batch in enumerate(dataloader):
        (
            orig_sentences_tokens,
            masked_sentences_tokens,
            masked_indices,
        ) = batch_to_token_ids(batch, tokenizer)

        _, loss = model(
            masked_sentences_tokens, (masked_indices, orig_sentences_tokens)
        )

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_id % 2 == 0:
            loss, current = loss.item(), batch_id + 1
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def main():
    books_dataloader = get_books_dataloader()
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BERTModel(tokenizer.vocab_size)
    model.to(DEVICE)
    train(model, books_dataloader, tokenizer)
    # predict(model, next(iter(books_dataloader)), tokenizer)


if __name__ == "__main__":
    main()
