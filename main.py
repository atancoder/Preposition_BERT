# type: ignore
import copy
import random

import sentencepiece as spm
import torch
from datasets import load_dataset
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizerFast

from model import CONTEXT_LENGTH, BERTModel
from prepositions import PREPOSITIONS_LIST

BATCH_SIZE = 64
torch.manual_seed(1337)

import numpy as np


def get_books_dataloader():
    # Load the BookCorpus dataset
    bookcorpus = load_dataset("bookcorpus")

    # Access the train split
    train_dataset = bookcorpus["train"]
    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)


def mask_out(tokens, tokenizer, preposition_tokens, vocab_list):
    """
    Masks out prepositions
    Returns the sentence tokens with some [MASK] tokens and 2D array of tokens masked
    """
    tokens = copy.deepcopy(tokens)
    masked_indices = []
    for token_batch in tokens:
        batch_masked_indicies = []
        for i in range(len(token_batch)):
            token = token_batch[i]
            if token == tokenizer.pad_token:
                break
            if token in preposition_tokens:
                rand = random.random()
                if rand <= 0.5:
                    # mask out
                    token_batch[i] = tokenizer.mask_token
                elif rand <= 0.8:
                    random_token = random.choice(vocab_list)
                    token_batch[i] = random_token
                else:
                    pass
                batch_masked_indicies.append(i)
        masked_indices.append(batch_masked_indicies)
    return tokens, masked_indices


def right_pad(token_sentence, tokenizer):
    num_to_pad = CONTEXT_LENGTH - len(token_sentence)
    token_sentence += [tokenizer.pad_token] * num_to_pad
    return token_sentence


def get_preposition_tokens(tokenizer):
    tokens = set()
    for p in PREPOSITIONS_LIST:
        tokens |= set(tokenizer.tokenize(p))
    return tokens


def main():
    books_dataloader = get_books_dataloader()
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    vocab_list = list(tokenizer.vocab.keys())
    model = BERTModel(tokenizer.vocab_size)
    preposition_tokens = get_preposition_tokens(tokenizer)
    for batch in books_dataloader:
        orig_sentences = batch["text"]
        token_orig_sentences = [
            right_pad(tokenizer.tokenize(sentence), tokenizer)
            for sentence in orig_sentences
        ]
        token_masked_sentences, masked_indices = mask_out(
            token_orig_sentences, tokenizer, preposition_tokens, vocab_list
        )
        token_id_orig_sentences = [
            tokenizer.convert_tokens_to_ids(tokens) for tokens in token_orig_sentences
        ]
        token_id_masked_sentences = [
            tokenizer.convert_tokens_to_ids(tokens) for tokens in token_masked_sentences
        ]
        import pdb

        pdb.set_trace()
        model(token_id_masked_sentences, (masked_indices, token_id_orig_sentences))


if __name__ == "__main__":
    main()
