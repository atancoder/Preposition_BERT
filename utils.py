# type: ignore
import copy
import random
from typing import List

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from prepositions import PREPOSITIONS_LIST


def flatten_indices(batch_indices: List[List[int]], context_length: int) -> List[int]:
    """
    Flattens the batch_indices list into 1D
    Assumes each batch has context_length indices

    e.g [[1,3], [0, 3]]  -> [1,3,4,7]

    [0,3] -> [4,7] because it's the 2nd batch
    """
    flattened_indices: List[int] = []
    for batch_id, batch in enumerate(batch_indices):
        batch_start_idx = batch_id * context_length
        new_indices = [batch_start_idx + idx for idx in batch]
        flattened_indices += new_indices
    return flattened_indices


def get_books_dataloader(batch_size):
    # Load the BookCorpus dataset
    bookcorpus = load_dataset("bookcorpus")

    # Access the train split
    train_dataset = bookcorpus["train"]
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


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


def right_pad_and_truncate(token_ids, pad_token_id, context_length):
    """
    Ensures token length = context length
    will pad with 0s or truncate the tokens
    """
    num_to_pad = context_length - len(token_ids)
    if num_to_pad > 0:
        token_ids += [pad_token_id] * num_to_pad
    else:
        print(f"Truncating sentence of size: {len(token_ids)}")
        token_ids = token_ids[:context_length]
    return token_ids


def get_preposition_token_ids(tokenizer):
    return set(tokenizer.convert_tokens_to_ids(PREPOSITIONS_LIST))


def get_vocab_token_ids(tokenizer):
    return tokenizer.convert_tokens_to_ids(tokenizer.vocab)


def batch_to_token_ids(batch_sentences, tokenizer, context_length, device):
    preposition_tokens_ids = get_preposition_token_ids(tokenizer)
    vocab_token_ids = get_vocab_token_ids(tokenizer)
    orig_sentences = batch_sentences["text"]
    orig_sentences_tokens = [
        right_pad_and_truncate(
            tokenizer.encode(sentence), tokenizer.pad_token_id, context_length
        )
        for sentence in orig_sentences
    ]
    orig_sentences_tokens = torch.tensor(orig_sentences_tokens)
    masked_sentences_tokens, masked_indices = mask_out(
        orig_sentences_tokens, tokenizer, preposition_tokens_ids, vocab_token_ids
    )
    return (
        orig_sentences_tokens.to(device),
        masked_sentences_tokens.to(device),
        masked_indices,
    )
