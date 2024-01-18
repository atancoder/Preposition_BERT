import copy

import torch
from torch.nn import functional

from utils import batch_to_token_ids, flatten_indices


def predict(model, batch, tokenizer, context_length, device):
    (
        orig_sentences_tokens,
        masked_sentences_tokens,
        masked_indices,
    ) = batch_to_token_ids(batch, tokenizer, context_length, device)
    model.eval()
    with torch.no_grad():
        logits, _ = model(masked_sentences_tokens)  # (B,T,C)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        flat_indices = flatten_indices(masked_indices, context_length)
        relevant_logits = logits[flat_indices]
        probabilities = functional.softmax(relevant_logits, dim=-1)
        predicted_tokens = torch.multinomial(probabilities, num_samples=1)
        sentences = []
        predicted_tokens_ptr = 0
        predicted_sentence_tokens = copy.deepcopy(masked_sentences_tokens)
        for batch_id in range(len(predicted_sentence_tokens)):
            relevant_masked_indices = masked_indices[batch_id]
            batch_predicted_tokens = predicted_tokens[
                predicted_tokens_ptr : predicted_tokens_ptr
                + len(relevant_masked_indices)
            ].reshape(-1)
            predicted_sentence_tokens[batch_id][
                relevant_masked_indices
            ] = batch_predicted_tokens
            predicted_tokens_ptr += len(relevant_masked_indices)

        sentences = tokenizer.batch_decode(predicted_sentence_tokens)

        for i in range(0, 10):
            orig_sentence = tokenizer.decode(orig_sentences_tokens[i])
            orig_sentence = orig_sentence[: orig_sentence.index(tokenizer.pad_token)]
            masked_sentence = tokenizer.decode(masked_sentences_tokens[i])
            masked_sentence = masked_sentence[
                : masked_sentence.index(tokenizer.pad_token)
            ]
            new_sentence = sentences[i][: sentences[i].index(tokenizer.pad_token)]
            print(f"Orig sentence: {orig_sentence}")
            print(f"Mask sentence: {masked_sentence}")
            print(f"Pred sentence: {new_sentence}\n\n")
        return sentences
