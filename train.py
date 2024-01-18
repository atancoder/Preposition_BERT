import time

import torch

from utils import batch_to_token_ids


def train(model, dataloader, tokenizer, context_length, learning_rate, device):
    start = time.time()
    size = len(dataloader.dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for batch_id, batch in enumerate(dataloader):
        (
            orig_sentences_tokens,
            masked_sentences_tokens,
            masked_indices,
        ) = batch_to_token_ids(batch, tokenizer, context_length, device)

        _, loss = model(
            masked_sentences_tokens, (masked_indices, orig_sentences_tokens)
        )

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch_id % 100 == 0:
            loss, current = loss.item(), batch_id + 1
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"{int((time.time() - start) / 60)} minutes have elapsed")
            if batch_id % 1000 == 0:
                torch.save(model, "model.pt")
