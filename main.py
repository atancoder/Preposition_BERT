import os

import torch
from transformers import BertTokenizerFast

from model import BERTModel
from predict import predict
from train import train
from utils import get_books_dataloader

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {DEVICE} device")
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
CONTEXT_LENGTH = 512  # T aka Time Steps

torch.manual_seed(1337)
MODEL_PT_FILE = "model.pt"


def main():
    books_dataloader = get_books_dataloader(BATCH_SIZE)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BERTModel(tokenizer.vocab_size, CONTEXT_LENGTH, DEVICE)
    if os.path.exists(MODEL_PT_FILE):
        print(f"Loading existing model: {MODEL_PT_FILE}")
        model.load_state_dict(
            torch.load(MODEL_PT_FILE, map_location=torch.device(DEVICE))
        )
    model.to(DEVICE)
    # train(model, books_dataloader, tokenizer, CONTEXT_LENGTH, LEARNING_RATE, DEVICE)
    predict(model, next(iter(books_dataloader)), tokenizer, CONTEXT_LENGTH, DEVICE)


if __name__ == "__main__":
    main()
