from util import tokenizer
from source import model

import torch
import numpy as np
import random

from inference import Session

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


DIM = 768
HEADS = 12
BLOCKS = 24
SEQUENCE_LENGTH = 1024

t = tokenizer.BPETokenizer(None, load_if_exists=True)

m = model.EnergyTransformer(
    DEVICE,
    DIM,
    HEADS,
    BLOCKS,
    t.get_vocab_size(),
    SEQUENCE_LENGTH,
)

model_save_path = "model.pt"


def train():
    m.fit(
        "data/tokenized/train",
        None,
        batch_size=80,
        augmentation_methods_mix=(.4, .3, .3), # sequence, nearby, unigram
        max_lr=2e-4,
        warmup_steps=800,
        start_lr_div_factor=25.0,
        end_lr_div_factor=10.0,
        max_tokens=m.count_parameters() * 20,
        model_save_path=model_save_path,
    )


def inference():
    m.load(model_save_path)
    session = Session(m, t)
    session.message(input("Enter prompt:"), role="system")
    session.generate_exact(p=0.002, logs=True)


if __name__ == '__main__':
    train()
    inference()
