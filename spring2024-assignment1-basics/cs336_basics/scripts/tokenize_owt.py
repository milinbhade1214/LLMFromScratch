from typing import Iterable
import numpy as np
import time
from tqdm import tqdm
import wandb

from cs336_basics.tokenizer import Tokenizer

owt = {
    'train':'data/raw/owt_train.txt',
    'val':'data/raw/owt_valid.txt',
    'vocab_filepath': 'data/out/owt_vocab.json',
    'merges_filepath': 'data/out/owt_merges.txt',
    'special_tokens': ['<|endoftext|>']
}
# wandb setup
wandb_name = 'cs336_basics'
wandb_run_name = 'tokenize_owt'
wandb_logging = True
# wandb logging
if wandb_logging:
    wandb.init(project=wandb_name, name=wandb_run_name, config=owt)


tokenizer = Tokenizer.from_files(**owt)

for split in ['train', 'val']:
    with open(owt[split]) as f:
        text = f.read()
    encoded = tokenizer.encode(text, progress_bar=True)

    # save the ids
    total_batches = 1024
    batch_size = len(encoded) // total_batches
    arr = np.memmap(f'data/owt/{split}.bin', dtype=np.uint16, mode='w+', shape=(len(encoded),))
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'Writing {split}.bin'):
        batch = encoded[idx:idx+batch_size]
        arr[idx:idx+batch_size] = batch
        idx += batch_size
arr.flush()