from typing import Iterable
import numpy as np
import time
from tqdm import tqdm
import os

from cs336_basics.tokenizer import Tokenizer

tinystory = {
    'train':'data/TinyStoriesV2-GPT4-train.txt',
    'val':'data/TinyStoriesV2-GPT4-valid.txt',
    'vocab_filepath': 'data/out/tinystories_vocab.json',
    'merges_filepath': 'data/out/tinystories_merges.txt',
    'special_tokens': ['<|endoftext|>']
}
# Create output directory
os.makedirs('data/ts', exist_ok=True)

tokenizer = Tokenizer.from_files(**tinystory)

for split in ['train', 'val']:
    with open(tinystory[split]) as f:
        text = f.read()
    encoded = tokenizer.encode(text, progress_bar=True)

    # save the ids
    total_batches = 1024
    batch_size = len(encoded) // total_batches
    arr = np.memmap(f'data/ts/{split}.bin', dtype=np.uint16, mode='w+', shape=(len(encoded),))
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'Writing {split}.bin'):
        batch = encoded[idx:idx+batch_size]
        arr[idx:idx+batch_size] = batch
        idx += batch_size
arr.flush()