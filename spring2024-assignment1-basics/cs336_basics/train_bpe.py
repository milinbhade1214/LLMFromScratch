"""
** Used to train the BPE model on the given text data

## Find pretoken
## Read text file -> return pretoken frequency table
## update byte tuple -> merge the byte tuple
## train_bpe -> train the BPE model on the text data
"""


import regex as re
from typing import Iterable
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from collections import Counter
import concurrent.futures

GPT2_PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _find_pretokens(text: str):
    ## Returns the pretoken frequency table
    logging.info(f"Pre-tokenizing the text of length {len(text)}")
    return Counter(re.findall(GPT2_PRETOKENIZER_PATTERN, text))

def _read_text_file(input_path: str, num_workers: int, special_tokens: Iterable[str]):
    with open(input_path, 'r') as f:
        text = f.read()

    ## Remove special tokens from the text
    for token in special_tokens:
        text = text.replace(token, "")

    logging.info("Initializing pretoken frequency table")
    if num_workers == 1:
        pretokens =  _find_pretokens(text)
        print(pretokens)
    else:
        chunk_size = len(text) // num_workers  
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            pretokens = executor.map(_find_pretokens, text_chunks)
        pretokens = sum(pretokens, Counter())

    gen_tuple_of_bytes = lambda pretoken: tuple([bytes([b]) for b in pretoken.encode("utf-8")])
    pretoken_freq = {}
    for pretoken, freq in pretokens.items():
        pretoken_freq[gen_tuple_of_bytes(pretoken)] = freq
    return pretoken_freq

def _update_byte_tuple(byte_tuple: Iterable[bytes], merge_loc: int):
    assert len(byte_tuple) > 1, "Cannot merge a byte tuple with length less than 2."
    prefix = byte_tuple[:merge_loc] 
    tomerge = byte_tuple[merge_loc:merge_loc+2]
    suffix = byte_tuple[merge_loc+2:]
    new_byte_tuple = prefix + (b''.join(tomerge),) + suffix
    return new_byte_tuple, prefix, suffix

def train_bpe(input_path:str, vocab_size:int, special_tokens: Iterable[str], progress_bar:bool=False, num_workers:int=1):
    vocab = {i: bytes([i]) for i in range(256)}

    for i, token in enumerate(special_tokens):
        vocab[i+256] = token.encode("utf-8")
    
    pretoken_freq = _read_text_file(input_path, num_workers, special_tokens)
    logging.info("Initialized the pretoken frequency table")

    pair_freq = Counter()
    for pretoken, freq in tqdm(pretoken_freq.items(), disable=not progress_bar):
        for i in range(len(pretoken)-1):
            pair = pretoken[i: i+2]
            if pair not in pair_freq:
                pair_freq[pair] = 0
            pair_freq[pair] += freq
    
    logging.info("Performing BPE algorithm")

    pre_merge_vocab_size = len(vocab)   
    pbar = tqdm(total=vocab_size-pre_merge_vocab_size) if progress_bar else None
    merges = []

    while len(vocab) < vocab_size:
        #find the most freq pair
        most_freq_pair = max(pair_freq, key=lambda k: (pair_freq[k], k))
        #add the pair to merge list
        merges.append(most_freq_pair)
        
        new_id = max(vocab.keys()) + 1
        vocab[new_id] = b''.join(most_freq_pair)

        #update the pretoken frequency table and pair frequency table
        new_pretoken_freq = {}
        for pretoken_tuple, freq in pretoken_freq.items():
            i = 0
            while i < len(pretoken_tuple)-1:
                pair = pretoken_tuple[i:i+2]

                if pair == most_freq_pair:
                    pretoken_tuple, prefix, suffix = _update_byte_tuple(pretoken_tuple, i)

                    # Update the pair frequency table
                    if prefix: 
                        add_pair = (prefix[-1], vocab[new_id])
                        pair_freq[add_pair] = pair_freq.get(add_pair, 0) + freq
                        del_pair = (prefix[-1], most_freq_pair[0])
                        pair_freq[del_pair] -= freq
                    if suffix:
                        add_pair = (vocab[new_id], suffix[0])
                        pair_freq[add_pair] = pair_freq.get(add_pair, 0) + freq
                        del_pair = (most_freq_pair[1], suffix[0])
                        pair_freq[del_pair] -= freq
                    pair_freq[most_freq_pair] -= freq
                i+=1
            # Update the pre-token frequency table
            new_pretoken_freq[pretoken_tuple] = freq
        pretoken_freq = new_pretoken_freq
        pbar.update(len(vocab) - pre_merge_vocab_size - pbar.n) if progress_bar else None
    pbar.close() if progress_bar else None

    return vocab, merges
                    




import time

vocab_size = 100

path = "./demo.txt"
special_tokens = ["<|endoftext|>"] 

res = _read_text_file(path, 2, special_tokens)

start_time = time.time()
train_bpe(path, vocab_size, special_tokens)
print("Time taken: " , time.time() - start_time)
