import re 
from collections import defaultdict 
import string
from typing import Iterable


def get_stats(vocab): 
    """ 
    Given a vocabulary (dictionary mapping words to frequency counts), returns a  
    dictionary of tuples representing the frequency count of pairs of characters  
    in the vocabulary. 
    """
    pairs = defaultdict(int) 
    for word, freq in vocab.items(): 
        chars = word.split() # split the word by any white space
        for i in range(len(chars)-1): 
            pairs[chars[i], chars[i+1]] += freq 
    return pairs 
  
def merge_vocab(token_pair, v_in): 
    """ 
    Given a pair of characters and a vocabulary, returns a new vocabulary with the  
    pair of characters merged together wherever they appear. 
    """
    v_out = defaultdict(int)  
    bigram = re.escape(' '.join(token_pair)) 
    new_token = ''.join(token_pair)
    # search for every occurance of bigram (token pairs with a space), 
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)') 
    for word in v_in:
        # replace the bigram (with space), with the new merged token (the concanated pair)
        w_out = p.sub(new_token, word)
        v_out[w_out] = v_in[word]
    return v_out
  
def get_init_vocab(data): 
    """ 
    Given a list of strings, returns a dictionary of words mapping to their frequency  
    count in the data. 
    """
    vocab = defaultdict(int)
    tokens = set()
    tokens.add('</w>')
    for line in data: 
        for word in line.split(): 
            vocab[' '.join(list(word)) + ' </w>'] += 1
            tokens.update(list(word))
    return vocab, tokens 
  
def byte_pair_encoding(data, n): 
    """ 
    Given a list of strings and an integer n, returns a list of n merged pairs 
    of characters found in the vocabulary of the input data. 
    """
    vocab, init_tokens = get_init_vocab(data)
    tokens = list(init_tokens)
    for i in range(n): 
        pairs = get_stats(vocab) 
        best_pair = max(pairs, key=pairs.get) 
        vocab = merge_vocab(best_pair, vocab)
        tokens.append(''.join(best_pair))
        # print('step {}: merging \"{}\" and \"{}\"'.format(i+1, best_pair[0], best_pair[1]))
    return tokens

def tokenize(data, token_dict):
    """split the data into a tokens and map into index"""
    encoded_ids = []
    for line in data: 
        for word in line.split():
            word = word + '</w>'
            last_idx = 0
            idx = len(word)
            while idx > last_idx:
                whole_word = word[last_idx:idx]
                if whole_word in token_dict:
                    encoded_ids.append(token_dict[whole_word])
                    last_idx = idx
                    idx = len(word)
                else:
                    idx = idx - 1
    return encoded_ids


def _read_text_file(input_path: str, num_workers: int, special_tokens: Iterable[str]):
    with open(input_path, 'r') as f:
        text = f.read()

    ## Remove special tokens from the text
    for token in special_tokens:
        text = text.replace(token, "")
    return text

 

GPT2_PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
import time

path = "./demo.txt"
special_tokens = ["<|endoftext|>"] 

start_time = time.time()
corpus = _read_text_file(path, 2, special_tokens)
print(time.time()-start_time) 

data = corpus.split('.') 

start_time = time.time()
n = 100 # number of merge operations
bpe_vocab = byte_pair_encoding(data, n) 

print("Time taken: " , time.time() - start_time)

bpe_dict = dict([(tk, id) for id, tk in enumerate(bpe_vocab)])
id_to_token = dict([(tid, tk) for tk, tid in bpe_dict.items()]  )

token_ids = tokenize(data, bpe_dict)

# print("The bpe tokens are: ")
# for tk, tid in bpe_dict.items():
#     print("{}: {}".format(tk, tid))

# print(token_ids)
# print(' '.join(id_to_token[tid] for tid in token_ids))