from cs336_basics.tokenizer import Tokenizer
from typing import Iterable
import time

ENDOFTEXT = '<|endoftext|>'
tinystory = {
    'textsource':'data/raw/TinyStoriesV2-GPT4-train.txt',
    'vocab_path': 'data/out/tinystories_vocab.json',
    'merge_path': 'data/out/tinystories_merges.txt'
}
owt = {
    'textsource':'data/raw/owt_train_2G.txt',
    'vocab_path': 'data/out/owt_vocab.json',
    'merge_path': 'data/out/owt_merges.txt'
}


def sample_text(textsource: str):
    with open(textsource) as f:
        text = f.read()
    text = text.split(ENDOFTEXT)
    return text[:10]

def estimate_compression_ratio_and_throughput(tokenizer: Tokenizer, text: Iterable[str]=None):
    start = time.time()
    encoded = [tokenizer.encode(t) for t in text]
    end = time.time()
    encoded_length = sum([len(e) for e in encoded])
    bytes_length = sum([len(t.encode('utf-8')) for t in text])
    return bytes_length / encoded_length, bytes_length / (end - start)


if __name__ == "__main__":
    # stats for tiny stories
    print('-'*50)
    print('Sample text from TinyStory:')
    tinystory_sample = sample_text(tinystory['textsource'])
    print(tinystory_sample[0][:50], '...')
    tiny_tokenizer = Tokenizer.from_files(tinystory['vocab_path'], tinystory['merge_path'])
    tiny_ratio, tiny_throughput = estimate_compression_ratio_and_throughput(tiny_tokenizer, tinystory_sample)
    print(f'Estimated compression ratio for TinyStory: {tiny_ratio}')
    print(f'Estimated throughput for TinyStory: {tiny_throughput} bytes/sec')
    
    # stats for owt
    print('-'*50)
    print('Sample text from OpenWebText:')
    owt_sample = sample_text(owt['textsource'])
    print(owt_sample[0][:50], '...')
    owt_tokenizer = Tokenizer.from_files(owt['vocab_path'], owt['merge_path'])
    owt_ratio, owt_throughput = estimate_compression_ratio_and_throughput(owt_tokenizer, owt_sample)
    print(f'Estimated compression ratio for OpenWebText: {owt_ratio}')
    print(f'Estimated throughput for OpenWebText: {owt_throughput} bytes/sec')

    # what if using tinystory tokenizer on owt
    print('-'*50)
    owt_compression_ratio_tiny, _ = estimate_compression_ratio_and_throughput(tiny_tokenizer, owt_sample)
    print(f'Estimated compression ratio for OpenWebText using TinyStory tokenizer: {owt_compression_ratio_tiny}')