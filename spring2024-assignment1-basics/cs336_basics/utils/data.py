## Implement Data Loader for Training

import torch
import numpy as np
from typing import Tuple

def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str)->Tuple[torch.Tensor, torch.Tensor]:   
    """
    inputs:
        x: numpy.array of integers with token IDs
        batch_size: int
        context_length: int
        device: str (eg. cpu or cuda:0)
    return: 
        sampled_input: torch.tensor of shape (batch_size, context_length)
        target: torch.tensor of shape (batch_size, context_length)
    """
    starting_index = torch.randint(0, len(dataset) - context_length, (batch_size,))
    x = torch.stack([torch.from_numpy(dataset[start_idx: start_idx + context_length]) for start_idx in starting_index])
    y = torch.stack([torch.from_numpy(dataset[start_idx + 1: start_idx + context_length + 1]) for start_idx in starting_index])

    return x.to(device), y.to(device)   

class Dataset:
    def __init__(self, dataset_name:str, context_length:int, batch_size:int, device:str, **kwargs):
        dataset_path = f'data/{dataset_name}'
        self.train_data = np.memmap(f'{dataset_path}/train.bin', dtype=np.uint16, mode='r').astype(np.int64)
        self.val_data = np.memmap(f'{dataset_path}/val.bin', dtype=np.uint16, mode='r').astype(np.int64)

        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device
    def get_batch(self, split:str):
        data = self.train_data if split == 'train' else self.val_data
        return get_batch(data, self.batch_size, self.context_length, self.device)