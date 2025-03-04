import torch

import numpy as np
from transformers import HfArgumentParser
from cs336_basics.model import BasicsTransformerLM

# from cs336_basics.train import TrainingConfig
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import clip_gradient, cross_entropy


from dataclasses import dataclass, field, asdict
from torch.profiler import profile, record_function, ProfilerActivity

import logging
import timeit

device: str = field(default='cuda:0' if torch.cuda.is_available() else 'cpu')

@dataclass
class BenchmarkingConfig:
    # Model parameters only
    vocab_size: int = field(default=10000)
    context_length: int = field(default=1024)
    num_layers: int = field(default=12)
    d_model: int = field(default=768)
    num_heads: int = field(default=12)
    d_ff: int = field(default=3072)
    attn_pdrop: float = field(default=0.1)
    residual_pdrop: float = field(default=0.1)
    batch_size: int = field(default=16)
    wandb_logging: bool = field(default=False)

    ## Profiling parameters
    enable_backward: bool = field(default=False)
    warmup_steps: int = field(default=0)
    num_steps: int = field(default=10)
    profile_memory: bool = field(default=False)
    

parser = HfArgumentParser(BenchmarkingConfig)
config = parser.parse_args_into_dataclasses()[0]

model = BasicsTransformerLM(**asdict(config))
print("Model Loaded...")
print(model)


## Generate random batch of data
input_data = np.random.randint(1, config.vocab_size, (config.batch_size, config.context_length))
input_data = torch.tensor(input_data)

# Create target data shifted by 1 position
target_data = torch.roll(input_data, -1, dims=1)
target_data[:, -1] = 1  # padding for last position


# Initialize loss function once outside the loop
loss = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters())
#
print("Enable backward: ", config.enable_backward)

def forward_pass():
    logits = model(input_data)
    loss = cross_entropy(logits, target_data)
    return loss

def backward_pass():
    optimizer.zero_grad()
    loss.backward()

for _ in range(config.warmup_steps):
    loss = forward_pass()
    backward_pass()
    clip_gradient(model.parameters(), 1.0)
    optimizer.step()

torch.cuda.synchronize()

## Profile time without using pytorch profiler
profile_time = 0
times = []
for i in range(config.num_steps):
    start_time = timeit.default_timer()
    loss = forward_pass()
    torch.cuda.synchronize()
    if config.enable_backward:
        print("Got inside back")
        backward_pass()
        torch.cuda.synchronize()
    end_time = timeit.default_timer()
    times.append(end_time - start_time)
    profile_time += (end_time - start_time)

## print average and std dev
average_time = profile_time/config.num_steps
std_dev = np.std(times)
print(f"Average time taken: {average_time:.4f} seconds")  # Add seconds unit and format to 4 decimal places
print("Standard Deviation : ", std_dev)



