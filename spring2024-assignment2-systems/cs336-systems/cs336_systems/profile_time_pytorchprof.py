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

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

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
model.to(device)
print("Model Loaded...")
print(model)


## Generate random batch of data
input_data = np.random.randint(1, config.vocab_size, (config.batch_size, config.context_length))
input_data = torch.tensor(input_data).to(device)

# Create target data shifted by 1 position
target_data = torch.roll(input_data, -1, dims=1)
target_data[:, -1] = 1  # padding for last position
target_data = target_data.to(device)


optimizer = AdamW(model.parameters())

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

# Verify CUDA is being used
print(f"Input data device: {input_data.device}")
print(f"Model device: {next(model.parameters()).device}")



with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    record_shapes=True,
    profile_memory=False,
    with_stack=True
) as prof:
    for _ in range(config.num_steps):
        with record_function("forward_pass"):
            loss = forward_pass()
            torch.cuda.synchronize()
            prof.step()
        if config.enable_backward:
            with record_function("backward_pass"):
                backward_pass()
                torch.cuda.synchronize()
                prof.step()
            with record_function("optimizer"):
                clip_gradient(model.parameters(), 1.0)
                optimizer.step()
                torch.cuda.synchronize()
                prof.step()


# Print profiling results
print("\nGPU Time Summary:")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("\nCPU Time Summary:")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

prof.export_stacks(f"out/lm_profiler_stacks.txt", "self_cuda_time_total")
print(prof.key_averages().table(sort_by="cpu_time_total"))  