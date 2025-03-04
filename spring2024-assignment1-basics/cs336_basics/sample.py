import torch

import numpy as np
from transformers import HfArgumentParser
from cs336_basics.model import TransformerLM, TransformerLMAblation
from cs336_basics.tokenizer import Tokenizer
# from cs336_basics.train import TrainingConfig

from dataclasses import dataclass, field, asdict
from cs336_basics.utils.io_fun import save_checkpoint, load_checkpoint

import logging
print("Hello World")    


@dataclass
class InferenceConfig:
    # Model parameters only
    vocab_size: int = field(default=50257)
    context_length: int = field(default=1024)
    num_layers: int = field(default=12)
    d_model: int = field(default=768)
    num_heads: int = field(default=12)
    d_ff: int = field(default=3072)
    attn_pdrop: float = field(default=0.1)
    resid_pdrop: float = field(default=0.1)
    device: str = field(default='cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size: int = field(default=1)
    wandb_logging: bool = field(default=False)

def softmax(logits, temperature=1.0):
    """Apply temperature-scaled softmax to logits."""
    logits = logits / temperature
    exp_logits = np.exp(logits - np.max(logits))  # For numerical stability
    return exp_logits / np.sum(exp_logits)


# def top_p_sampling(probabilities, p):
#     """Perform top-p (nucleus) sampling."""
#     # Sort probabilities in descending order
#     sorted_indices = np.argsort(probabilities)[::-1]
#     sorted_probs = probabilities[sorted_indices]
#     cumulative_probs = np.cumsum(sorted_probs)
    
#     # Find indices where cumulative prob >= p
#     cutoff_idx = np.where(cumulative_probs >= p)[0][0] + 1
    
#     # Create mask array of original size
#     mask = np.zeros_like(probabilities)
#     mask[sorted_indices[:cutoff_idx]] = sorted_probs[:cutoff_idx]
#     return mask


# def decode(model, prompt, max_tokens=50, temperature=1.0, top_p=0.9):
#     """Decode text using model with temperature and top-p sampling."""
#     model.eval() 
#     device = next(model.parameters()).device
    
#     # Convert prompt to tensor
#     if isinstance(prompt, list):
#         input_ids = torch.tensor(prompt).unsqueeze(0)
#     else:
#         input_ids = prompt.unsqueeze(0)
    
#     input_ids = input_ids.to(device)
#     generated = input_ids
    
#     with torch.no_grad():
#         for _ in range(max_tokens):
#             # Get model output
#             outputs = model(generated)
#             next_token_logits = outputs[0, -1, :]
            
#             # Apply temperature and get probabilities
#             probs = softmax(next_token_logits.cpu().numpy(), temperature)
            
#             # Apply top-p sampling and normalize
#             filtered_probs = top_p_sampling(probs, top_p)
#             filtered_probs = filtered_probs / filtered_probs.sum()
            
#             # Sample from probability distribution
#             idx = np.arange(len(probs))
#             # print(idx)

#             next_token = int(np.random.choice(idx, p=filtered_probs))
#             next_token_tensor = torch.tensor([[next_token]]).to(device)
            
#             # Append to sequence
#             generated = torch.cat([generated, next_token_tensor], dim=1)
            
#             # Check for EOS
#             if next_token == 256:
#                 break
    
#     return generated[0].cpu().tolist()

def top_p_sampling(probabilities, p):
    """Perform top-p (nucleus) sampling."""
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff_idx = np.where(cumulative_probs >= p)[0][0] + 1
    
    # Zero out low probability tokens
    filtered_probs = np.zeros_like(probabilities)
    filtered_probs[sorted_indices[:cutoff_idx]] = sorted_probs[:cutoff_idx]
    return filtered_probs

def decode(model, prompt, max_tokens=50, temperature=1.0, top_p=0.9):
    """Decode text using model with temperature and top-p sampling."""
    model.eval()
    device = next(model.parameters()).device
    
    # Convert prompt to tensor
    if isinstance(prompt, list):
        input_ids = torch.tensor(prompt).unsqueeze(0)
    else:
        input_ids = prompt.unsqueeze(0)
    
    input_ids = input_ids.to(device)
    generated = input_ids
    
    # Get EOS token from tokenizer
    eos_token = 256
    
    with torch.no_grad():
        for _ in range(max_tokens):
            if generated.size(1) >= 256:
                break
                
            # Get model output
            try:
                outputs = model(generated)
                next_token_logits = outputs[0, -1, :]
            except RuntimeError as e:
                print(f"Error in model forward pass: {e}")
                break
            
            # Apply temperature and get probabilities
            probs = softmax(next_token_logits.cpu().numpy(), temperature)
            
            # Apply top-p sampling
            filtered_probs = top_p_sampling(probs, top_p)
            if filtered_probs.sum() == 0:
                print("Warning: All probabilities were filtered out")
                filtered_probs = probs
            
            # Normalize probabilities
            filtered_probs = filtered_probs / filtered_probs.sum()
            
            # Sample next token
            try:
                next_token = int(np.random.choice(len(filtered_probs), p=filtered_probs))
                next_token_tensor = torch.tensor([[next_token]]).to(device)
            except ValueError as e:
                print(f"Sampling error: {e}")
                break
            
            # Append to sequence
            generated = torch.cat([generated, next_token_tensor], dim=1)
            
            # Check for EOS
            if next_token == eos_token:
                break
    
    return generated[0].cpu().tolist()



checkpoint = "./data/out/checkpoints/tinystories_lr_0.0005.pt"
print(f"Loading model from {checkpoint}")


parser = HfArgumentParser(InferenceConfig)
config = parser.parse_args_into_dataclasses()[0]

if config.wandb_logging:
    import wandb
    wandb.init(project=config.wandb_project, name=config.wandb_run_name)

logging.info(f'Training with config: {asdict(config)}')


print("Loading Model...")
# Write a code to initialize the model
model = TransformerLM(**asdict(config))
print("Model Loaded...")
print(model)

print("Loading Checkpoint...")
model.load_state_dict(torch.load(checkpoint)['model_state_dict'])
print("Checkpoint Loaded...")

model.eval()


tinystory = {
    'train':'data/TinyStoriesV2-GPT4-train.txt',
    'val':'data/TinyStoriesV2-GPT4-valid.txt',
    'vocab_filepath': 'data/out/tinystories_vocab.json',
    'merges_filepath': 'data/out/tinystories_merges.txt',
    'special_tokens': ['<|endoftext|>']
}

tokenizer = Tokenizer.from_files(**tinystory)

input_text = "Once upon a time, there was a girl named Amy. "

print("EOS: ", tokenizer.encode("<|endoftext|>"))


prompt = tokenizer.encode(input_text)
print(prompt)
print("Len of prompt: ", len(prompt))

print("Generating text...") 
decoded_tokens = decode(model=model, prompt=prompt, max_tokens=200, temperature=0.8, top_p=0.95)
print(decoded_tokens)
print("Len: ", len(decoded_tokens))

decoded_text = tokenizer.decode(decoded_tokens)
print(decoded_text)
