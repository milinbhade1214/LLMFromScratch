from dataclasses import dataclass, field, asdict
from typing import Optional, Iterable
from transformers import HfArgumentParser
import torch
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import time



from cs336_basics.utils.data import Dataset
from cs336_basics.utils.nn import cross_entropy
from cs336_basics.utils.io_fun import save_checkpoint, load_checkpoint
from cs336_basics.model import TransformerLM, TransformerLMAblation
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping


@dataclass
class TrainingConfig:
    # dataset parameters
    dataset_name: str 
    context_length: int
    batch_size: int
    device: Optional[str] = field(default='cuda:0' if torch.cuda.is_available() else 'cpu')
    # model parameters
    vocab_size: Optional[int] = field(default=50257)
    context_size: Optional[int] = field(default=1024)
    num_layers: Optional[int] = field(default=12)
    d_model: Optional[int] = field(default=768)
    num_heads: Optional[int] = field(default=12)
    d_ff: Optional[int] = field(default=3072)
    attn_pdrop: Optional[float] = field(default=0.1)
    resid_pdrop: Optional[float] = field(default=0.1)
    init_from: str = field(default='scratch')

    # Training Parameters (additional adamw params use as default)
    total_iters: Optional[int] = field(default=10*(10**3))
    warmup_iters: Optional[int] = field(default=None)
    lr_max: Optional[float] = field(default=5e-4)
    lr_min: Optional[float] = field(default=0)
    weight_decay: Optional[float] = field(default=0.001)

    # logging params
    wandb_logging: Optional[bool] = field(default=False)
    wandb_project: Optional[str] = field(default=None)
    wandb_run_name: Optional[str] = field(default=None)
    log_interval: Optional[int] = field(default=None)
    eval_interval: Optional[int] = field(default=None)
    eval_iters: Optional[int] = field(default=100)

    # ablation studies
    no_rmsnorm: Optional[bool] = field(default=False)
    parallel_layers: Optional[bool] = field(default=False)
    post_norm: Optional[bool] = field(default=False)

    def __post_init__(self):
        if self.warmup_iters is None:
            self.warmup_iters = int(self.total_iters * 0.01)
        if self.log_interval is None:
            self.log_interval = int(self.total_iters * 0.001)
        if self.eval_interval is None:
            self.eval_interval = int(self.total_iters * 0.01)
        if self.wandb_logging:
            assert self.wandb_project is not None, 'wandb_project must be provided if wandb_logging is True'
            assert self.wandb_run_name is not None, 'wandb_run_name must be provided if wandb_logging is True'
        self.ablation = self.no_rmsnorm or self.parallel_layers or self.post_norm

        
# parsing config
parser = HfArgumentParser(TrainingConfig)
config = parser.parse_args_into_dataclasses()[0]

if config.wandb_logging:
    import wandb
    wandb.init(project=config.wandb_project, name=config.wandb_run_name)

logging.info(f'Training with config: {asdict(config)}')

# loading the dataset
dataset = Dataset(**asdict(config))
# loading the model
if config.ablation:
    model = TransformerLMAblation(**asdict(config))
else:
    model = TransformerLM(**asdict(config))

model.to(config.device)


# loading the optimizer
optimizer = AdamW(model.parameters(), **asdict(config))


if config.init_from != 'scratch':
    ckpt_dir = f'data/out/checkpoints/{config.init_from}'
    iter_num = load_checkpoint(model, optimizer, ckpt_dir)




def eval():
    total_loss = 0
    for _ in range(config.eval_iters):
        x, y = dataset.get_batch('val')
        x, y = x.to(config.device), y.to(config.device)
        with torch.no_grad():
            logits = model(x)
            loss = cross_entropy(logits, y)
            total_loss += loss.item()
    total_loss /= config.eval_iters
    logging.info(f'Iter: {iter_num}, Val loss: {loss.item():.4f}, LR: {lr:.6f}')
    if config.wandb_logging:
        wandb.log({'val_loss': total_loss, 'lr': lr, 'iter': iter_num})
        save_checkpoint(model, optimizer, iter_num, f'data/out/checkpoints/{config.wandb_run_name}.pt')


iter_num = 0
curr_time = time.time()
while iter_num < config.total_iters:
    optimizer.zero_grad()

    # core backward pass
    x, y = dataset.get_batch('train')
    logits = model(x)
    loss = cross_entropy(logits, y)
    loss.backward()
    gradient_clipping(model.parameters(), 1.0)
    lr = get_lr_cosine_schedule(iter_num, **asdict(config))
    optimizer.set_lr(lr)
    optimizer.step()
    finish_time = time.time()

    # logging
    if iter_num % config.log_interval == 0:
        logging.info(f'Iter: {iter_num}, Train loss: {loss.item():.4f}, LR: {lr:.6f}, Time: {1000*(finish_time - curr_time):.2f}ms')
    # evaluation
    if iter_num % config.eval_interval == 0:
        eval()
    
    curr_time = finish_time
    iter_num += 1