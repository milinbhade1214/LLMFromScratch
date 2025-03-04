from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import numpy as np


def get_lr_cosine_schedule(t, lr_max, lr_min, warmup_iters, total_iters, **kwargs):
    if t < warmup_iters:
        return lr_max * t / warmup_iters
    elif t < total_iters:
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos((t - warmup_iters) / (total_iters - warmup_iters) * 3.141592653589793))
    else:
        return lr_min
        
def gradient_clipping(parameters, max_norm, eps=1e-6,**kwargs):
    total_norm_2 = sum([torch.sum(p.grad ** 2) for p in parameters])
    total_norm = total_norm_2 ** 0.5
    if total_norm    > max_norm:
        for p in parameters:
            p.grad.detach().mul_(max_norm / (total_norm + eps))
    


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.95, lambda_=1e-4, eps=1e-8, **kwargs):
        if lr < 0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = {"lr":lr, "beta1":beta1, "beta2":beta2, "lambda_":lambda_, "eps":eps}
        super().__init__(params, defaults)


    def set_lr(self, lr):
        for group in self.param_groups:
            group['lr'] = lr
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            weight_decay = group["lambda_"] # Get the weight decay.
            beta1 = group["beta1"] # Get the beta1 parameter.
            beta2 = group["beta2"] # Get the beta2 parameter.
            eps = group["eps"] # Get the epsilon parameter.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p.grad.data)
                    state["v"] = torch.zeros_like(p.grad.data)
                    state["t"] = 0
                
                t = state["t"]  + 1
                m = state["m"]
                v = state["v"]

                # Update biased first and second moment estimates.
                m = beta1 * m + (1 - beta1) * p.grad.data
                v = beta2 * v + (1 - beta2) * p.grad.data ** 2

                grad = p.grad.data # Get the gradient of loss with respect to p.
                
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t) 

                # Update the parameters.
                p.data = p.data - lr_t * m / (torch.sqrt(v) + eps)
                p.data = p.data - weight_decay * lr * p.data 

                # Update the state dictionary.
                state["m"] = m
                state["v"] = v
                state["t"] = t 
        return loss





# for lr in [1e-1, 1e-2, 1e-3]:
#     weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
#     opt = AdamW([weights], lr=lr)
#     print(f"***********************************Learning rate: {lr}*****************************************")
#     for i in range(100):
#         opt.zero_grad()
#         loss = (weights ** 2).sum()
#         print(f"Step {i}, loss {loss.item()}")
#         loss.backward()
#         opt.step()
        


# import matplotlib.pyplot as plt
# # Parameters
# max_lr = 0.5
# min_lr = 0.01
# tw = 10
# tc = 1000
# total_steps = 2000

# # Generate learning rates
# lrs = [get_lr_cosine_schedule(t, max_lr, min_lr, tw, tc) for t in range(total_steps)]

# # Plot the learning rate schedule
# plt.plot(range(total_steps), lrs)
# plt.xlabel('Training Steps')
# plt.ylabel('Learning Rate')
# plt.title('Cosine Learning Rate Schedule')
# plt.grid(True)
# plt.show()
# plt.savefig("cosine_lr_schedule.png")