import torch

def cross_entropy(inputs, targets):
    """
    inputs: (batch_size, num_classes)
    targets: (batch_size)
    """
    log_softmax = inputs - torch.logsumexp(inputs, dim=-1, keepdim=True)
    log_probs = torch.gather(log_softmax, dim=-1, index=targets.unsqueeze(-1))
    loss = -log_probs.mean()

    return loss

