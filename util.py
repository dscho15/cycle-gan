import torch
import random, torch, os, numpy as np


# use moving average
def moving_average(x: torch.Tensor | list, n=100):

    if isinstance(x, list):
        x = torch.tensor(x)

    if x.size(0) > n:
        return x[-n:].mean().item()

    return x.mean().item()


def seed_env(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
