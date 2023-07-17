"""
Misc. functions useful during model training
"""
import random
import numpy as np
import torch


def set_seed(seed):
    """Manually sets the seed for python internals, numpy, torch and CUDA

    Args:
        seed: SEED
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
