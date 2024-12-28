import torch
import numpy as np
import random

def remove_duplicates(lst):
    seen = set()
    return [x for x in lst if x not in seen and not seen.add(x)]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

