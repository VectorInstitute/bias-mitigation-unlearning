import numpy as np
import torch
import random
import json

def set_random_seed(seed): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_jsonl(file_name):
    """
    Reads objects from jsonl file to list.
    """
    content = []
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            content.append(json.loads(line.strip("\n|\r")))
    return content
