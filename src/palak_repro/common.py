import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
PAPER_ASR = {("uce", "none"): 10.0, ("uce", "shallow"): 92.0, ("uce", "deep"): 94.2,
             ("esd", "none"): 10.0, ("esd", "shallow"): 38.0, ("esd", "deep"): 70.8}


def read_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg["targets"] or any(t not in CLASSES for t in cfg["targets"]):
        raise ValueError("targets must be CIFAR-10 class names")
    if cfg["trigger"] in CLASSES:
        raise ValueError("trigger must differ from a class name")
    return cfg


def device_for(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def done(path):
    return Path(path).is_file()
