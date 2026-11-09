import platform
import random

import av
import numpy as np
import torch
import torchvision
import transformers
import yaml


def load_config(path, overrides=()):
    """YAML config + `key=value` overrides (values parsed as YAML: k=1, epochs=2, ...)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, value = item.split("=", 1)
        if key not in cfg:
            raise KeyError(f"Unknown config key: {key}")
        cfg[key] = yaml.safe_load(value)
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and the DataLoader worker seeds


def env_info():
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "av": av.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
    }
