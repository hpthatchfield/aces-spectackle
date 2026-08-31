### Configuration: deep_update, set_cpu_safety, set_seed
from copy import deepcopy


def set_seed(seed: int = 42):
    """Fix RNG for reproducible runs."""
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deep_update(base: dict, updates: dict) -> dict:
    """Recursive dict merge. Doesn't mutate inputs."""
    out = deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k, None), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def set_cpu_safety(num_threads: int = 1):
    """Limit torch threads (avoids CPU oversubscription crashes)."""
    import torch
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)


def get_device() -> str:
    """cuda > mps > cpu."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
