### Configuration: deep_update, set_cpu_safety
from copy import deepcopy


def deep_update(base: dict, updates: dict) -> dict:
    """Recursively update dicts (for cfg overrides). Does not mutate inputs."""
    out = deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k, None), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def set_cpu_safety(num_threads: int = 1):
    """Set torch thread limits to avoid CPU oversubscription / kernel death in notebooks."""
    import torch
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
