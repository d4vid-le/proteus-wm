"""Device selection with an explicit override.

Some virtualized macOS runners advertise MPS but cannot allocate even small
tensors. Set ``PROTEUS_DEVICE=cpu`` to force a device; otherwise prefer MPS,
then CUDA, then CPU.
"""

import os

import torch


def get_device() -> torch.device:
    forced = os.environ.get("PROTEUS_DEVICE", "").strip()
    if forced:
        return torch.device(forced)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = str(get_device())
