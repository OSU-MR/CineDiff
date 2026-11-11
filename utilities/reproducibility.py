"""Utilities for reproducible reconstruction runs."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


MAX_NUMPY_SEED = 2**32 - 1


def validate_seed(seed: Any) -> int:
    """Return *seed* as an integer accepted by NumPy and PyTorch."""
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer, not a boolean")

    try:
        normalized = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"seed must be an integer, got {seed!r}") from exc

    if normalized < 0 or normalized > MAX_NUMPY_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_NUMPY_SEED}")
    return normalized


def seed_everything(seed: int) -> int:
    """Seed Python, NumPy, and PyTorch without changing backend settings.

    ``PYTHONHASHSEED`` only takes full effect when it is present before Python
    starts. Reconstruction launchers and the reproducibility harness therefore
    also propagate it to child processes.
    """
    normalized = validate_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(normalized)
    random.seed(normalized)
    np.random.seed(normalized)

    import torch

    torch.manual_seed(normalized)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized)

    return normalized


def make_torch_generator(seed: int, device: Any):
    """Create an RNG dedicated to sampler noise on ``device``."""
    import torch

    generator = torch.Generator(device=device)
    generator.manual_seed(validate_seed(seed))
    return generator


def randn_like(tensor, *, generator):
    """Generate normal noise like ``tensor`` from a dedicated generator."""
    import torch

    return torch.randn(
        tensor.shape,
        dtype=tensor.dtype,
        layout=tensor.layout,
        device=tensor.device,
        generator=generator,
    )
