"""Load the prepared k-space stored in each case's ``kspace.npz``.

The file holds ``mask``, ``samples``, ``coils``, ``data_shape`` and, optionally,
``negative_zero``. The loader expands them into ``Y``, the complex k-space of
shape (coils, H, W, T) that is zero where not acquired, and ``samp_4d_idx``,
the acquired indices of shape (4, samples) with axes (H, W, coil, T).

``mask`` is a boolean (H, W, T) array of acquired locations, identical for
every coil. ``samples`` holds the acquired complex values in the order of
``samp_4d_idx`` (row-major over H, W, coil, T), which is also the measurement
vector consumed by the MRI operator. The optional ``negative_zero`` array
(packed bits over the real/imaginary components of ``Y``) records zeros that
were stored as -0.0; they come from multiplying k-space by the sampling mask
and are never read by reconstruction, but keeping them makes the expansion
reproduce the original arrays byte for byte, including dtypes.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np

COMPACT_FILENAME = "kspace.npz"


def expand_kspace_arrays(
    arrays,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild ``(Y, samp_4d_idx, data_shape)`` from compact arrays."""
    mask = np.asarray(arrays["mask"], dtype=bool)
    samples = np.asarray(arrays["samples"])
    coils = int(arrays["coils"])
    data_shape = np.asarray(arrays["data_shape"])
    if mask.ndim != 3 or coils <= 0:
        raise ValueError(f"invalid compact k-space mask {mask.shape} or coils {coils}")
    full_mask = np.broadcast_to(mask[:, :, None, :], mask.shape[:2] + (coils, mask.shape[2]))
    samp_4d_idx = np.stack(np.nonzero(full_mask)).astype(np.int64)
    if samples.shape != (samp_4d_idx.shape[1],):
        raise ValueError(
            f"compact k-space has {samples.shape} samples for {samp_4d_idx.shape[1]} locations"
        )
    Y = np.zeros((coils, *mask.shape), dtype=samples.dtype)
    Y[samp_4d_idx[2], samp_4d_idx[0], samp_4d_idx[1], samp_4d_idx[3]] = samples
    if "negative_zero" in arrays:
        components = Y.view(Y.real.dtype).reshape(-1)
        bits = np.unpackbits(np.asarray(arrays["negative_zero"]), count=components.size)
        components[bits.astype(bool)] = -0.0
    return Y, samp_4d_idx, data_shape


def load_prepared_kspace(
    folder: str | os.PathLike[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(Y, samp_4d_idx, data_shape)`` from a prepared case directory."""
    with np.load(os.path.join(folder, COMPACT_FILENAME), allow_pickle=False) as arrays:
        return expand_kspace_arrays(arrays)


def load_data_shape(folder: str | os.PathLike[str]) -> np.ndarray:
    """Return the image shape without expanding the k-space."""
    with np.load(os.path.join(folder, COMPACT_FILENAME), allow_pickle=False) as arrays:
        return np.asarray(arrays["data_shape"])
