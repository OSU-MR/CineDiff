"""Layout helpers for MATLAB ESPIRiT soft-SENSE arrays."""

from __future__ import annotations

import numpy as np


def resolve_map_set_count(source_count: int, requested_count: int | None) -> int:
    """Validate an override and return the number of final map sets to use."""
    source_count = int(source_count)
    if source_count < 1:
        raise ValueError("source sensitivity-map count must be positive")
    if requested_count is None:
        return source_count
    if (
        not isinstance(requested_count, int)
        or isinstance(requested_count, bool)
        or requested_count < 1
    ):
        raise ValueError("num_sense_maps must be a positive integer or None")
    if requested_count > source_count:
        raise ValueError(
            f"requested {requested_count} sensitivity-map sets, but the source "
            f"contains {source_count}"
        )
    return requested_count


def matlab_sense_maps_to_map_coil(sense_maps: np.ndarray) -> np.ndarray:
    """Convert sensitivity maps to ``(map_sets, coils, kx, ky)``."""
    maps = np.asarray(sense_maps)
    if maps.ndim in (4, 5) and maps.shape[2] == 1:
        maps = maps[:, :, 0, ...]
    if maps.ndim == 3:
        kx, ky, coils = maps.shape
        if min(kx, ky, coils) < 1:
            raise ValueError(f"invalid sensitivity-map shape {maps.shape}")
        return np.ascontiguousarray(maps.transpose(2, 0, 1)[None, ...])
    if maps.ndim == 4:
        kx, ky, coils, map_sets = maps.shape
        if min(kx, ky, coils, map_sets) < 1:
            raise ValueError(f"invalid sensitivity-map shape {maps.shape}")
        return np.ascontiguousarray(maps.transpose(3, 2, 0, 1))
    raise ValueError(
        "MATLAB sensitivity maps must reduce to (kx, ky, coils) or "
        f"(kx, ky, coils, map_sets), got {maps.shape}"
    )


def matlab_reconstruction_to_map_series(
    image_data: np.ndarray,
    *,
    map_set_count: int,
) -> np.ndarray:
    """Convert a MATLAB reconstruction to ``(map_sets, kx, ky, frames)``."""
    series = np.asarray(image_data)
    if series.ndim in (5, 6) and series.shape[2:4] == (1, 1):
        series = series[:, :, 0, 0, ...]
    if series.ndim == 3:
        components = series[None, ...]
    elif series.ndim == 4:
        components = series.transpose(3, 0, 1, 2)
    else:
        raise ValueError(
            "MATLAB reconstruction must reduce to (kx, ky, frames) or "
            f"(kx, ky, frames, map_sets), got {series.shape}"
        )
    if components.shape[0] != int(map_set_count):
        raise ValueError(
            "reconstruction map-set count does not match sensitivity maps: "
            f"expected {map_set_count}, got {components.shape[0]}"
        )
    return np.ascontiguousarray(components)
