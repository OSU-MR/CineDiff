"""Overlapping 3-D patch extraction and weighted reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from numbers import Integral
from typing import Sequence

import torch
from torch import Tensor


TEMPORAL_BOUNDARY_MODES = ("legacy_pad", "balanced_full")


def balanced_temporal_weights(length: int, *, device=None) -> Tensor:
    """Return uniform temporal weights for one balanced window."""
    if isinstance(length, bool) or not isinstance(length, Integral) or length <= 0:
        raise ValueError("temporal weight length must be a positive integer")
    return torch.ones(int(length), dtype=torch.float32, device=device)


@dataclass(frozen=True)
class TemporalWindowPlan:
    """Temporal window coordinates shared by extraction and aggregation."""

    frame_count: int
    patch_length: int
    starts: tuple[int, ...]
    valid_lengths: tuple[int, ...]

    @property
    def patch_count(self) -> int:
        return len(self.starts)


def plan_temporal_windows(
    frame_count: int,
    patch_length: int,
    stride: int,
    *,
    temporal_boundary_mode: str = "balanced_full",
) -> TemporalWindowPlan:
    """Plan padded fixed-stride or evenly distributed full temporal windows."""
    values = {
        "frame_count": frame_count,
        "patch_length": patch_length,
        "stride": stride,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    frame_count = int(frame_count)
    patch_length = int(patch_length)
    stride = int(stride)
    if temporal_boundary_mode not in TEMPORAL_BOUNDARY_MODES:
        raise ValueError(
            "temporal_boundary_mode must be 'legacy_pad' or 'balanced_full', "
            f"got {temporal_boundary_mode!r}"
        )

    patch_count = max(1, ceil((frame_count - patch_length) / stride) + 1)
    if temporal_boundary_mode == "legacy_pad":
        starts = tuple(index * stride for index in range(patch_count))
        valid_lengths = tuple(
            max(0, min(patch_length, frame_count - start)) for start in starts
        )
        return TemporalWindowPlan(
            frame_count,
            patch_length,
            starts,
            valid_lengths,
        )

    if frame_count < patch_length:
        raise ValueError(
            "balanced_full temporal windows require frame_count to be at least "
            f"patch_length, got {frame_count} < {patch_length}"
        )
    if patch_count == 1:
        starts = (0,)
    else:
        span = frame_count - patch_length
        intervals = patch_count - 1
        starts = tuple(
            (index * span + intervals // 2) // intervals
            for index in range(patch_count)
        )
    if any(right <= left for left, right in zip(starts, starts[1:])):
        raise ValueError(
            "balanced_full could not create distinct temporal windows; reduce "
            "the requested temporal overlap"
        )
    return TemporalWindowPlan(
        frame_count,
        patch_length,
        starts,
        (patch_length,) * patch_count,
    )


def compute_stride_and_n_patches(
    data_length: int,
    patch_length: int,
) -> tuple[int, int]:
    """Cover an axis while requiring overlap between adjacent patches."""
    if data_length < 1 or patch_length < 1:
        raise ValueError("data and patch lengths must be positive")

    patch_count = 1
    while True:
        stride = 1 if patch_count == 1 else max(
            1,
            ceil((data_length - patch_length) / (patch_count - 1)),
        )
        starts = (index * stride for index in range(patch_count))
        covered = set()
        for start in starts:
            covered.update(range(start, min(start + patch_length, data_length)))
        has_overlap = patch_count == 1 or patch_length == 1 or stride < patch_length
        if len(covered) >= data_length and has_overlap:
            return stride, patch_count
        patch_count += 1


def compute_strides_and_N(
    data_shape: Sequence[int],
    patch_size: Sequence[int] = (64, 64, 8),
) -> tuple[tuple[int, int, int], int, tuple[int, int, int]]:
    """Compute strides and the number of patches for a 3-D volume."""
    if len(data_shape) != 3 or len(patch_size) != 3:
        raise ValueError("data_shape and patch_size must each contain 3 values")

    dimensions = [
        compute_stride_and_n_patches(int(length), int(patch))
        for length, patch in zip(data_shape, patch_size)
    ]
    strides = tuple(item[0] for item in dimensions)
    counts = tuple(item[1] for item in dimensions)
    total = counts[0] * counts[1] * counts[2]
    return strides, total, counts


def patchify(
    data: Tensor,
    patch_size: Sequence[int] = (64, 64, 8),
    *,
    temporal_boundary_mode: str = "balanced_full",
) -> tuple[Tensor, tuple[int, int, int]]:
    """Extract zero-padded overlapping patches from a 3-D tensor."""
    if data.ndim != 3:
        raise ValueError(f"data must be 3-D, got shape {tuple(data.shape)}")

    patch_size = tuple(int(value) for value in patch_size)
    strides, total, counts = compute_strides_and_N(data.shape, patch_size)
    temporal_plan = plan_temporal_windows(
        int(data.shape[2]),
        patch_size[2],
        strides[2],
        temporal_boundary_mode=temporal_boundary_mode,
    )
    if temporal_plan.patch_count != counts[2]:
        raise RuntimeError("temporal boundary planning changed the patch count")
    patches = torch.zeros(
        (total, *patch_size),
        dtype=data.dtype,
        device=data.device,
    )

    patch_index = 0
    for index0 in range(counts[0]):
        start0 = index0 * strides[0]
        end0 = min(start0 + patch_size[0], data.shape[0])
        for index1 in range(counts[1]):
            start1 = index1 * strides[1]
            end1 = min(start1 + patch_size[1], data.shape[1])
            for start2 in temporal_plan.starts:
                end2 = min(start2 + patch_size[2], data.shape[2])
                patches[
                    patch_index,
                    : end0 - start0,
                    : end1 - start1,
                    : end2 - start2,
                ] = data[start0:end0, start1:end1, start2:end2]
                patch_index += 1
    return patches, strides


def _axis_weights(
    length: int,
    patch_index: int,
    patch_count: int,
    overlap: int,
    *,
    device: torch.device,
) -> Tensor:
    """Build the live sampler's linear overlap weights for one axis."""
    left_length = min(overlap if patch_index > 0 else 0, length)
    right_length = min(overlap if patch_index < patch_count - 1 else 0, length)

    if left_length + right_length > length:
        if left_length and right_length:
            total = left_length + right_length
            left_length = max(1, int(round(length * left_length / total)))
            right_length = length - left_length
        else:
            left_length = min(left_length, length)
            right_length = length - left_length

    weights = torch.ones(length, dtype=torch.float32, device=device)
    if left_length == 1:
        weights[:1] = 0.5
    elif left_length > 1:
        weights[:left_length] = torch.linspace(
            0.0,
            1.0,
            steps=left_length,
            device=device,
        )

    if right_length == 1:
        weights[-1:] = 0.5
    elif right_length > 1:
        weights[-right_length:] = torch.linspace(
            1.0,
            0.0,
            steps=right_length,
            device=device,
        )
    return weights


def depatchify(
    patches: Tensor,
    data_shape: Sequence[int],
    patch_size: Sequence[int],
    strides: Sequence[int],
    *,
    temporal_boundary_mode: str = "balanced_full",
) -> Tensor:
    """Reconstruct a 3-D tensor with separable linear weighted overlap-add."""
    if patches.ndim != 4:
        raise ValueError(f"patches must be 4-D, got shape {tuple(patches.shape)}")
    if len(data_shape) != 3 or len(patch_size) != 3 or len(strides) != 3:
        raise ValueError("data_shape, patch_size, and strides must contain 3 values")

    data_shape = tuple(int(value) for value in data_shape)
    patch_size = tuple(int(value) for value in patch_size)
    strides = tuple(int(value) for value in strides)
    axis_counts = tuple(
        max(1, ceil((length - patch) / stride) + 1)
        for length, patch, stride in zip(data_shape, patch_size, strides)
    )
    temporal_plan = plan_temporal_windows(
        data_shape[2],
        patch_size[2],
        strides[2],
        temporal_boundary_mode=temporal_boundary_mode,
    )
    counts = (axis_counts[0], axis_counts[1], temporal_plan.patch_count)
    expected = counts[0] * counts[1] * counts[2]
    if patches.shape[0] != expected:
        raise ValueError(
            f"received {patches.shape[0]} patches, expected {expected} "
            f"({counts[0]}*{counts[1]}*{counts[2]})"
        )

    numerator = torch.zeros(data_shape, dtype=patches.dtype, device=patches.device)
    denominator = torch.zeros(data_shape, dtype=torch.float32, device=patches.device)
    overlaps = tuple(max(0, patch - stride) for patch, stride in zip(patch_size, strides))

    patch_index = 0
    for index0 in range(counts[0]):
        start0 = index0 * strides[0]
        end0 = min(start0 + patch_size[0], data_shape[0])
        weight0 = _axis_weights(
            end0 - start0,
            index0,
            counts[0],
            overlaps[0],
            device=patches.device,
        )
        for index1 in range(counts[1]):
            start1 = index1 * strides[1]
            end1 = min(start1 + patch_size[1], data_shape[1])
            weight1 = _axis_weights(
                end1 - start1,
                index1,
                counts[1],
                overlaps[1],
                device=patches.device,
            )
            for index2, start2 in enumerate(temporal_plan.starts):
                end2 = min(start2 + patch_size[2], data_shape[2])
                if temporal_boundary_mode == "balanced_full":
                    weight2 = balanced_temporal_weights(
                        end2 - start2,
                        device=patches.device,
                    )
                else:
                    weight2 = _axis_weights(
                        end2 - start2,
                        index2,
                        counts[2],
                        overlaps[2],
                        device=patches.device,
                    )
                weights = (
                    weight0[:, None, None]
                    * weight1[None, :, None]
                    * weight2[None, None, :]
                )
                target = (
                    slice(start0, end0),
                    slice(start1, end1),
                    slice(start2, end2),
                )
                source = (
                    slice(0, end0 - start0),
                    slice(0, end1 - start1),
                    slice(0, end2 - start2),
                )
                numerator[target] += patches[patch_index][source] * weights.to(patches.dtype)
                denominator[target] += weights
                patch_index += 1

    denominator[denominator == 0] = 1.0
    return numerator / denominator.to(patches.dtype)
