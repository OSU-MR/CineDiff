"""MRI measurement and patch conversion used by the active CineDiff sampler.

This module contains only the reconstruction primitives exercised by
``GaussianDiffusion.ddim_sample``.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import scipy.io as sio
import torch
from einops import rearrange
from torch import Tensor

from utilities.data_consistency.patches import (
    TEMPORAL_BOUNDARY_MODES,
    balanced_temporal_weights,
    depatchify,
    patchify,
    plan_temporal_windows,
)
from utilities.data_consistency.soft_sense import (
    matlab_reconstruction_to_map_series,
    matlab_sense_maps_to_map_coil,
    resolve_map_set_count,
)
from utilities.data_consistency.prepared_kspace import load_prepared_kspace


PATCH_SIZE = (64, 64, 8)
SOFT_SENSE_GAMMA_S = 0.02

__all__ = [
    "A",
    "Ah",
    "H_operator",
    "PATCH_SIZE",
    "PatchConverter",
    "SOFT_SENSE_GAMMA_S",
    "devectorize_image",
    "soft_A",
    "soft_Ah",
    "vectorize_image",
]


@torch.jit.script
def vectorize_image(Y: Tensor, samp_4d_idx: Tensor) -> Tensor:
    return Y[samp_4d_idx[2], samp_4d_idx[0], samp_4d_idx[1], samp_4d_idx[3]]


@torch.jit.script
def devectorize_image(
    y: Tensor,
    samp_4d_idx: Tensor,
    kData_shape: List[int],
    dim_info: List[str],
) -> Tensor:
    kx = kData_shape[dim_info.index("kx")]
    ky = kData_shape[dim_info.index("ky")]
    phase = kData_shape[dim_info.index("phase")]
    coil_num = kData_shape[dim_info.index("coil")]
    Y = torch.zeros([coil_num, kx, ky, phase], dtype=y.dtype, device=y.device)
    Y[samp_4d_idx[2], samp_4d_idx[0], samp_4d_idx[1], samp_4d_idx[3]] = y
    return Y


@torch.jit.script
def A(img_x: Tensor, sense_maps: Tensor, samp_4d_idx: Tensor) -> Tensor:
    Y = []
    for i in range(sense_maps.shape[0]):
        if len(sense_maps.shape) == 3:
            img_sx = sense_maps[i, :, :, None] * img_x
        elif len(sense_maps.shape) == 4:
            img_sx = sense_maps[i, :, :, :] * img_x
        else:
            raise ValueError(
                "Please provide the dimension information for non-standard data"
            )
        kspace_sx = torch.fft.fftshift(
            torch.fft.fft2(
                torch.fft.ifftshift(img_sx, dim=(0, 1)),
                dim=(0, 1),
                norm="ortho",
            ),
            dim=(0, 1),
        )
        Y.append(kspace_sx)
    Y = torch.stack(Y)
    y = vectorize_image(Y, samp_4d_idx)
    return y


@torch.jit.script
def Ah(
    y: Tensor,
    sense_maps: Tensor,
    samp_4d_idx: Tensor,
    kData_shape: Optional[List[int]] = None,
    dim_info: Optional[List[str]] = None,
    return_Y: bool = False,
) -> Tensor:
    if kData_shape is None:
        kData_shape = [8, 208, 180, 16]
    else:
        kData_shape = [
            sense_maps.shape[0],
            sense_maps.shape[1],
            sense_maps.shape[2],
            kData_shape[2],
        ]
    if dim_info is None:
        dim_info = ["coil", "kx", "ky", "phase"]

    Y = devectorize_image(y, samp_4d_idx, kData_shape, dim_info)
    if return_Y:
        return Y

    img_x = torch.zeros(Y.shape[1:], dtype=Y.dtype, device=Y.device)
    for i in range(sense_maps.shape[0]):
        img_sx = Y[i]
        img_sx = torch.fft.fftshift(
            torch.fft.ifft2(
                torch.fft.ifftshift(img_sx, dim=(0, 1)),
                dim=(0, 1),
                norm="ortho",
            ),
            dim=(0, 1),
        )
        if len(sense_maps.shape) == 3:
            img_x += torch.conj(sense_maps[i, :, :, None]) * img_sx
        elif len(sense_maps.shape) == 4:
            img_x += torch.conj(sense_maps[i, :, :, :]) * img_sx
        else:
            raise ValueError(
                "Please provide the dimension information for non-standard data"
            )
    return img_x


@torch.jit.script
def soft_A(img_x: Tensor, sense_maps: Tensor, samp_4d_idx: Tensor) -> Tensor:
    """Apply the joint soft-SENSE forward model for multiple map sets."""
    if len(img_x.shape) != 4 or len(sense_maps.shape) != 4:
        raise ValueError(
            "soft-SENSE requires images [maps, kx, ky, phase] and "
            "sensitivity maps [maps, coils, kx, ky]"
        )
    if (
        img_x.shape[0] != sense_maps.shape[0]
        or img_x.shape[1] != sense_maps.shape[2]
        or img_x.shape[2] != sense_maps.shape[3]
    ):
        raise ValueError("soft-SENSE image and sensitivity-map shapes do not agree")
    coil_images = torch.sum(
        sense_maps[:, :, :, :, None] * img_x[:, None, :, :, :],
        dim=0,
    )
    kspace = torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(coil_images, dim=(1, 2)),
            dim=(1, 2),
            norm="ortho",
        ),
        dim=(1, 2),
    )
    return vectorize_image(kspace, samp_4d_idx)


@torch.jit.script
def soft_Ah(
    y: Tensor,
    sense_maps: Tensor,
    samp_4d_idx: Tensor,
    phase: int,
) -> Tensor:
    """Apply the adjoint joint soft-SENSE model for every map component."""
    if len(sense_maps.shape) != 4:
        raise ValueError(
            "soft-SENSE sensitivity maps must have shape [maps, coils, kx, ky]"
        )
    kdata_shape: List[int] = [
        sense_maps.shape[1],
        sense_maps.shape[2],
        sense_maps.shape[3],
        phase,
    ]
    dim_info: List[str] = ["coil", "kx", "ky", "phase"]
    kspace = devectorize_image(y, samp_4d_idx, kdata_shape, dim_info)
    coil_images = torch.fft.fftshift(
        torch.fft.ifft2(
            torch.fft.ifftshift(kspace, dim=(1, 2)),
            dim=(1, 2),
            norm="ortho",
        ),
        dim=(1, 2),
    )
    return torch.sum(
        torch.conj(sense_maps[:, :, :, :, None])
        * coil_images[None, :, :, :, :],
        dim=1,
    )


class H_operator:
    """Prepared-data MRI forward/adjoint operator for raw CineDiff reconstruction."""

    def __init__(
        self,
        device: torch.device,
        *,
        input_data_folder: str,
        num_sense_maps: int | None = None,
    ) -> None:
        if not input_data_folder:
            raise ValueError("input_data_folder is required")

        self.device = torch.device(device)
        self.input_data_folder = os.path.abspath(input_data_folder)
        kspace_array, sample_indices, data_shape = load_prepared_kspace(
            self.input_data_folder
        )
        data_shape = np.asarray(data_shape).reshape(-1)
        self.image_shape = tuple(int(size) for size in data_shape)
        if len(self.image_shape) != 3 or any(size <= 0 for size in self.image_shape):
            raise ValueError(
                "kspace.npz data_shape must contain [kx, ky, frame], "
                f"got {self.image_shape}"
            )

        self.samp_4d_idx = torch.as_tensor(
            sample_indices,
            dtype=torch.long,
            device=self.device,
        )
        if self.samp_4d_idx.ndim != 2 or self.samp_4d_idx.shape[0] != 4:
            raise ValueError(
                "kspace.npz sample indices must have shape [4, number_of_samples], "
                f"got {tuple(self.samp_4d_idx.shape)}"
            )
        if self.samp_4d_idx.shape[1] == 0:
            raise ValueError("kspace.npz contains no samples")

        sense_array = sio.loadmat(
            os.path.join(self.input_data_folder, "cs_S.mat")
        )["S"]
        if sense_array.ndim == 4 and sense_array.shape[2] == 1:
            source_maps = torch.as_tensor(
                rearrange(sense_array, "kx ky 1 c -> c kx ky"),
                dtype=torch.complex64,
                device=self.device,
            )
            self.source_map_set_count = 1
            self.map_set_count = resolve_map_set_count(1, num_sense_maps)
            self.sense_maps = source_maps
        elif sense_array.ndim == 5 and sense_array.shape[2] == 1:
            source_maps = torch.as_tensor(
                matlab_sense_maps_to_map_coil(sense_array),
                dtype=torch.complex64,
                device=self.device,
            )
            self.source_map_set_count = int(source_maps.shape[0])
            self.map_set_count = resolve_map_set_count(
                self.source_map_set_count,
                num_sense_maps,
            )
            selected_maps = source_maps[-self.map_set_count :]
            self.sense_maps = (
                selected_maps[0] if self.map_set_count == 1 else selected_maps
            )
        else:
            raise ValueError(
                "cs_S.mat must contain sensitivity maps shaped "
                "[kx, ky, 1, coils] or [kx, ky, 1, coils, map_sets], "
                f"got {sense_array.shape}"
            )
        self.diffusion_map_index = self.map_set_count - 1
        self.coil_count = int(
            self.sense_maps.shape[0]
            if self.map_set_count == 1
            else self.sense_maps.shape[1]
        )

        self._validate_operator_shapes()

        Y = torch.as_tensor(
            kspace_array,
            dtype=torch.complex64,
            device=self.device,
        )
        expected_y_shape = (
            self.coil_count,
            *self.image_shape,
        )
        if tuple(Y.shape) != expected_y_shape:
            raise ValueError(
                f"kspace.npz k-space must have shape {expected_y_shape}, got {tuple(Y.shape)}"
            )
        self.scaling_factor = torch.as_tensor(
            np.load(os.path.join(self.input_data_folder, "max_mag.npy")),
            dtype=torch.float32,
            device=self.device,
        )
        if (
            self.scaling_factor.numel() != 1
            or not torch.isfinite(self.scaling_factor).item()
            or self.scaling_factor.item() <= 0
        ):
            raise ValueError(
                "max_mag.npy must contain one finite, positive normalization value"
            )
        Y = Y / self.scaling_factor

        self._measurement = vectorize_image(Y, self.samp_4d_idx)
        measured = self._measurement.unsqueeze(0)
        self.y = torch.stack([measured.real, measured.imag], dim=-1).view(1, -1)
        self.ky = int(Y.shape[2])

        self.cs_init = self._load_cs_initialization()

        # Prime the scripted operators outside autograd, as the original
        # H_operator did before the first DPS gradient.  A single call is
        # sufficient to keep TorchScript's FFT backward graph stable.
        with torch.no_grad():
            self.H(self.cs_init)
            self.Ht(self.y)

    def _validate_operator_shapes(self) -> None:
        limits = (
            self.image_shape[0],
            self.image_shape[1],
            self.coil_count,
            self.image_shape[2],
        )
        for name, indices, limit in zip(
            ("kx", "ky", "coil", "frame"),
            self.samp_4d_idx,
            limits,
        ):
            minimum = int(torch.min(indices).item())
            maximum = int(torch.max(indices).item())
            if minimum < 0 or maximum >= limit:
                raise ValueError(
                    f"Sampling {name} indices must be in [0, {limit}), "
                    f"got [{minimum}, {maximum}]"
                )

    def _load_cs_initialization(self) -> Tensor:
        cs_path = os.path.join(self.input_data_folder, "CS_recon.mat")
        cs_contents = sio.loadmat(cs_path)
        if "image_data" not in cs_contents:
            raise KeyError(f"{cs_path} does not contain 'image_data'")
        cs_array = cs_contents["image_data"]
        if self.source_map_set_count > 1:
            source_components = matlab_reconstruction_to_map_series(
                cs_array,
                map_set_count=self.source_map_set_count,
            )
            expected_shape = (self.source_map_set_count, *self.image_shape)
            if tuple(source_components.shape) != expected_shape:
                raise ValueError(
                    f"CS_recon.mat must contain shape {expected_shape}, "
                    f"got {tuple(source_components.shape)}"
                )
            components = source_components[-self.map_set_count :]
            if self.map_set_count == 1:
                cs_init = torch.as_tensor(
                    components[-1],
                    dtype=torch.complex64,
                    device=self.device,
                ) / self.scaling_factor
                return self.c2i(cs_init).unsqueeze(0)
            self._component_state = torch.as_tensor(
                components,
                dtype=torch.complex64,
                device=self.device,
            ) / self.scaling_factor
            return self.c2i(
                self._component_state[self.diffusion_map_index]
            ).unsqueeze(0)

        cs_init = torch.as_tensor(
            cs_array,
            dtype=torch.complex64,
            device=self.device,
        )
        cs_init = cs_init.squeeze()
        if tuple(cs_init.shape) != self.image_shape:
            raise ValueError(
                f"CS_recon.mat must contain shape {self.image_shape}, "
                f"got {tuple(cs_init.shape)}"
            )
        cs_max_mag = torch.max(torch.abs(cs_init))
        if cs_max_mag.item() == 0:
            raise ValueError("CS_recon.mat has zero magnitude and cannot be normalized")
        return self.c2i(cs_init / cs_max_mag).unsqueeze(0)

    def i2c(self, input_tensor: Tensor) -> Tensor:
        """Convert real/imag channels ``[2, kx, ky*frame]`` to a complex cine."""

        channels = rearrange(
            input_tensor,
            "c kx (ky phase) -> c kx ky phase",
            ky=self.ky,
        )
        return torch.complex(channels[0], channels[1])

    def c2i(self, input_tensor: Tensor) -> Tensor:
        """Convert a complex cine to real/imag channels ``[2, kx, ky*frame]``."""

        channels = torch.stack([input_tensor.real, input_tensor.imag], dim=0)
        return rearrange(channels, "c kx ky phase -> c kx (ky phase)")

    def _component_images(self, primary_image: Tensor) -> Tensor:
        """Insert the learned last-map image into the current component state."""
        primary = self.i2c(primary_image.squeeze(0)).to(self.device)
        components = self._component_state.clone()
        components[self.diffusion_map_index] = primary
        return components

    def Ht(self, y: Tensor, sense_maps: Optional[Tensor] = None) -> Tensor:
        """Apply the adjoint MRI operator to flattened real/imag measurements."""

        maps = self.sense_maps if sense_maps is None else sense_maps.squeeze(0)
        complex_y = torch.complex(
            y.view(y.shape[0], -1, 2)[..., 0],
            y.view(y.shape[0], -1, 2)[..., 1],
        ).squeeze(0)
        if self.map_set_count > 1:
            output = soft_Ah(
                complex_y,
                maps,
                self.samp_4d_idx,
                self.image_shape[2],
            )[self.diffusion_map_index]
            return self.c2i(output).unsqueeze(0).to(output.device)
        output = Ah(
            complex_y,
            maps,
            self.samp_4d_idx,
            kData_shape=list(self.image_shape),
        )
        return self.c2i(output).unsqueeze(0).to(output.device)

    def H(self, x: Tensor, sense_maps: Optional[Tensor] = None) -> Tensor:
        """Apply the MRI operator and flatten complex measurements as real/imag."""

        maps = self.sense_maps if sense_maps is None else sense_maps.squeeze(0)
        if self.map_set_count > 1:
            output = soft_A(
                self._component_images(x),
                maps,
                self.samp_4d_idx,
            ).unsqueeze(0)
            return torch.stack([output.real, output.imag], dim=-1).view(
                output.shape[0], -1
            )
        complex_x = self.i2c(x.squeeze(0)).to(self.device)
        output = A(complex_x, maps, self.samp_4d_idx).unsqueeze(0)
        return torch.stack([output.real, output.imag], dim=-1).view(output.shape[0], -1)

    @torch.no_grad()
    def update_auxiliary_components(
        self,
        primary_image: Tensor,
        *,
        data_weight: float,
        l1_weight: float,
    ) -> None:
        """Update non-diffusion map components by one complex-L1 proximal step."""
        if self.map_set_count == 1:
            return
        data_weight = float(data_weight)
        l1_weight = float(l1_weight)
        if not np.isfinite(data_weight) or data_weight <= 0:
            raise ValueError("gamma_w must be finite and positive for multi-map DPS")
        if not np.isfinite(l1_weight) or l1_weight < 0:
            raise ValueError("gamma_s must be finite and non-negative")

        components = self._component_images(primary_image.detach())
        residual = (
            soft_A(components, self.sense_maps, self.samp_4d_idx)
            - self._measurement
        )
        backprojection = soft_Ah(
            residual,
            self.sense_maps,
            self.samp_4d_idx,
            self.image_shape[2],
        )

        sensitivity_energy = torch.sum(
            torch.abs(self.sense_maps[: self.diffusion_map_index]) ** 2,
            dim=(0, 1),
        )
        lipschitz_bound = float(torch.max(sensitivity_energy).item())
        if not np.isfinite(lipschitz_bound) or lipschitz_bound <= 0:
            raise RuntimeError("auxiliary sensitivity maps have zero or invalid energy")
        step_size = 0.95 / (data_weight * lipschitz_bound)

        auxiliary = (
            self._component_state[: self.diffusion_map_index]
            - step_size
            * data_weight
            * backprojection[: self.diffusion_map_index]
        )
        threshold = step_size * l1_weight
        if threshold:
            magnitude = torch.abs(auxiliary)
            epsilon = torch.finfo(magnitude.dtype).eps
            shrinkage = torch.clamp(
                1.0 - threshold / magnitude.clamp_min(epsilon),
                min=0.0,
            )
            auxiliary = auxiliary * shrinkage

        updated_state = self._component_state.clone()
        updated_state[: self.diffusion_map_index] = auxiliary
        self._component_state = updated_state.detach()

    def components_from_primary(self, primary_image: Tensor) -> Tensor:
        """Return all complex map components in their original map order."""
        if self.map_set_count == 1:
            return self.i2c(primary_image.squeeze(0)).unsqueeze(0)
        return self._component_images(primary_image)


class PatchConverter:
    """Convert between overlapping diffusion patches and a full complex cine."""

    def __init__(
        self,
        data_shape: Sequence[int],
        strides: Sequence[int],
        patch_size: Sequence[int] = PATCH_SIZE,
        *,
        temporal_boundary_mode: str = "balanced_full",
    ) -> None:
        self.data_shape = tuple(int(size) for size in data_shape)
        self.strides = tuple(int(stride) for stride in strides)
        self.patch_size = tuple(int(size) for size in patch_size)
        if len(self.data_shape) != 3 or len(self.strides) != 3 or len(self.patch_size) != 3:
            raise ValueError("data_shape, strides, and patch_size must each contain 3 values")
        if temporal_boundary_mode not in TEMPORAL_BOUNDARY_MODES:
            raise ValueError(
                "temporal_boundary_mode must be 'legacy_pad' or 'balanced_full', "
                f"got {temporal_boundary_mode!r}"
            )
        self.temporal_boundary_mode = temporal_boundary_mode
        self.temporal_plan = plan_temporal_windows(
            self.data_shape[2],
            self.patch_size[2],
            self.strides[2],
            temporal_boundary_mode=temporal_boundary_mode,
        )
        spatial_counts = tuple(
            max(1, (length - patch + stride - 1) // stride + 1)
            for length, patch, stride in zip(
                self.data_shape[:2],
                self.patch_size[:2],
                self.strides[:2],
            )
        )
        self.patch_counts = (
            spatial_counts[0],
            spatial_counts[1],
            self.temporal_plan.patch_count,
        )
        self.patch_count = (
            self.patch_counts[0]
            * self.patch_counts[1]
            * self.patch_counts[2]
        )

    def p2i(
        self,
        patches: Tensor,
        offset: Sequence[int] = (0, 0, 0),
    ) -> Tensor:
        """Convert ``[patch, 2, frame, kx, ky]`` patches to a full cine image."""

        complex_patches = patches[:, 0, ...] + 1j * patches[:, 1, ...]
        complex_patches = rearrange(
            complex_patches,
            "N phase kx ky -> N kx ky phase",
        )
        image = depatchify(
            complex_patches,
            data_shape=self.data_shape,
            patch_size=self.patch_size,
            strides=self.strides,
            temporal_boundary_mode=self.temporal_boundary_mode,
        )
        image = torch.roll(
            image,
            shifts=[-int(value) for value in offset],
            dims=(0, 1, 2),
        )
        channels = torch.stack([image.real, image.imag], dim=0)
        return rearrange(channels, "c kx ky phase -> c kx (ky phase)")

    def i2p(
        self,
        image: Tensor,
        offset: Sequence[int] = (0, 0, 0),
    ) -> Tensor:
        """Convert a full real/imag cine image to overlapping diffusion patches."""

        channels = rearrange(
            image,
            "c kx (ky phase) -> c kx ky phase",
            ky=self.data_shape[1],
        )
        complex_image = torch.complex(channels[0], channels[1])
        complex_image = torch.roll(
            complex_image,
            shifts=[int(value) for value in offset],
            dims=(0, 1, 2),
        )
        patches, actual_strides = patchify(
            complex_image,
            patch_size=self.patch_size,
            temporal_boundary_mode=self.temporal_boundary_mode,
        )
        if actual_strides != self.strides:
            raise ValueError(
                f"configured strides {self.strides} do not match patchify strides "
                f"{actual_strides}"
            )
        patches = rearrange(patches, "N kx ky phase -> N phase kx ky")
        return torch.stack([patches.real, patches.imag], dim=1)

    def project_consensus(self, patches: Tensor) -> Tensor:
        """Synchronize temporal copies without blending spatial patches."""
        if self.temporal_boundary_mode == "legacy_pad":
            return patches
        return self._project_temporal_consensus(patches)

    def _expected_patch_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.patch_count,
            2,
            self.patch_size[2],
            self.patch_size[0],
            self.patch_size[1],
        )

    def _validate_patch_shape(self, patches: Tensor, *, operation: str) -> None:
        expected = self._expected_patch_shape()
        if tuple(patches.shape) != expected:
            raise ValueError(
                f"temporal {operation} shape does not match the patch plan: "
                f"expected {expected}, got {tuple(patches.shape)}"
            )

    def _project_temporal_consensus(self, patches: Tensor) -> Tensor:
        self._validate_patch_shape(patches, operation="consensus")
        spatial_count = self.patch_counts[0] * self.patch_counts[1]
        grouped = patches.reshape(
            spatial_count,
            self.temporal_plan.patch_count,
            *patches.shape[1:],
        )
        canonical = patches.new_zeros(
            spatial_count,
            patches.shape[1],
            self.data_shape[2],
            patches.shape[3],
            patches.shape[4],
        )
        denominator = torch.zeros(
            self.data_shape[2],
            dtype=torch.float32,
            device=patches.device,
        )
        weights = balanced_temporal_weights(
            self.patch_size[2],
            device=patches.device,
        )
        patch_weights = weights.to(patches.dtype).view(1, 1, -1, 1, 1)
        for window_index, start in enumerate(self.temporal_plan.starts):
            canonical[:, :, start : start + self.patch_size[2]] += (
                grouped[:, window_index] * patch_weights
            )
            denominator[start : start + self.patch_size[2]] += weights
        if bool(torch.any(denominator == 0)):
            raise RuntimeError("temporal consensus plan leaves frames uncovered")
        canonical = canonical / denominator.to(patches.dtype).view(
            1,
            1,
            -1,
            1,
            1,
        )
        projected = torch.stack(
            [
                canonical[:, :, start : start + self.patch_size[2]]
                for start in self.temporal_plan.starts
            ],
            dim=1,
        )
        return projected.reshape_as(patches)

    def pullback_temporal_gradient(self, gradients: Tensor) -> Tensor:
        """Sum duplicate-frame gradients and copy them to every temporal window."""
        if self.temporal_boundary_mode == "legacy_pad":
            return gradients
        self._validate_patch_shape(gradients, operation="gradient")
        spatial_count = self.patch_counts[0] * self.patch_counts[1]
        grouped = gradients.reshape(
            spatial_count,
            self.temporal_plan.patch_count,
            *gradients.shape[1:],
        )
        canonical = gradients.new_zeros(
            spatial_count,
            gradients.shape[1],
            self.data_shape[2],
            gradients.shape[3],
            gradients.shape[4],
        )
        for window_index, start in enumerate(self.temporal_plan.starts):
            canonical[:, :, start : start + self.patch_size[2]] += grouped[
                :, window_index
            ]
        pulled_back = torch.stack(
            [
                canonical[:, :, start : start + self.patch_size[2]]
                for start in self.temporal_plan.starts
            ],
            dim=1,
        )
        return pulled_back.reshape_as(gradients)
