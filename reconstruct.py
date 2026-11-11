#!/usr/bin/env python
"""Reconstruct one bundled CineDiff case from data/cases.json.

Example::

    python reconstruct.py --case fs_0008_1_5T_sli0_R16 --output-root outputs/example
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from utilities.data_consistency.mri import PatchConverter, SOFT_SENSE_GAMMA_S
from utilities.data_consistency.prepared_kspace import load_data_shape
from utilities.data_consistency.patches import (
    TEMPORAL_BOUNDARY_MODES,
    compute_strides_and_N,
)
from utilities.data_consistency.soft_sense import (
    matlab_reconstruction_to_map_series,
    matlab_sense_maps_to_map_coil,
    resolve_map_set_count,
)
from utilities.models.video_diffusion_model.video_diffusion_pytorch import (
    GaussianDiffusion,
    Unet3D,
)
from utilities.reproducibility import seed_everything, validate_seed


MODEL_DIM = 64
MODEL_CHANNELS = 2
MODEL_FRAMES = 8
MODEL_TIMESTEPS = 1000
CHECKPOINT_MILESTONE = 192000
CHECKPOINT_SHA256 = "53b9239eb68eeb021115bedd7c21d4d8a94864eef7c05f1069a13a9743dc1535"
# Standard DDIM stochasticity; the CLI may explicitly override this value.
DDIM_ETA = 0.85


def _sampling_trajectories(
    first_seed: int,
    num_samples: int,
) -> list[int]:
    """Return consecutive independent seeds for reconstruction."""
    return [first_seed + index for index in range(num_samples)]


def _sampling_eta(*, eta: float | None = None) -> float:
    """Resolve an optional DDIM eta override against the mode default."""
    if eta is None:
        return DDIM_ETA
    if (
        not isinstance(eta, (int, float))
        or isinstance(eta, bool)
        or not math.isfinite(eta)
        or not 0 <= eta <= 1
    ):
        raise ValueError("eta must be a number between 0 and 1")
    return 0.0 if eta == 0.0 else float(eta)


def _trajectory_stem(
    seed: int,
) -> str:
    """Return a collision-free artifact stem for one trajectory."""
    stem = f"seed_{seed:010d}"
    return stem


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_checkpoint(path: Path) -> bytes:
    """Return the checkpoint bytes after checking them against the release."""
    if not path.is_file():
        raise FileNotFoundError(
            f"CineDiff weights not found at {path}; run `python download.py` first"
        )
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError(
            f"CineDiff checkpoint at {path} is incomplete or corrupted "
            "(SHA256 does not match the released model); delete it and run "
            "`python download.py` again"
        )
    return data


def _load_complex_mat(
    path: Path,
    expected_shape: Sequence[int],
    *,
    source_map_set_count: int,
) -> np.ndarray:
    from scipy.io import loadmat

    if not path.is_file():
        raise FileNotFoundError(f"required prepared input not found: {path}")
    contents = loadmat(path)
    if "image_data" not in contents:
        raise KeyError(f"{path} does not contain 'image_data'")
    shape = tuple(int(value) for value in expected_shape)
    if source_map_set_count == 1:
        volume = np.asarray(contents["image_data"]).squeeze()
    else:
        components = matlab_reconstruction_to_map_series(
            contents["image_data"],
            map_set_count=source_map_set_count,
        )
        expected_components_shape = (source_map_set_count, *shape)
        if components.shape != expected_components_shape:
            raise ValueError(
                f"{path.name} has shape {components.shape}, "
                f"expected {expected_components_shape}"
            )
        volume = components[-1]
    if volume.shape != shape:
        raise ValueError(f"{path.name} has shape {volume.shape}, expected {shape}")
    if not np.isfinite(volume).all():
        raise ValueError(f"{path.name} contains non-finite values")
    return volume.astype(np.complex64, copy=False)


def _load_reference_and_cs(
    prepared_directory: Path,
    data_shape: Sequence[int],
    *,
    num_sense_maps: int | None = None,
    reference_required: bool = True,
) -> tuple[np.ndarray | None, np.ndarray, float, int, int]:
    normalization_path = prepared_directory / "max_mag.npy"
    if not normalization_path.is_file():
        raise FileNotFoundError(
            f"required prepared input not found: {normalization_path}"
        )
    normalization_values = np.asarray(np.load(normalization_path)).squeeze()
    if normalization_values.size != 1:
        raise ValueError("max_mag.npy must contain exactly one value")
    normalization = float(normalization_values)
    if not math.isfinite(normalization) or normalization == 0.0:
        raise ValueError("max_mag.npy must contain one finite, non-zero value")

    sensitivity_path = prepared_directory / "cs_S.mat"
    if not sensitivity_path.is_file():
        raise FileNotFoundError(f"required prepared input not found: {sensitivity_path}")
    from scipy.io import loadmat

    sensitivity_contents = loadmat(sensitivity_path)
    if "S" not in sensitivity_contents:
        raise KeyError(f"{sensitivity_path} does not contain 'S'")
    source_map_set_count = int(
        matlab_sense_maps_to_map_coil(sensitivity_contents["S"]).shape[0]
    )
    map_set_count = resolve_map_set_count(source_map_set_count, num_sense_maps)

    reference = None
    if reference_required:
        reference = _load_complex_mat(
            prepared_directory / "CS_recon_full.mat",
            data_shape,
            source_map_set_count=source_map_set_count,
        )
    cs_reconstruction = _load_complex_mat(
        prepared_directory / "CS_recon.mat",
        data_shape,
        source_map_set_count=source_map_set_count,
    )
    return (
        None if reference is None else reference / normalization,
        cs_reconstruction / normalization,
        normalization,
        map_set_count,
        source_map_set_count,
    )


def _patches_to_image(
    patches: torch.Tensor,
    *,
    data_shape: Sequence[int],
    strides: Sequence[int],
    temporal_boundary_mode: str = "balanced_full",
) -> np.ndarray:
    converter = PatchConverter(
        data_shape=data_shape,
        strides=strides,
        temporal_boundary_mode=temporal_boundary_mode,
    )
    channels = converter.p2i(patches.detach().to("cpu"))
    height, width, frames = (int(value) for value in data_shape)
    channels = channels.reshape(2, height, width, frames)
    image = torch.complex(channels[0], channels[1])
    return image.numpy().astype(np.complex64, copy=False)


class _PerceptualMetrics:
    """Lazily construct the two heavy learned perceptual metrics."""

    def __init__(self, device: torch.device, batch_size: int = 8) -> None:
        try:
            import lpips
            from DISTS_pytorch import DISTS
        except ImportError as exc:
            raise RuntimeError(
                "LPIPS and DISTS are required to write the requested metric set"
            ) from exc

        self.device = device
        self.batch_size = batch_size
        self.lpips = lpips.LPIPS(net="vgg").eval().to(device)
        self.dists = DISTS().eval().to(device)
        for model in (self.lpips, self.dists):
            model.requires_grad_(False)

    @staticmethod
    def _frames(volume: np.ndarray) -> torch.Tensor:
        magnitude = np.clip(np.abs(volume), 0.0, 1.0).astype(np.float32)
        tensor = torch.from_numpy(np.moveaxis(magnitude, -1, 0)).unsqueeze(1)
        return tensor.repeat(1, 3, 1, 1).contiguous()

    def __call__(self, reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
        reference_frames = self._frames(reference)
        candidate_frames = self._frames(candidate)
        lpips_values = []
        dists_values = []
        with torch.no_grad():
            for start in range(0, reference_frames.shape[0], self.batch_size):
                stop = start + self.batch_size
                first = reference_frames[start:stop].to(self.device)
                second = candidate_frames[start:stop].to(self.device)
                lpips_values.extend(
                    self.lpips(first, second).detach().reshape(-1).cpu().tolist()
                )
                dists_values.extend(
                    self.dists(first, second).detach().reshape(-1).cpu().tolist()
                )
        return float(np.mean(lpips_values)), float(np.mean(dists_values))


def _ratio_db(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return math.inf if numerator > 0.0 else 0.0
    if numerator == 0.0:
        return -math.inf
    return 10.0 * math.log10(numerator / denominator)


def _compute_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    perceptual: _PerceptualMetrics,
) -> dict[str, float]:
    from skimage.metrics import structural_similarity

    if reference.shape != candidate.shape:
        raise ValueError(
            f"metric inputs must have equal shapes, got {reference.shape} and {candidate.shape}"
        )
    difference = candidate - reference
    reference_power = float(np.mean(np.abs(reference) ** 2))
    error_power = float(np.mean(np.abs(difference) ** 2))
    peak_power = float(np.max(np.abs(reference)) ** 2)

    reference_magnitude = np.abs(reference)
    candidate_magnitude = np.abs(candidate)
    data_range = float(max(reference_magnitude.max(), candidate_magnitude.max()))
    if data_range == 0.0:
        ssim_value = 1.0
    else:
        smallest_axis = min(reference_magnitude.shape)
        window = min(7, smallest_axis if smallest_axis % 2 else smallest_axis - 1)
        if window < 3:
            raise ValueError(
                f"image shape {reference.shape} is too small for SSIM"
            )
        ssim_value = float(
            structural_similarity(
                reference_magnitude,
                candidate_magnitude,
                data_range=data_range,
                win_size=window,
            )
        )

    lpips_value, dists_value = perceptual(reference, candidate)
    return {
        "snr": _ratio_db(reference_power, error_power),
        "psnr": _ratio_db(peak_power, error_power),
        "ssim": ssim_value,
        "lpips": lpips_value,
        "dists": dists_value,
    }


def _metric_text(metrics: Mapping[str, float]) -> str:
    return (
        f"SNR {metrics['snr']:.2f} | PSNR {metrics['psnr']:.2f}\n"
        f"SSIM {metrics['ssim']:.3f} | LPIPS {metrics['lpips']:.4f} | "
        f"DISTS {metrics['dists']:.4f}"
    )


def _save_visualizations(
    reference: np.ndarray | None,
    cinediff: np.ndarray,
    cs_reconstruction: np.ndarray | None,
    metrics: Mapping[str, Mapping[str, float]],
    output_directory: Path,
    *,
    evaluation_name: str,
    panel_height_inches: float = 3.5,
    dpi: int = 150,
) -> Path:
    """Save a comparison GIF of the cine magnitude images.

    With a reference, the top row shows the reference, CineDiff, and CS images
    and the bottom row the CineDiff and CS error maps on one shared scale.
    Without a reference, only CineDiff and CS are shown.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    def frames(volume: np.ndarray) -> np.ndarray:
        return np.moveaxis(np.abs(volume), -1, 0)

    def upper_limit(stacks, percentile: float) -> float:
        value = float(np.percentile(np.concatenate([s.ravel() for s in stacks]), percentile))
        return value if math.isfinite(value) and value > 0 else 1.0

    # (title, frame stack, colormap, row, column); error maps share one scale.
    panels = []
    if reference is None:
        panels.append(("CineDiff", frames(cinediff), "gray", 0, 0))
        if cs_reconstruction is not None:
            panels.append(("CS reconstruction", frames(cs_reconstruction), "gray", 0, 1))
    else:
        reference_frames = frames(reference)
        panels.append(("Reference", reference_frames, "gray", 0, 0))
        panels.append(
            ("CineDiff\n" + _metric_text(metrics["CineDiff"]), frames(cinediff), "gray", 0, 1)
        )
        panels.append(
            ("|CineDiff - reference|", frames(cinediff - reference), "magma", 1, 1)
        )
        if cs_reconstruction is not None:
            panels.append(
                ("CS reconstruction\n" + _metric_text(metrics["CS"]),
                 frames(cs_reconstruction), "gray", 0, 2)
            )
            panels.append(
                ("|CS - reference|", frames(cs_reconstruction - reference), "magma", 1, 2)
            )
    image_vmax = upper_limit([p[1] for p in panels if p[2] == "gray"], 98)
    error_stacks = [p[1] for p in panels if p[2] == "magma"]
    error_vmax = upper_limit(error_stacks, 99) if error_stacks else 1.0

    rows = 1 + max(p[3] for p in panels)
    columns = 1 + max(p[4] for p in panels)
    frame_count, height, width = panels[0][1].shape
    panel_width_inches = panel_height_inches * width / height
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            columns * panel_width_inches,
            rows * panel_height_inches + (1.0 if reference is not None else 0.5),
        ),
        squeeze=False,
        constrained_layout=True,
        dpi=dpi,
    )
    for axis in axes.ravel():
        axis.axis("off")
    images = []
    for title, stack, colormap, row, column in panels:
        axis = axes[row, column]
        image = axis.imshow(
            stack[0],
            cmap=colormap,
            vmin=0,
            vmax=error_vmax if colormap == "magma" else image_vmax,
            interpolation="nearest",
        )
        axis.set_title(title, fontsize=9)
        images.append((image, stack))
    display_name = evaluation_name.replace("_", " ").title()
    suptitle = figure.suptitle(f"{display_name} | Frame 1/{frame_count}")

    # The automatic layout shifts the panels slightly on each of the first few
    # draws; settle it, then freeze it so every GIF frame has the same layout.
    previous = None
    for _ in range(10):
        figure.canvas.draw()
        current = [axis.get_position().bounds for axis in figure.axes]
        if current == previous:
            break
        previous = current
    figure.set_layout_engine("none")

    def render(frame_number: int) -> "Image.Image":
        for image, stack in images:
            image.set_data(stack[frame_number])
        suptitle.set_text(f"{display_name} | Frame {frame_number + 1}/{frame_count}")
        figure.canvas.draw()
        return Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())

    gif_frames = []
    try:
        # One palette for the whole GIF, built from a sample of frames, keeps
        # unchanged pixels identical from frame to frame.
        samples = [render(i) for i in np.linspace(0, frame_count - 1, min(frame_count, 8)).astype(int)]
        montage = Image.new("RGB", (samples[0].width, samples[0].height * len(samples)))
        for index, sample in enumerate(samples):
            montage.paste(sample, (0, index * sample.height))
        palette = montage.quantize(
            colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
        )
        del samples, montage
        for frame_number in range(frame_count):
            gif_frames.append(
                render(frame_number).quantize(palette=palette, dither=Image.Dither.NONE)
            )
    finally:
        plt.close(figure)
    gif_path = output_directory / "comparison.gif"
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=100,
        loop=0,
    )
    return gif_path


def _json_metric(value: float) -> float | str:
    if math.isnan(value):
        return "nan"
    if math.isfinite(value):
        return value
    return "inf" if value > 0 else "-inf"


def _write_evaluation(
    *,
    evaluation_name: str,
    reference: np.ndarray | None,
    cinediff: np.ndarray,
    cs_reconstruction: np.ndarray,
    perceptual: "_PerceptualMetrics",
    output_root: Path,
    dataset_name: str,
    slice_number: int,
    gamma_w: float,
    nfe: int,
    eta: float,
    first_seed: int,
    num_samples: int,
) -> dict[str, Any]:
    """Calculate and save one post-reconstruction evaluation view."""
    evaluation_directory = output_root / "evaluations" / evaluation_name
    evaluation_directory.mkdir(parents=True, exist_ok=True)
    metrics = (
        {
            "CineDiff": _compute_metrics(reference, cinediff, perceptual),
            "CS": _compute_metrics(reference, cs_reconstruction, perceptual),
        }
        if reference is not None
        else {}
    )

    metrics_path = evaluation_directory / "metrics.csv"
    metric_names = ("snr", "psnr", "ssim", "lpips", "dists")
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = (
            "dataset",
            "slice",
            "evaluation",
            "method",
            "gamma_w",
            "nfe",
            "eta",
            "first_seed",
            "num_samples",
            *metric_names,
        )
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for method, values in metrics.items():
            writer.writerow(
                {
                    "dataset": dataset_name,
                    "slice": slice_number,
                    "evaluation": evaluation_name,
                    "method": method,
                    "gamma_w": gamma_w,
                    "nfe": nfe,
                    "eta": eta,
                    "first_seed": first_seed,
                    "num_samples": num_samples,
                    **values,
                }
            )

    gif_path = _save_visualizations(
        reference,
        cinediff,
        cs_reconstruction,
        metrics,
        evaluation_directory,
        evaluation_name=evaluation_name,
    )
    return {
        "reference_available": reference is not None,
        "metrics": {
            method: {name: _json_metric(value) for name, value in values.items()}
            for method, values in metrics.items()
        },
        "outputs": {
            "metrics": str(metrics_path.relative_to(output_root)),
            "gif": str(gif_path.relative_to(output_root)),
        },
    }


class _PatchBatchDataParallel(torch.nn.DataParallel):
    """Scatter independent denoiser patches while preserving their order."""

    def forward(self, inputs, noise_condition, *args, **kwargs):
        if inputs.ndim < 1 or inputs.shape[0] < 1:
            raise ValueError("denoiser inputs must have a non-empty batch dimension")
        if noise_condition.ndim != 1:
            raise ValueError("denoiser noise_condition must be one-dimensional")
        if noise_condition.shape[0] == 1 and inputs.shape[0] != 1:
            noise_condition = noise_condition.expand(inputs.shape[0])
        elif noise_condition.shape[0] != inputs.shape[0]:
            raise ValueError(
                "denoiser noise_condition must contain one value or one value "
                "per input patch"
            )
        return super().forward(inputs, noise_condition, *args, **kwargs)


class CineDiffReconstructor:
    """Load the fixed CineDiff checkpoint once and reconstruct prepared cases."""

    def __init__(self, devices: Sequence[int] | None = None) -> None:
        """Use the given CUDA devices, or every visible one when ``devices`` is None.

        The denoiser's patch batch is split across the devices; the first one
        holds the model, the measurements, and the outputs.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CineDiff reconstruction requires a CUDA-capable GPU")
        visible_device_count = torch.cuda.device_count()
        if devices is None:
            devices = range(visible_device_count)
        if isinstance(devices, (str, bytes)):
            raise ValueError("devices must be a sequence of CUDA indices")
        self.device_indices = tuple(devices)
        if not self.device_indices:
            raise ValueError("devices cannot be empty")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.device_indices
        ):
            raise ValueError("devices must contain non-negative integers")
        if len(set(self.device_indices)) != len(self.device_indices):
            raise ValueError("devices must be unique")
        unavailable = [
            index for index in self.device_indices if index >= visible_device_count
        ]
        if unavailable:
            raise ValueError(
                f"CUDA device(s) {unavailable} unavailable; "
                f"process sees {visible_device_count} device(s)"
            )
        self.device_index = self.device_indices[0]
        self.device = torch.device(f"cuda:{self.device_index}")
        torch.cuda.set_device(self.device)

        self.checkpoint_path = (
            Path(__file__).resolve().parent
            / "weights"
            / f"model-{CHECKPOINT_MILESTONE}.pt"
        )
        checkpoint_bytes = _read_checkpoint(self.checkpoint_path)

        denoiser = Unet3D(
            dim=MODEL_DIM,
            channels=MODEL_CHANNELS,
            dim_mults=(1, 2, 4, 8),
            init_dim=None,
            init_kernel_size=7,
        )
        diffusion = GaussianDiffusion(
            denoiser,
            image_size=MODEL_DIM,
            num_frames=MODEL_FRAMES,
            channels=MODEL_CHANNELS,
            timesteps=MODEL_TIMESTEPS,
            beta_scheduler="linear",
            beta_start=1e-5,
        )
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu")
        del checkpoint_bytes
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError(f"invalid CineDiff checkpoint: {self.checkpoint_path}")
        diffusion.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        diffusion = diffusion.to(self.device)
        if len(self.device_indices) > 1:
            diffusion.denoise_fn = _PatchBatchDataParallel(
                diffusion.denoise_fn,
                device_ids=list(self.device_indices),
                output_device=self.device_index,
            )
        self.diffusion = diffusion.eval()
        self.checkpoint_sha256 = CHECKPOINT_SHA256
        self._perceptual: _PerceptualMetrics | None = None

    def run(
        self,
        prepared_directory: str | os.PathLike[str],
        output_directory: str | os.PathLike[str],
        dataset_name: str,
        slice_number: int,
        gamma_w: float,
        nfe: int,
        first_seed: int,
        num_samples: int,
        gamma_s: float = SOFT_SENSE_GAMMA_S,
        num_sense_maps: int | None = None,
        eta: float | None = None,
        temporal_boundary_mode: str = "balanced_full",
    ) -> dict[str, Any]:
        """Reconstruct one case and save a complete, seeded result bundle."""
        prepared_path = Path(prepared_directory).expanduser().resolve(strict=True)
        if not prepared_path.is_dir():
            raise NotADirectoryError(f"prepared directory not found: {prepared_path}")
        output_path = Path(output_directory).expanduser().resolve()
        if output_path == prepared_path or prepared_path in output_path.parents:
            raise ValueError("output_directory must not be the prepared input directory")
        if not isinstance(nfe, int) or isinstance(nfe, bool) or not 2 <= nfe <= MODEL_TIMESTEPS:
            raise ValueError(f"nfe must be an integer between 2 and {MODEL_TIMESTEPS}")
        gamma_w = float(gamma_w)
        if not math.isfinite(gamma_w) or gamma_w < 0:
            raise ValueError("gamma_w must be finite and non-negative")
        gamma_s = float(gamma_s)
        if not math.isfinite(gamma_s) or gamma_s < 0:
            raise ValueError("gamma_s must be finite and non-negative")
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")
        if temporal_boundary_mode not in TEMPORAL_BOUNDARY_MODES:
            raise ValueError(
                "temporal_boundary_mode must be 'legacy_pad' or 'balanced_full', "
                f"got {temporal_boundary_mode!r}"
            )
        sampling_eta = _sampling_eta(
            eta=eta,
        )
        first_seed = validate_seed(first_seed)
        seed_count = num_samples
        last_seed = first_seed + seed_count - 1
        validate_seed(last_seed)
        if not isinstance(slice_number, int) or isinstance(slice_number, bool) or slice_number < 0:
            raise ValueError("slice_number must be a non-negative integer")
        if not dataset_name or Path(dataset_name).name != dataset_name:
            raise ValueError("dataset_name must be one non-empty path component")

        data_shape_values = np.asarray(load_data_shape(prepared_path)).reshape(-1)
        if data_shape_values.size != 3:
            raise ValueError("kspace.npz data_shape must contain exactly three dimensions")
        data_shape = tuple(int(value) for value in data_shape_values)
        if any(value <= 0 for value in data_shape):
            raise ValueError(f"invalid data shape: {data_shape}")
        strides, patch_count, _ = compute_strides_and_N(data_shape)

        (
            reference,
            cs_reconstruction,
            normalization,
            map_set_count,
            source_map_set_count,
        ) = _load_reference_and_cs(
            prepared_path,
            data_shape,
            num_sense_maps=num_sense_maps,
            reference_required=not dataset_name.startswith("us_"),
        )
        output_path.mkdir(parents=True, exist_ok=True)

        trajectories = _sampling_trajectories(
            first_seed,
            num_samples,
        )
        seeds = [seed for seed in trajectories]
        reconstructions = []
        component_reconstructions = []
        seed_outputs = []
        temporal_sampling_options = {"temporal_boundary_mode": temporal_boundary_mode}
        for seed in trajectories:
            seed_everything(seed)
            sampled_patches = self.diffusion.sample(
                batch_size=patch_count,
                input_data_folder=str(prepared_path),
                strides=strides,
                rand_seed=seed,
                eta=sampling_eta,
                gamma_w=gamma_w,
                gamma_s=gamma_s,
                num_sense_maps=num_sense_maps,
                NFE_tgt=nfe,
                **temporal_sampling_options,
            )
            if tuple(sampled_patches.shape) != (
                patch_count,
                MODEL_CHANNELS,
                MODEL_FRAMES,
                MODEL_DIM,
                MODEL_DIM,
            ):
                raise ValueError(
                    "sampler returned an unexpected shape: "
                    f"{tuple(sampled_patches.shape)}"
                )
            sampled_patches = sampled_patches.detach().to("cpu").contiguous()
            reconstruction = _patches_to_image(
                sampled_patches,
                data_shape=data_shape,
                strides=strides,
                temporal_boundary_mode=temporal_boundary_mode,
            )

            stem = _trajectory_stem(
                seed,
            )
            tensor_path = output_path / f"{stem}_patches.pt"
            reconstruction_path = output_path / f"{stem}_reconstruction.npy"
            torch.save(sampled_patches, tensor_path)
            np.save(reconstruction_path, reconstruction)
            seed_outputs.append(
                {
                    "seed": seed,
                    "patch_tensor": tensor_path.name,
                    "patch_tensor_sha256": _sha256(tensor_path),
                    "reconstruction": reconstruction_path.name,
                    "reconstruction_sha256": _sha256(reconstruction_path),
                }
            )
            if map_set_count > 1:
                components_tensor = self.diffusion.last_soft_sense_components
                if components_tensor is None:
                    raise RuntimeError(
                        "multi-map sampler did not return its soft-SENSE components"
                    )
                components = components_tensor.numpy().astype(
                    np.complex64,
                    copy=False,
                )
                expected_components_shape = (map_set_count, *data_shape)
                if components.shape != expected_components_shape:
                    raise ValueError(
                        "sampler returned soft-SENSE components with shape "
                        f"{components.shape}; expected {expected_components_shape}"
                    )
                if not np.allclose(
                    components[-1],
                    reconstruction,
                    rtol=1e-5,
                    atol=1e-6,
                ):
                    raise RuntimeError(
                        "sampler's diffusion component does not match its returned "
                        "primary reconstruction"
                    )
                components = components.copy()
                components[-1] = reconstruction
                components_path = output_path / f"{stem}_components.npy"
                np.save(components_path, components)
                seed_outputs[-1].update(
                    {
                        "components": components_path.name,
                        "components_sha256": _sha256(components_path),
                    }
                )
                component_reconstructions.append(components)
            reconstructions.append(reconstruction)
            del sampled_patches

        average_reconstruction = np.mean(
            np.stack(reconstructions, axis=0),
            axis=0,
            dtype=np.complex64,
        )
        average_path = output_path / "cinediff_reconstruction_average.npy"
        np.save(average_path, average_reconstruction)
        average_components_path = None
        if map_set_count > 1:
            average_components = np.mean(
                np.stack(component_reconstructions, axis=0),
                axis=0,
                dtype=np.complex64,
            )
            average_components_path = output_path / "soft_sense_components_average.npy"
            np.save(average_components_path, average_components)

        evaluation_reference = None if dataset_name.startswith("us_") else reference
        if evaluation_reference is not None and self._perceptual is None:
            self._perceptual = _PerceptualMetrics(self.device)
        evaluation_arguments = {
            "perceptual": self._perceptual,
            "output_root": output_path,
            "dataset_name": dataset_name,
            "slice_number": slice_number,
            "gamma_w": gamma_w,
            "nfe": nfe,
            "eta": sampling_eta,
            "first_seed": first_seed,
            "num_samples": num_samples,
        }
        evaluations = {
            "full_image": _write_evaluation(
                evaluation_name="full_image",
                reference=evaluation_reference,
                cinediff=average_reconstruction,
                cs_reconstruction=cs_reconstruction,
                **evaluation_arguments,
            )
        }

        manifest = {
            "schema_version": 3,
            "dataset": dataset_name,
            "slice": slice_number,
            "prepared_directory": str(prepared_path),
            "data_shape": list(data_shape),
            "patch_strides": list(strides),
            "patch_count": patch_count,
            "normalization": normalization,
            "model": {
                "milestone": CHECKPOINT_MILESTONE,
                "checkpoint": str(self.checkpoint_path),
                "checkpoint_sha256": self.checkpoint_sha256,
                "timesteps": MODEL_TIMESTEPS,
            },
            "sampling": {
                "gamma_w": gamma_w,
                "nfe": nfe,
                "eta": sampling_eta,
                "seeds": seeds,
            },
            "evaluations": evaluations,
            "outputs": {
                "seeds": seed_outputs,
                "average_reconstruction": average_path.name,
                "average_reconstruction_sha256": _sha256(average_path),
            },
        }
        if len(self.device_indices) > 1:
            manifest["runtime"] = {
                "cuda_devices": list(self.device_indices),
                "primary_cuda_device": self.device_index,
                "denoiser_parallelism": "patch_batch_data_parallel",
            }
        if temporal_boundary_mode == "balanced_full":
            manifest["sampling"]["temporal_boundary_mode"] = "balanced_full"
            manifest["sampling"]["coherent_patch_noise"] = True
        if map_set_count > 1:
            manifest["sampling"]["soft_sense"] = {
                "map_set_count": map_set_count,
                "diffusion_map_set": map_set_count,
                "diffusion_map_index": map_set_count - 1,
                "gamma_s": gamma_s,
            }
            manifest["outputs"].update(
                {
                    "average_components": average_components_path.name,
                    "average_components_sha256": _sha256(average_components_path),
                    "average_components_shape": [map_set_count, *data_shape],
                }
            )
        if num_sense_maps is not None:
            manifest["sampling"]["sense_maps"] = {
                "source_map_set_count": source_map_set_count,
                "selected_map_set_count": map_set_count,
                "selection": "last",
            }
        manifest_path = output_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return manifest


__all__ = ["CineDiffReconstructor"]


ROOT = Path(__file__).resolve().parent


def _device_list(values: Sequence[str]) -> list[int]:
    """Parse CUDA indices given as '0,1', '0 1', or a mix of both."""
    devices = []
    for value in values:
        for part in value.split(","):
            if not part.strip():
                continue
            if not part.strip().isdigit():
                raise ValueError(f"invalid CUDA device index: {part!r}")
            devices.append(int(part))
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("--devices needs one or more distinct CUDA indices")
    return devices


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", required=True, help="Case key from data/cases.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--devices",
        nargs="+",
        metavar="INDEX",
        help="CUDA devices to use, e.g. 0,1 or 0 1 (default: all visible GPUs)",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=1,
        help="Number of independent reconstructions to average (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the case settings and exit without reconstructing",
    )
    args = parser.parse_args(argv)
    if args.avg < 1:
        parser.error("--avg must be positive")
    devices = None
    if args.devices is not None:
        try:
            devices = _device_list(args.devices)
        except ValueError as error:
            parser.error(str(error))
    case_files = [ROOT / "data" / "cases.json", ROOT / "example_data" / "cases.json"]
    cases = [
        case
        for path in case_files
        if path.is_file()
        for case in json.loads(path.read_text())
    ]
    matches = [case for case in cases if case["case_key"] == args.case]
    if len(matches) != 1:
        if not case_files[0].is_file():
            parser.error("unknown case; run `python download.py` to get data/")
        parser.error("unknown case; see data/cases.json and example_data/cases.json")
    case = matches[0]
    prepared = (ROOT / case["prepared"]).resolve(strict=True)
    output = args.output_root.expanduser().resolve() / case["case_key"]
    protected = [(ROOT / name).resolve() for name in ("data", "example_data", "weights")]
    if any(output == path or path in output.parents for path in protected):
        parser.error("Output must be outside data/, example_data/, and weights/")
    settings = dict(
        prepared_directory=prepared,
        output_directory=output,
        dataset_name=case["dataset"],
        slice_number=case["slice"],
        gamma_w=case["gamma_w"],
        gamma_s=case["gamma_s"],
        nfe=case["nfe"],
        first_seed=case["seed"],
        num_samples=args.avg,
        eta=case["eta"],
        num_sense_maps=case["num_sense_maps"],
        temporal_boundary_mode=case["temporal_boundary_mode"],
    )
    print(json.dumps(settings, default=str, indent=2))
    print("CUDA devices:", "all visible" if devices is None else devices)
    if args.dry_run:
        return 0
    CineDiffReconstructor(devices=devices).run(**settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
