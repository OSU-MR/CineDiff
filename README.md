[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OSU-MR/CineDiff/blob/main/CineDiff_demo.ipynb)


Patch-Based Diffusion Reconstruction for Accelerated Cardiac Cine MRI \[[arXiv](https://arxiv.org/abs/2608.28927)\]
===

This repo contains the official implementation for the paper [Patch-Based Diffusion Reconstruction for Accelerated Cardiac Cine MRI](https://arxiv.org/abs/2608.28927).

By: Xuan Lei, Philip Schniter, Juliet Varghese, and Rizwan Ahmad

## Abstract

We propose CineDiff, a diffusion-based reconstruction framework for highly
accelerated 2D real-time cine cardiovascular MRI, which combines an
unconditional patch-based diffusion model with diffusion posterior sampling for
data consistency. On retrospectively undersampled breath-held cine, CineDiff
achieved higher PSNR and SSIM and lower LPIPS and DISTS than the comparison
methods at all evaluated acceleration rates, and on prospectively undersampled
free-breathing real-time cine it received higher expert image-quality scores.
Compared with compressed sensing and the variational network CineVN, it reduced
block-like artifacts and preserved finer anatomical detail, and it remained
robust on out-of-distribution mid-field (0.55T) and porcine data.

## Requirements

- An NVIDIA GPU with CUDA 12.1. Large cases need substantial GPU memory.
- [Conda](https://docs.conda.io) for the environment.

## Installation

This repository contains the reconstruction code. The prepared OCMR cases and
the pretrained model weights are downloaded separately from the
[data&weights release](https://github.com/OSU-MR/CineDiff/releases/tag/data%26weights)
by `download.py`:

```bash
git clone https://github.com/OSU-MR/CineDiff.git
cd CineDiff
conda env create -f environment.yml
conda activate cinediff-reconstruction
python download.py
```

`download.py` downloads the data (1.1 GB) into `data/` and the model weights
(286 MB) into `weights/`, checking each download's SHA256. Running it again
skips whatever is already present; use `--data-only` or `--weights-only` to
download one of them. To replace damaged data, delete `data/` and run it again.

The environment uses Python 3.12, PyTorch 2.3.1, and CUDA 12.1.
LPIPS and DISTS download their own feature-network weights on first use.

## Data and model weights

| Path | Contents |
|---|---|
| `data/fs/` | 14 preprocessed fully sampled OCMR slices, each retrospectively undersampled at R8, R12, R16, and R20 (56 cases) |
| `data/us/` | 15 preprocessed prospectively undersampled OCMR cases |
| `data/cases.json` | Case identifiers, paths, image shapes, and reconstruction settings |
| `weights/model-192000.pt` | Pretrained CineDiff model weights |

The cases are prepared from the [OCMR dataset](https://arxiv.org/abs/2008.03410).
All cases are already prepared, so no calibration, mask generation, resizing, or normalization is
needed. See `DATA_FORMAT.md` for the file layout.

## Quick start

Check a case and print its settings without running inference:

```bash
python reconstruct.py --case fs_0008_1_5T_sli0_R16 \
  --output-root outputs/example --dry-run
```

Reconstruct a fully sampled (FS) case and an undersampled (US) case:

```bash
python reconstruct.py --case fs_0008_1_5T_sli0_R16 \
  --output-root outputs/example
```

```bash
python reconstruct.py --case ocmr_0004_sli0 \
  --output-root outputs/example
```

For smaller GPUs, `example_data/` contains two cropped copies of
`fs_0008_1_5T_sli0_R16`: `fs_0008_1_5T_sli0_R16_8frames` uses about 14 GB of
GPU memory and fits on 24 GB GPUs, and `fs_0008_1_5T_sli0_R16_15frames` uses
about 26 GB (see `example_data/README.md`). They are included in the
repository, so only the model weights need to be downloaded:

```bash
python download.py --weights-only
python reconstruct.py --case fs_0008_1_5T_sli0_R16_8frames \
  --output-root outputs/example
```

Case keys are listed in `data/cases.json`. Each case runs with its stored
settings, with 50 sampling steps for FS
cases and 200 for US cases. Sampling starts from the compressed-sensing
reconstruction, noised to the first step of the shortened schedule.
To use a different number of sampling steps, pass `--nfe` (<= 1000). For
example:

```bash
python reconstruct.py --case fs_0008_1_5T_sli0_R16_8frames \
  --output-root outputs/nfe100 --nfe 100
```

The results in the paper are posterior means over 8 samples. Add `--avg 8` to
run eight independent reconstructions one after another and average them.
Runs take about eight times as long. For example, with the 8-frame example:

```bash
python reconstruct.py --case fs_0008_1_5T_sli0_R16_8frames \
  --output-root outputs/avg8 --avg 8
```

By default every visible GPU is used: the denoiser's patch batch is split
across them. Pass `--devices` to choose GPUs, for example `--devices 0` for a
single GPU or `--devices 0,1` for two; the first one listed holds the model
and the outputs.

The runner prints its settings before inference and writes reconstructed
tensors, complex image arrays, a `manifest.json`, and a comparison GIF under the
output directory.

FS cases are evaluated against their fully sampled reference with SNR, PSNR,
SSIM, LPIPS, and DISTS over the full image. US cases have no fully sampled
reference, so these image metrics do not apply to them.

## Code structure

| Path | Purpose |
|---|---|
| `reconstruct.py` | Command line, inference, evaluation, and output writing for one case from `data/cases.json` |
| `download.py` | Download and check the data and model weights |
| `utilities/models/video_diffusion_model/` | Network and sampler |
| `utilities/data_consistency/` | MRI forward model, patching, and data loading |

The sampler is `GaussianDiffusion.ddim_sample` in
`utilities/models/video_diffusion_model/video_diffusion_pytorch.py`. The
denoiser call, measurement residual, guidance gradient, and update
are written out in a single loop. 

## Questions and Concerns

If you have any questions, or run into any issues, please
[open an issue](https://github.com/OSU-MR/CineDiff/issues).

## Citation

If you find this code helpful, please cite our paper:

```bibtex
@article{lei2026patch,
  title={Patch-Based Diffusion Reconstruction for Accelerated Cardiac Cine},
  author={Lei, Xuan and Schniter, Philip and Varghese, Juliet and Ahmad, Rizwan},
  journal={arXiv preprint arXiv:2608.28927},
  year={2026}
}
```

## Acknowledgements

Our network architecture is based on
[video-diffusion-pytorch](https://github.com/lucidrains/video-diffusion-pytorch) and [EDM](https://github.com/nvlabs/edm).
We thank the [OCMR](https://arxiv.org/abs/2008.03410) and [CMRxRecon](https://github.com/CmrxRecon/CMRxRecon) teams for making their data
openly available.
