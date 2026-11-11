# Small examples for smaller GPUs

Both cases are `fs_0008_1_5T_sli0_R16` cropped along the fully sampled readout
direction from 172 to 112 pixels (rows 46 to 157, which keeps the heart) and
shortened to their first frames. They are stored in Git, so they can be run
without downloading the data once the model weights are present.

| Case | Size | Patches | GPU memory |
|---|---|---|---|
| `fs_0008_1_5T_sli0_R16_8frames` | 112 x 180 x 8 | 6 | about 14 GB |
| `fs_0008_1_5T_sli0_R16_15frames` | 112 x 180 x 15 | 12 | about 26 GB |

For comparison, the full case (172 x 180 x 21) uses 27 patches. On a GPU with
24 GB of memory, such as an RTX 3090 or 4090, run the 8-frame case:

    python download.py --weights-only
    python reconstruct.py --case fs_0008_1_5T_sli0_R16_8frames --output-root outputs/example

The k-space was cropped by an inverse Fourier transform along the readout,
keeping those rows, and a forward transform back, using the same centred,
orthonormal transform as the reconstruction; each frame is measured
independently, so the later frames were simply dropped. The coil maps, CS
reconstruction, and reference were cropped to the same rows and frames;
`max_mag.npy` is unchanged.
