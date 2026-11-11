# Data format

`python download.py` creates `data/`. Each `data/cases.json` entry points to a
`prepared/` directory:

| File | Contents |
|---|---|
| `kspace.npz` | Acquired k-space in compact form (see below) |
| `max_mag.npy` | Original positive scalar normalization |
| `cs_S.mat` | MATLAB variable `S`, shape `(H, W, 1, coils)` for bundled single-map cases |
| `CS_recon.mat` | MATLAB variable `image_data`, shape `(H, W, 1, 1, T)` |
| `CS_recon_full.mat` | CS reconstruction of the fully sampled data, used as the reference; same layout as `CS_recon.mat`; absent for US cases |

`kspace.npz` contains:

| Array | Contents |
|---|---|
| `mask` | Boolean acquired locations, shape `(H, W, T)`, identical for every coil |
| `samples` | Complex64 acquired values in stored measurement units, ordered row-major over `(H, W, coil, T)` |
| `coils` | Number of coils |
| `data_shape` | Three integers `(H, W, T)` |
| `negative_zero` | Optional packed bits marking zeros of `Y` stored as -0.0 |

`utilities.data_consistency.prepared_kspace.load_prepared_kspace` expands it
into `Y`, the k-space of shape `(coils, H, W, T)` that is zero where not
acquired, and `samp_4d_idx`, the acquired indices of shape `(4, samples)` with
axes `(H, W, coil, T)`.

Keep native dtypes, FFT convention, geometry, coil maps, and scalar normalization.
The existing loader normalizes CS/reference images; the MRI operator uses the
stored scalar for measurement consistency. Do not normalize the files again.

US `prepared/` k-space contains only the lines used for reconstruction; a
held-out subset of the acquired lines was set aside before preparation and is
not distributed here. `data/cases.json` supplies the portable runtime paths.
The repository contains reconstruction-ready inputs rather than original
scanner files or the raw-data preparation pipeline.
