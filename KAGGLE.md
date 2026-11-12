# Final DDP training on Kaggle

Kaggle trains and nothing else. Prediction and scoring run locally afterwards, on the same machine that
built the splits, so the test numbers come from one place. The notebook produces exactly two files: the
checkpoint and the history CSV.

Local training on the GTX 1650 is single GPU and fp32, because GTX 16-series cards are unreliable in
fp16. Kaggle gives two T4s, which handle fp16, so the final run uses DDP with AMP on.

The configuration below is the sweep winner in `reports/sweep.csv`: lr 1e-3, l1 loss, drop position,
val rmse 0.7212 C. Dropping position was better than keeping it, so `--drop position` is not a typo.

## 1. Build the upload

From the project root:

```powershell
.\.venv\Scripts\python -m scripts.make_kaggle_zip
```

This writes `kaggle_upload.zip` in the project root. It holds, at paths relative to the root:

```
data/processed/uae.zarr/**
data/processed/bangladesh.zarr/**
outputs/uae/splits.npz
outputs/bangladesh/splits.npz
outputs/norm.json
```

The two zarr stores are the bulk. The splits and norm files are small but they must come from this
machine rather than be regenerated on Kaggle, or the held-out blocks and the normalisation statistics
will differ and the local scoring afterwards stops being comparable.

Upload the zip as a **private** Kaggle dataset named `temperature-downscaling-data`, which puts it at
`/kaggle/input/temperature-downscaling-data/`.

## 2. Notebook setup

New notebook, accelerator **GPU T4 x2**, internet on, the dataset attached. Run the cells below in order.

### Cell 1: check the hardware

```python
!nvidia-smi
```

Two Tesla T4 entries. One card means the accelerator is set to T4 x1 and torchrun will not give you DDP.

### Cell 2: code

```python
!git clone https://github.com/KsKarim7/temperature-downscaling.git /kaggle/working/td
%cd /kaggle/working/td
```

### Cell 3: packages

```python
!pip list 2>/dev/null | grep -iE "^(torch|numpy|pandas|xarray|zarr|dask|netCDF4|rioxarray|pyyaml) "
!pip install -q "zarr>=3" rioxarray
```

Do not reinstall torch. The preinstalled build matches the Kaggle drivers.

Kaggle images ship torch, numpy, pandas, xarray, dask and pyyaml. zarr is usually present but at
version 2, which this project cannot read, and rioxarray is usually missing; `downscale.prepare` imports
it at module load and `downscale.data` imports `prepare`, so training fails at import without it.
lightgbm and pygeohash are not needed here, since Kaggle only trains.

### Cell 4: data

Kaggle sometimes auto-extracts an uploaded zip and sometimes leaves it alone, so handle both:

```python
import shutil, zipfile
from pathlib import Path

src = Path("/kaggle/input/temperature-downscaling-data")
z = src / "kaggle_upload.zip"
if z.exists():
    zipfile.ZipFile(z).extractall(".")
else:
    for sub in ("data", "outputs"):
        shutil.copytree(src / sub, sub, dirs_exist_ok=True)
```

Either way the files land at the paths the code expects. Confirm:

```python
!ls data/processed && ls outputs outputs/uae outputs/bangladesh
```

You want `uae.zarr` and `bangladesh.zarr` under `data/processed`, `norm.json` under `outputs`, and a
`splits.npz` in each region folder.

### Cell 5: train

```python
!torchrun --nproc_per_node=2 -m downscale.train \
    --epochs 40 --lr 1e-3 --loss l1 --drop position --batch 64 --name unet_main
```

AMP is on by default, so there is no `--no-amp` here. That is the one deliberate difference from the
local runs.

### Cell 6: results

```python
!cd /kaggle/working/td && zip -qj /kaggle/working/unet_main_results.zip \
    checkpoints/unet_main.pt outputs/unet_main_history.csv
!ls -lh /kaggle/working/unet_main_results.zip
```

Download `unet_main_results.zip` from the notebook output panel.

## 3. What DDP changes and what it does not

Each rank prints its rank, world size and GPU at startup, before any training:

```
rank 0 of 2 on cuda:0 Tesla T4
rank 1 of 2 on cuda:1 Tesla T4
```

Two lines means two processes. A single line, or `rank 0 of 1`, means torchrun did not spawn both ranks
and you are about to spend 40 epochs on one GPU. Stop and fix it.

With `--nproc_per_node=2`:

- A `DistributedSampler` splits the 20,000 train crops per epoch, so each rank draws 10,000. An epoch
  covers the same total number of crops as a single GPU run, in about half the wall time. It is not
  two epochs' worth of data.
- `--batch 64` is per GPU, so the global batch is 128.
- The learning rate is unchanged at 1e-3. Gradients are averaged across ranks rather than summed, so
  the step size stays comparable without rescaling.
- Only rank 0 validates, and it validates on all 4,000 val crops, not half. Those crops are drawn with
  a fixed seed, so the validation set is identical every epoch and identical to the local sweep runs.
- Only rank 0 prints the epoch line, writes the checkpoint and writes the history CSV. The other rank
  prints nothing after its startup line. That is expected, not a hang.

While it trains, a second cell confirms both cards are working:

```python
!nvidia-smi
```

A python process and non-zero utilisation on each card. A card sitting at 0 MiB means one rank is idle.

## 4. Back on the local machine

Unzip into the project root so the files land at:

```
checkpoints/unet_main.pt
outputs/unet_main_history.csv
```

Then predict and score locally:

```powershell
.\.venv\Scripts\python -m downscale.predict --name unet_main
```

This writes `outputs/<region>/pred_unet_main.zarr` for both regions and prints val and test_blocks RMSE
for bilinear, lgbm and the U-Net. The checkpoint carries its own channel list and normalisation, so the
prediction uses the same features the Kaggle run trained on without any extra flags.

## Notes

Kaggle sessions stop after 12 hours, sooner if idle, and the GPU quota is weekly. Forty epochs at this
size finish well inside that. The checkpoint is written whenever validation improves rather than at the
end, so an interrupted session still leaves the best model so far.

Keep the dataset private. ERA5 and Copernicus DEM are redistributable under their own licences, but
there is no reason to republish them.
