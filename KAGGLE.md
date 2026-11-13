# Final DDP training on Kaggle

Kaggle trains and nothing else. Prediction and scoring run locally afterwards, on the same machine that
built the splits, so the test numbers come from one place. The notebook produces exactly two files: the
checkpoint and the history CSV.

Local training on the GTX 1650 is single GPU and fp32, because GTX 16-series cards are unreliable in
fp16. Kaggle gives two T4s, which handle fp16, so the final run uses DDP with AMP on.

The configuration below is the sweep winner in `reports/sweep.csv`: lr 1e-3, l1 loss, drop position,
val rmse 0.7212 C. Dropping position was better than keeping it, so `--drop position` is not a typo.

This ran on 2026-11-13. Forty epochs took 14 s per epoch on two T4s with AMP, against about 200 s per
epoch on the GTX 1650 in fp32, so roughly 9 minutes of training instead of a little over two hours.
Results are in `reports/kaggle_runs.csv` and the full console output in `reports/kaggle_train_log.txt`.

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
!pip install -q "zarr>=3" rioxarray cdsapi pyyaml requests
```

```python
import torch, zarr
print(torch.__version__, zarr.__version__, torch.cuda.device_count())
```

```python
!python -c "import downscale.train; print('imports ok')"
```

Do not reinstall torch. The preinstalled build matches the Kaggle drivers. The run on 2026-11-13
printed `2.11.0+cu130 3.4.2 2`, so torch sees both cards and zarr is on version 3.

cdsapi is needed because the import chain reaches it: `train` imports `data`, which imports `prepare`
for `open_store`, and `prepare` imports `worldcover_urls` from `download`, which imports cdsapi at module
load. None of it is used for training, but the import fails without it. The `imports ok` check catches
that in two seconds instead of after the data cell. lightgbm and pygeohash are not needed, since Kaggle
only trains.

### Cell 4: data

Kaggle sometimes auto-extracts an uploaded zip and sometimes leaves it alone, so handle both:

```python
import shutil, zipfile
from pathlib import Path

src = Path("/kaggle/input")
zips = list(src.rglob("kaggle_upload.zip"))
if zips:
    zipfile.ZipFile(zips[0]).extractall(".")
else:
    root = next(src.rglob("uae.zarr")).parents[2]
    for sub in ("data", "outputs"):
        shutil.copytree(root / sub, sub, dirs_exist_ok=True)
```

The rglob search means the dataset slug does not have to match, and it finds the files whether Kaggle
left the zip alone or auto-extracted it. Either way the files land at the paths the code expects.
Confirm:

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
import zipfile
from pathlib import Path

dst = "/kaggle/working/unet_main_results.zip"
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    for p in ("checkpoints/unet_main.pt", "outputs/unet_main_history.csv"):
        z.write(p, p)
print(Path(dst).stat().st_size / 2**20, "MiB")
```

The archive keeps the `checkpoints/` and `outputs/` prefixes, so unzipping it in the project root puts
both files where they belong. It came to 8.3 MiB.

Download `unet_main_results.zip` from the notebook output panel.

## 3. What DDP changes and what it does not

Each rank prints its rank, world size and GPU at startup, before any training:

```
rank 0 of 2 on cuda:0 Tesla T4
rank 1 of 2 on cuda:1 Tesla T4
```

Two lines means two processes. A single line, or `rank 0 of 1`, means torchrun did not spawn both ranks
and you are about to spend 40 epochs on one GPU. Stop and fix it. The two ranks write at the same
moment, so in the Kaggle log the lines can arrive spliced onto one line, as they did on 2026-11-13:
`rank 0 of 2 on cuda:0 Tesla T4rank 1 of 2 on cuda:1 Tesla T4`. Both ranks are there; only the newline
is missing.

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

Kaggle sessions stop after 12 hours, sooner if idle, and the GPU quota is weekly. Forty epochs took
about 9 minutes of training and 11 minutes of session time including the clone, the installs and
loading both zarr stores, so the quota is not a constraint at this size. The checkpoint is written
whenever validation improves rather than at the end, so an interrupted session still leaves the best
model so far.

The first epoch took 21 s and the rest 14 s. Both runs in the session landed within 0.004 C of each
other, which sets the floor on what counts as a real difference between configurations.

Keep the dataset private. ERA5 and Copernicus DEM are redistributable under their own licences, but
there is no reason to republish them.
