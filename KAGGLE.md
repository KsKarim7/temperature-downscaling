# Final DDP training on Kaggle

Local training on the GTX 1650 is single GPU and fp32, because GTX 16-series cards are unreliable in fp16.
Kaggle gives two T4s, which handle fp16, so the final run goes there with DDP and AMP on.
Use the sweep result from `reports/sweep.csv` for lr, loss and drop. The commands below assume the full
feature set and l1 loss at 1e-3; change them to match the sweep.

## 1. Build the upload

Locally, zip these three things into one archive:

```
data/processed/uae.zarr
data/processed/bangladesh.zarr
outputs/uae/splits.npz
outputs/bangladesh/splits.npz
outputs/norm.json
```

From PowerShell in the project root:

```powershell
Compress-Archive -Path data\processed\uae.zarr, data\processed\bangladesh.zarr -DestinationPath upload\processed.zip
Compress-Archive -Path outputs\uae\splits.npz, outputs\bangladesh\splits.npz, outputs\norm.json -DestinationPath upload\splits.zip
```

The two zarr stores are the bulk, roughly 1 GB together. The splits and norm file are tiny but they must
come from this machine, not be regenerated on Kaggle, or the held-out blocks will differ and the test
numbers stop being comparable.

Upload both zips as one **private** Kaggle dataset. Call it `temperature-downscaling-data`.

## 2. Notebook setup

Create a new notebook. In the settings panel set accelerator to **GPU T4 x2** and attach the dataset.
Internet must be on to clone the repository and install packages.

## 3. Code and data

```python
!git clone https://github.com/KsKarim7/temperature-downscaling /kaggle/working/td
%cd /kaggle/working/td
```

Check what the image already has before installing anything:

```python
!pip list 2>/dev/null | grep -iE "^(torch|xarray|zarr|numpy|pandas|dask|netCDF4|rasterio|rioxarray|lightgbm|pyyaml) "
```

Kaggle images normally ship torch, numpy, pandas, xarray, dask, netCDF4 and pyyaml. zarr is usually
present but at version 2, which this project cannot use, and rioxarray is often missing. Install only
what the check above shows as missing or too old:

```python
!pip install -q "zarr>=3" rioxarray pygeohash meteostat
```

Do not reinstall torch. The preinstalled build matches the Kaggle CUDA drivers.

Unpack the data into the paths the code expects:

```python
import os, zipfile
os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
src = "/kaggle/input/temperature-downscaling-data"
zipfile.ZipFile(f"{src}/processed.zip").extractall("data/processed")
zipfile.ZipFile(f"{src}/splits.zip").extractall("outputs")
```

The splits zip holds `uae/splits.npz`, `bangladesh/splits.npz` and `norm.json`, so after extraction they
land at `outputs/uae/splits.npz`, `outputs/bangladesh/splits.npz` and `outputs/norm.json`. Confirm:

```python
!ls data/processed && ls outputs && ls outputs/uae outputs/bangladesh
```

Then confirm the stores open and the splits match what was uploaded:

```python
!python -m downscale.checks --region all
```

## 4. Train with DDP on both T4s

```python
!torchrun --nproc_per_node=2 -m downscale.train \
    --epochs 30 --lr 1e-3 --loss l1 --drop "" --batch 64 --name unet_main
```

AMP is on by default, so there is no `--no-amp` here. That is the one deliberate difference from the
local runs.

`--drop ""` means keep every feature. If the sweep picked a drop group, pass it instead, for example
`--drop position`.

## 5. Confirm DDP really used both GPUs

At startup each rank prints its own line. Two ranks means two lines:

```
rank 0 of 2 on cuda:0 Tesla T4
rank 1 of 2 on cuda:1 Tesla T4
```

One line, or `rank 0 of 1`, means torchrun did not spawn two processes and the run is single GPU.
Stop and fix it before spending 30 epochs.

While training runs, open a second cell and check both cards are busy:

```python
!nvidia-smi
```

Both GPUs should show non-zero memory and utilisation, with a python process on each. A card sitting at
0 MiB means only one rank is doing work.

Only rank 0 prints the per-epoch line, validates and writes the checkpoint. That is expected.

## 6. Predict

```python
!python -m downscale.predict --name unet_main
```

This writes `outputs/<region>/pred_unet_main.zarr` for both regions and prints val and test_blocks RMSE
for bilinear, lgbm and the U-Net. The lgbm comparison needs `outputs/lgbm.txt`, which is in the
repository, so it works without extra uploads.

## 7. Bring the results back

Zip the prediction stores, since they are directories:

```python
!cd outputs/uae && zip -qr /kaggle/working/pred_uae.zip pred_unet_main.zarr
!cd outputs/bangladesh && zip -qr /kaggle/working/pred_bangladesh.zip pred_unet_main.zarr
!cp checkpoints/unet_main.pt outputs/unet_main_history.csv /kaggle/working/
```

Download from the notebook output panel:

- `unet_main.pt`
- `unet_main_history.csv`
- `pred_uae.zip`
- `pred_bangladesh.zip`

Put them back locally at exactly these paths:

```
checkpoints/unet_main.pt
outputs/unet_main_history.csv
outputs/uae/pred_unet_main.zarr
outputs/bangladesh/pred_unet_main.zarr
```

Unzip each prediction archive inside the matching region folder so the `.zarr` directory sits directly
under `outputs/<region>/`. Then re-run the comparison locally to confirm the numbers survived the trip:

```powershell
.\.venv\Scripts\python -m downscale.predict --name unet_main
```

It will recompute the predictions from the checkpoint rather than read the zarr, so the printed RMSE
values should match the Kaggle ones to about four decimals. A larger difference points at a checkpoint
or norm mismatch.

## Notes

Kaggle sessions stop after 12 hours and shorter if idle, and the GPU quota is weekly. Thirty epochs at
this size take well under that, but save the checkpoint as soon as it exists rather than at the end.

Keep the dataset private. It is derived from ERA5 and Copernicus DEM, which are redistributable under
their own licences, but there is no reason to republish them.
