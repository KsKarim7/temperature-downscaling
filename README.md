# Temperature Downscaling

Work in progress.

## Install

```
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

torch is not in requirements.txt. It is installed separately from the PyTorch CUDA index so the build matches the local GPU.

## Downloading

Run only one download process at a time. The Climate Data Store limits how many requests one account may have queued for a dataset, and a second process pushes past that limit and gets every request rejected. The downloader sends one request, waits for it to finish, and only then sends the next. If it hits the queue limit it waits 10 minutes and retries the same request rather than moving on.

ERA5 is requested one year at a time and ERA5-Land one quarter at a time, then split into monthly files. A period that the store rejects as too large falls back to monthly requests.
