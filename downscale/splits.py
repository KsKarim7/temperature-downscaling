import argparse

import numpy as np
import pandas as pd

from downscale.config import ROOT, load_config
from downscale.prepare import open_store


def spatial_blocks(land, block_size, test_fraction, seed):
    ny, nx = land.shape
    cand = []
    for by in range(0, ny, block_size):
        for bx in range(0, nx, block_size):
            if land[by : by + block_size, bx : bx + block_size].mean() >= 0.5:
                cand.append((by, bx))
    mask = np.zeros(land.shape, dtype=bool)
    if not cand:
        return mask
    rng = np.random.default_rng(seed)
    k = int(round(len(cand) * test_fraction))
    for i in rng.choice(len(cand), size=k, replace=False):
        by, bx = cand[i]
        mask[by : by + block_size, bx : bx + block_size] = True
    return mask


def val_days(times, fraction, seed):
    year = load_config()["years"]["train"]
    idx = pd.DatetimeIndex(times)
    day = idx.normalize().values
    in_train = np.asarray(idx.year) == year
    days = np.unique(day[in_train])
    if days.size == 0:
        return np.zeros(idx.size, dtype=bool)
    rng = np.random.default_rng(seed)
    k = int(round(days.size * fraction))
    pick = days[np.sort(rng.choice(days.size, size=k, replace=False))]
    return np.isin(day, pick) & in_train


def make_splits(region):
    cfg = load_config()
    ds = open_store(ROOT / cfg["paths"]["processed"] / f"{region}.zarr")
    land = ds.land.values > 0.5
    times = ds.time.values
    test_mask = spatial_blocks(land, cfg["block_size"], cfg["test_block_fraction"], cfg["seed"])
    val_mask = val_days(times, cfg["val_day_fraction"], cfg["seed"])
    out = ROOT / cfg["paths"]["outputs"] / region
    out.mkdir(parents=True, exist_ok=True)
    path = out / "splits.npz"
    np.savez(path, test_mask=test_mask, val_day=val_mask, times=times)
    ds.close()
    print(f"{region} test blocks {int(test_mask.sum())} cells, val days {int(val_mask.sum()) // 24}")
    return path


def load_splits(region):
    cfg = load_config()
    f = np.load(ROOT / cfg["paths"]["outputs"] / region / "splits.npz")
    return {"test_mask": f["test_mask"], "val_day": f["val_day"], "times": f["times"]}


def split_masks(region):
    cfg = load_config()
    s = load_splits(region)
    ds = open_store(ROOT / cfg["paths"]["processed"] / f"{region}.zarr")
    land = ds.land.values > 0.5
    ds.close()
    idx = pd.DatetimeIndex(s["times"])
    yr = np.asarray(idx.year)
    tr = yr == cfg["years"]["train"]
    te = yr == cfg["years"]["test"]
    val = s["val_day"]
    blk = s["test_mask"]
    return {
        "train": (tr & ~val, land & ~blk),
        "val": (tr & val, land & ~blk),
        "test_blocks": (te, land & blk),
        "all_land": (te, land),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True, choices=["uae", "bangladesh", "all"])
    a = p.parse_args()
    cfg = load_config()
    regions = ["uae", "bangladesh"] if a.region == "all" else [a.region]
    for region in regions:
        if (ROOT / cfg["paths"]["processed"] / f"{region}.zarr").exists():
            make_splits(region)


if __name__ == "__main__":
    main()
