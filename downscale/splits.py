import argparse

import numpy as np
import pandas as pd

from downscale.config import ROOT, load_config
from downscale.prepare import open_store


def eligible_blocks(land, block_size):
    ny, nx = land.shape
    out = []
    for by in range(0, ny, block_size):
        for bx in range(0, nx, block_size):
            if land[by : by + block_size, bx : bx + block_size].mean() >= 0.5:
                out.append((by, bx))
    return out


def paint(land, block_size, picks):
    mask = np.zeros(land.shape, dtype=bool)
    for by, bx in picks:
        mask[by : by + block_size, bx : bx + block_size] = True
    return mask


def spatial_blocks(land, block_size, test_fraction, seed):
    cand = eligible_blocks(land, block_size)
    if not cand:
        return np.zeros(land.shape, dtype=bool)
    rng = np.random.default_rng(seed)
    k = int(round(len(cand) * test_fraction))
    return paint(land, block_size, [cand[i] for i in rng.choice(len(cand), size=k, replace=False)])


def val_blocks(land, block_size, val_fraction, seed, test_mask):
    cand = eligible_blocks(land, block_size)
    left = [(by, bx) for by, bx in cand if not test_mask[by, bx]]
    if not left:
        return np.zeros(land.shape, dtype=bool)
    rng = np.random.default_rng(seed)
    k = min(len(left), int(round(len(cand) * val_fraction)))
    return paint(land, block_size, [left[i] for i in rng.choice(len(left), size=k, replace=False)])


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
    ds.close()
    bs = cfg["block_size"]
    test_mask = spatial_blocks(land, bs, cfg["test_block_fraction"], cfg["seed"])
    val_mask = val_blocks(land, bs, cfg["val_block_fraction"], cfg["seed"], test_mask)
    day_mask = val_days(times, cfg["val_day_fraction"], cfg["seed"])
    out = ROOT / cfg["paths"]["outputs"] / region
    out.mkdir(parents=True, exist_ok=True)
    path = out / "splits.npz"
    np.savez(path, test_mask=test_mask, val_block=val_mask, val_day=day_mask, times=times)
    print(
        f"{region} test blocks {int(test_mask.sum())} cells, "
        f"val blocks {int(val_mask.sum())} cells, val days {int(day_mask.sum()) // 24}"
    )
    return path


def load_splits(region):
    cfg = load_config()
    f = np.load(ROOT / cfg["paths"]["outputs"] / region / "splits.npz")
    return {
        "test_mask": f["test_mask"],
        "val_block": f["val_block"],
        "val_day": f["val_day"],
        "times": f["times"],
    }


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
    day = s["val_day"]
    blk = s["test_mask"]
    vblk = s["val_block"]
    return {
        "train": (tr & ~day, land & ~blk & ~vblk),
        "val": (tr & day, land & vblk),
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
