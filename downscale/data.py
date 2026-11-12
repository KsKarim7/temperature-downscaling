import json

import numpy as np
from torch.utils.data import Dataset

from downscale.config import ROOT, load_config
from downscale.features import CHANNELS, TIME_CHANNELS, channels, time_feats
from downscale.prepare import open_store
from downscale.splits import split_masks

STATIC = [c for c in CHANNELS if c != "t2m_coarse" and c not in TIME_CHANNELS]
CACHE = {}


def region_list():
    cfg = load_config()
    out = []
    for r in ("uae", "bangladesh"):
        if (ROOT / cfg["paths"]["processed"] / f"{r}.zarr").exists():
            out.append(r)
    return out


def load_region(region, split):
    key = (region, split)
    if key in CACHE:
        return CACHE[key]
    cfg = load_config()
    ds = open_store(ROOT / cfg["paths"]["processed"] / f"{region}.zarr")
    tmask, cmask = split_masks(region)[split]
    ti = np.where(tmask)[0]
    d = {
        "tc": ds.t2m_coarse.isel(time=ti).values.astype("float32"),
        "tl": ds.t2m_land.isel(time=ti).values.astype("float32"),
        "st": {c: ds[c].values.astype("float32") for c in STATIC},
        "tf": time_feats(ds.time.values[ti], cfg["regions"][region]["utc_offset"]),
        "cells": cmask,
    }
    ds.close()
    CACHE[key] = d
    return d


def compute_norm(data):
    out = {}
    for ch in CHANNELS:
        vals = []
        for d in data:
            cm = d["cells"]
            if ch == "t2m_coarse":
                v = d["tc"][:, cm]
            elif ch in d["tf"]:
                v = d["tf"][ch]
            else:
                v = d["st"][ch][cm]
            vals.append(np.asarray(v).ravel())
        a = np.concatenate(vals)
        a = a[np.isfinite(a)]
        sd = float(a.std())
        out[ch] = [float(a.mean()), sd if sd > 1e-6 else 1.0]
    return out


class Crops(Dataset):
    def __init__(self, split, names, n, seed, crop=32):
        self.split = split
        self.names = names
        self.n = n
        self.seed = seed
        self.crop = crop
        self.regions = region_list()
        self.data = [load_region(r, split) for r in self.regions]
        w = np.array([float(d["cells"].sum()) for d in self.data])
        self.w = w / w.sum()
        self.norm = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + i) if self.seed is not None else np.random.default_rng()
        d = self.data[rng.choice(len(self.data), p=self.w)]
        nt, ny, nx = d["tc"].shape
        c = self.crop
        t = int(rng.integers(nt))
        i0 = j0 = 0
        for _ in range(10):
            i0 = int(rng.integers(ny - c + 1))
            j0 = int(rng.integers(nx - c + 1))
            if d["cells"][i0 : i0 + c, j0 : j0 + c].any():
                break
        cm = d["cells"][i0 : i0 + c, j0 : j0 + c]
        tc = d["tc"][t, i0 : i0 + c, j0 : j0 + c]
        tl = d["tl"][t, i0 : i0 + c, j0 : j0 + c]
        y = tl - tc
        m = cm & np.isfinite(y)
        x = np.empty((len(self.names), c, c), "float32")
        for q, ch in enumerate(self.names):
            if ch == "t2m_coarse":
                v = tc
            elif ch in d["tf"]:
                v = np.full((c, c), d["tf"][ch][t], "float32")
            else:
                v = d["st"][ch][i0 : i0 + c, j0 : j0 + c]
            mu, sd = self.norm[ch]
            x[q] = np.nan_to_num((v - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
        return x, np.nan_to_num(y, nan=0.0)[None].astype("float32"), m[None].astype("float32")


def norm_path():
    cfg = load_config()
    return ROOT / cfg["paths"]["outputs"] / "norm.json"


def build(drop, n_train, n_val, seed):
    names = channels(drop)
    tr = Crops("train", names, n_train, None)
    va = Crops("val", names, n_val, seed)
    path = norm_path()
    if path.exists():
        norm = json.loads(path.read_text(encoding="utf-8"))
    else:
        norm = compute_norm(tr.data)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(norm, indent=1), encoding="utf-8")
    tr.norm = norm
    va.norm = norm
    return tr, va, norm, names
