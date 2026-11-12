import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xarray as xr

from downscale.baselines import score_cells, store_path
from downscale.config import ROOT, load_config
from downscale.data import region_list
from downscale.features import time_feats
from downscale.prepare import open_store, write_store
from downscale.splits import split_masks
from downscale.unet import Unet, pad_to


def load_model(name, dev):
    ck = torch.load(ROOT / "checkpoints" / f"{name}.pt", map_location=dev, weights_only=False)
    model = Unet(len(ck["channels"])).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["channels"], ck["norm"]


def stack(ds, names, norm, sel, utc_offset):
    tf = time_feats(ds.time.values[sel], utc_offset)
    tc = ds.t2m_coarse.isel(time=sel).values.astype("float32")
    nt, ny, nx = tc.shape
    x = np.empty((nt, len(names), ny, nx), "float32")
    for k, c in enumerate(names):
        mu, sd = norm[c]
        if c == "t2m_coarse":
            v = tc
        elif c in tf:
            v = tf[c][:, None, None] * np.ones((1, ny, nx), "float32")
        else:
            v = ds[c].values.astype("float32")[None]
        x[:, k] = np.nan_to_num((v - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    return x, tc


def run_model(model, ds, names, norm, sel, utc_offset, dev, batch):
    out = []
    for s in np.array_split(sel, max(1, len(sel) // batch)):
        x, tc = stack(ds, names, norm, s, utc_offset)
        t = torch.from_numpy(x).to(dev)
        t, h, w = pad_to(t, 8)
        with torch.no_grad():
            p = model(t)[:, 0, :h, :w].cpu().numpy()
        out.append(p + tc)
    return np.concatenate(out)


def score_model(region, model, names, norm, split, dev, batch):
    cfg = load_config()
    ds = open_store(store_path(region))
    tmask, cmask = split_masks(region)[split]
    sel = np.where(tmask)[0]
    if sel.size == 0 or cmask.sum() == 0:
        ds.close()
        return 0.0, 0
    off = cfg["regions"][region]["utc_offset"]
    pred = run_model(model, ds, names, norm, sel, off, dev, batch)
    tl = ds.t2m_land.isel(time=sel).values.astype("float32")
    ds.close()
    d = (pred - tl)[:, cmask]
    good = np.isfinite(d)
    return float(np.sum(d[good] ** 2)), int(good.sum())


def predict_region(region, model, names, norm, label, dev, batch):
    cfg = load_config()
    ds = open_store(store_path(region))
    idx = pd.DatetimeIndex(ds.time.values)
    te = np.where(np.asarray(idx.year) == cfg["years"]["test"])[0]
    land = ds.land.values > 0.5
    off = cfg["regions"][region]["utc_offset"]
    pred = run_model(model, ds, names, norm, te, off, dev, batch)
    pred[:, ~land] = np.nan
    coords = {"time": ds.time.values[te], "lat": ds.lat.values, "lon": ds.lon.values}
    ds.close()
    da = xr.DataArray(pred.astype("float32"), coords=coords, dims=("time", "lat", "lon"), name="pred")
    out_dir = ROOT / cfg["paths"]["outputs"] / region
    out_dir.mkdir(parents=True, exist_ok=True)
    write_store(da.to_dataset().chunk({"time": 24 * 31}), out_dir / f"pred_{label}.zarr")
    print(f"{region} wrote pred_{label}.zarr")


def rmse_of(pair):
    sse, n = pair
    return float(np.sqrt(sse / n)) if n else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="unet_main")
    p.add_argument("--batch", type=int, default=24)
    a = p.parse_args()
    cfg = load_config()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, names, norm = load_model(a.name, dev)
    regions = region_list()

    for region in regions:
        predict_region(region, model, names, norm, a.name, dev, a.batch)

    booster = lgb.Booster(model_file=str(ROOT / cfg["paths"]["outputs"] / "lgbm.txt"))
    lnames = booster.feature_name()
    for region in regions:
        print(f"--- {region} ---")
        for split in ("val", "test_blocks"):
            b = rmse_of(score_cells(region, None, [], split))
            g = rmse_of(score_cells(region, booster, lnames, split))
            u = rmse_of(score_model(region, model, names, norm, split, dev, a.batch))
            print(f"{split:12s} bilinear {b:.4f} c   lgbm {g:.4f} c   {a.name} {u:.4f} c")


if __name__ == "__main__":
    main()
