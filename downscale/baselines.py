import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr

from downscale.config import ROOT, load_config
from downscale.features import channels, static_feats, time_feats
from downscale.prepare import open_store, write_store
from downscale.splits import make_splits, split_masks

PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}


def store_path(region):
    cfg = load_config()
    return ROOT / cfg["paths"]["processed"] / f"{region}.zarr"


def has_store(region):
    return store_path(region).exists()


def rmse(a, b):
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def draw_rows(region, split, n_max, seed, names):
    cfg = load_config()
    ds = open_store(store_path(region))
    tmask, cmask = split_masks(region)[split]
    ti = np.where(tmask)[0]
    ii, jj = np.where(cmask)
    if ti.size == 0 or ii.size == 0:
        ds.close()
        return np.zeros((0, len(names)), "float32"), np.zeros(0, "float32")

    tc = ds.t2m_coarse.isel(time=ti).values
    tl = ds.t2m_land.isel(time=ti).values
    st = static_feats(ds, names)
    tf = time_feats(ds.time.values[ti], cfg["regions"][region]["utc_offset"])
    ds.close()

    total = ti.size * ii.size
    rng = np.random.default_rng(seed)
    if total > n_max:
        flat = rng.choice(total, size=n_max, replace=False)
        a, b = np.divmod(flat, ii.size)
        del flat
    else:
        a, b = np.divmod(np.arange(total), ii.size)

    y = (tl[a, ii[b], jj[b]] - tc[a, ii[b], jj[b]]).astype("float32")
    x = np.empty((a.size, len(names)), "float32")
    for k, c in enumerate(names):
        if c == "t2m_coarse":
            x[:, k] = tc[a, ii[b], jj[b]]
        elif c in tf:
            x[:, k] = tf[c][a]
        else:
            x[:, k] = st[c][ii[b], jj[b]]
    good = np.isfinite(y)
    return x[good], y[good]


def shares(avail, n_max):
    total = sum(avail)
    if total == 0:
        return [0] * len(avail)
    return [int(round(n_max * a / total)) for a in avail]


def pair_counts(regions, split):
    out = []
    for region in regions:
        tmask, cmask = split_masks(region)[split]
        out.append(int(tmask.sum()) * int(cmask.sum()))
    return out


def pool_rows(regions, split, n_max, seed, names):
    avail = pair_counts(regions, split)
    xs, ys = [], []
    for region, per in zip(regions, shares(avail, n_max)):
        if per == 0:
            continue
        x, y = draw_rows(region, split, per, seed, names)
        if y.size:
            xs.append(x)
            ys.append(y)
    if not xs:
        return np.zeros((0, len(names)), "float32"), np.zeros(0, "float32")
    return np.concatenate(xs), np.concatenate(ys)


def train_lgbm(regions, names, seed):
    xtr, ytr = pool_rows(regions, "train", 3_000_000, seed, names)
    xva, yva = pool_rows(regions, "val", 300_000, seed + 1, names)
    print(f"lgbm train rows {ytr.size} val rows {yva.size}")
    params = dict(PARAMS)
    params["seed"] = seed
    params["bagging_seed"] = seed
    params["feature_fraction_seed"] = seed
    dtr = lgb.Dataset(xtr, ytr, feature_name=names)
    dva = lgb.Dataset(xva, yva, feature_name=names, reference=dtr)
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=3000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)],
    )
    cfg = load_config()
    out = ROOT / cfg["paths"]["outputs"]
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "lgbm.txt"))
    gain = booster.feature_importance("gain")
    imp = pd.DataFrame({"feature": names, "gain": gain}).sort_values("gain", ascending=False)
    rep = ROOT / cfg["paths"]["reports"]
    rep.mkdir(parents=True, exist_ok=True)
    imp.to_csv(rep / "lgbm_importance.csv", index=False)
    val_lgbm = rmse(booster.predict(xva, num_iteration=booster.best_iteration), yva)
    val_bilinear = rmse(np.zeros_like(yva), yva)
    print(f"val rmse bilinear {val_bilinear:.4f} lgbm {val_lgbm:.4f}")
    return booster, val_bilinear, val_lgbm


def val_scores(regions, booster, names, seed):
    out = {}
    for region, per in zip(regions, shares(pair_counts(regions, "val"), 300_000)):
        if per == 0:
            continue
        x, y = draw_rows(region, "val", per, seed + 1, names)
        pred = booster.predict(x, num_iteration=booster.best_iteration)
        out[region] = (rmse(np.zeros_like(y), y), rmse(pred, y))
    return out


def predict_region(region, booster, names):
    cfg = load_config()
    ds = open_store(store_path(region))
    idx = pd.DatetimeIndex(ds.time.values)
    te = np.asarray(idx.year) == cfg["years"]["test"]
    if not te.any():
        print(f"{region} has no test year data, skipping prediction")
        ds.close()
        return
    land = ds.land.values > 0.5
    ii, jj = np.where(land)
    st = static_feats(ds, names)
    out_dir = ROOT / cfg["paths"]["outputs"] / region
    out_dir.mkdir(parents=True, exist_ok=True)
    key_of = np.asarray(idx.strftime("%Y-%m"))

    bl, lg = [], []
    for key in sorted(set(key_of[te])):
        sel = np.where(te & (key_of == key))[0]
        tc = ds.t2m_coarse.isel(time=sel).values
        tf = time_feats(ds.time.values[sel], cfg["regions"][region]["utc_offset"])
        nt = sel.size
        x = np.empty((nt * ii.size, len(names)), "float32")
        for k, c in enumerate(names):
            if c == "t2m_coarse":
                x[:, k] = tc[:, ii, jj].ravel()
            elif c in tf:
                x[:, k] = np.repeat(tf[c], ii.size)
            else:
                x[:, k] = np.tile(st[c][ii, jj], nt)
        resid = booster.predict(x, num_iteration=booster.best_iteration).astype("float32")
        a = np.full(tc.shape, np.nan, "float32")
        a[:, ii, jj] = resid.reshape(nt, ii.size) + tc[:, ii, jj]
        b = np.full(tc.shape, np.nan, "float32")
        b[:, ii, jj] = tc[:, ii, jj]
        coords = {"time": ds.time.values[sel], "lat": ds.lat.values, "lon": ds.lon.values}
        lg.append(xr.DataArray(a, coords=coords, dims=("time", "lat", "lon"), name="pred"))
        bl.append(xr.DataArray(b, coords=coords, dims=("time", "lat", "lon"), name="pred"))
        print(f"{region} predict {key} done")
    ds.close()
    for name, parts in (("lgbm", lg), ("bilinear", bl)):
        da = xr.concat(parts, dim="time").astype("float32")
        write_store(da.to_dataset().chunk({"time": 24 * 31}), out_dir / f"pred_{name}.zarr")


def score_region(region, name, split):
    cfg = load_config()
    path = ROOT / cfg["paths"]["outputs"] / region / f"pred_{name}.zarr"
    if not path.exists():
        return None
    ds = open_store(store_path(region))
    pr = open_store(path)
    tmask, cmask = split_masks(region)[split]
    ti = np.where(tmask)[0]
    want = pd.DatetimeIndex(ds.time.values[ti])
    sel = np.where(np.isin(pd.DatetimeIndex(pr.time.values), want))[0]
    truth = ds.t2m_land.isel(time=ti).values
    pred = pr.pred.isel(time=sel).values
    m = np.broadcast_to(cmask, truth.shape)
    out = rmse(pred[m], truth[m])
    ds.close()
    pr.close()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="all", choices=["uae", "bangladesh", "all"])
    a = p.parse_args()
    cfg = load_config()
    regions = ["uae", "bangladesh"] if a.region == "all" else [a.region]
    regions = [r for r in regions if has_store(r)]
    if not regions:
        print("no processed store found")
        return
    names = channels()
    seed = cfg["seed"]

    for region in regions:
        make_splits(region)

    booster, val_bilinear, val_lgbm = train_lgbm(regions, names, seed)

    for region in regions:
        predict_region(region, booster, names)

    val = val_scores(regions, booster, names, seed)
    for region in regions:
        m = split_masks(region)
        print(f"--- {region} ---")
        for split in ("train", "val", "test_blocks", "all_land"):
            tmask, cmask = m[split]
            print(f"{split:12s} {int(cmask.sum())} cells, {int(tmask.sum())} hours")
        vb, vl = val[region]
        print(f"val rmse          bilinear {vb:.4f} C   lgbm {vl:.4f} C")
        sb = score_region(region, "bilinear", "test_blocks")
        sl = score_region(region, "lgbm", "test_blocks")
        if sb is not None and sl is not None:
            print(f"test_blocks rmse  bilinear {sb:.4f} C   lgbm {sl:.4f} C")


if __name__ == "__main__":
    main()
