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

CONFIGS = [("none", ()), ("position", ("position",)), ("landcover", ("landcover",)), ("terrain", ("terrain",))]


def store_path(region):
    cfg = load_config()
    return ROOT / cfg["paths"]["processed"] / f"{region}.zarr"


def has_store(region):
    return store_path(region).exists()


def rmse(a, b):
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


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


def build_x(names, tc, tf, st, ii, jj, nt):
    x = np.empty((nt * ii.size, len(names)), "float32")
    for k, c in enumerate(names):
        if c == "t2m_coarse":
            x[:, k] = tc.ravel()
        elif c in tf:
            x[:, k] = np.repeat(tf[c], ii.size)
        else:
            x[:, k] = np.tile(st[c][ii, jj], nt)
    return x


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
    print(f"train rows {ytr.size} val rows {yva.size} features {len(names)}")
    params = dict(PARAMS)
    params["seed"] = seed
    params["bagging_seed"] = seed
    params["feature_fraction_seed"] = seed
    dtr = lgb.Dataset(xtr, ytr, feature_name=names)
    dva = lgb.Dataset(xva, yva, feature_name=names, reference=dtr)
    return lgb.train(
        params,
        dtr,
        num_boost_round=3000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(400)],
    )


def score_cells(region, booster, names, split):
    cfg = load_config()
    ds = open_store(store_path(region))
    tmask, cmask = split_masks(region)[split]
    ti = np.where(tmask)[0]
    ii, jj = np.where(cmask)
    if ti.size == 0 or ii.size == 0:
        ds.close()
        return 0.0, 0
    st = static_feats(ds, names) if booster is not None else {}
    sse = 0.0
    n = 0
    for s in np.array_split(ti, max(1, ti.size // 744)):
        tc = ds.t2m_coarse.isel(time=s).values[:, ii, jj]
        tl = ds.t2m_land.isel(time=s).values[:, ii, jj]
        if booster is None:
            d = tc - tl
        else:
            tf = time_feats(ds.time.values[s], cfg["regions"][region]["utc_offset"])
            x = build_x(names, tc, tf, st, ii, jj, s.size)
            p = booster.predict(x, num_iteration=booster.best_iteration).astype("float32")
            d = p.reshape(tc.shape) + tc - tl
        good = np.isfinite(d)
        sse += float(np.sum(d[good] ** 2))
        n += int(good.sum())
    ds.close()
    return sse, n


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
        x = build_x(names, tc[:, ii, jj], tf, st, ii, jj, nt)
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
    seed = cfg["seed"]
    out = ROOT / cfg["paths"]["outputs"]
    rep = ROOT / cfg["paths"]["reports"]
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)

    for region in regions:
        make_splits(region)

    counts = {}
    for region in regions:
        m = split_masks(region)
        counts[region] = {k: (int(c.sum()), int(t.sum())) for k, (t, c) in m.items()}

    base = {}
    for region in regions:
        vs, vn = score_cells(region, None, [], "val")
        ts, tn = score_cells(region, None, [], "test_blocks")
        base[region] = (np.sqrt(vs / vn), np.sqrt(ts / tn))

    rows = []
    fits = {}
    pooled = {}
    for tag, drop in CONFIGS:
        names = channels(drop)
        print(f"fitting drop={tag}")
        booster = train_lgbm(regions, names, seed)
        fits[tag] = (booster, names)
        vsse = vn = 0.0
        for region in regions:
            vs, vc = score_cells(region, booster, names, "val")
            ts, tc = score_cells(region, booster, names, "test_blocks")
            vsse += vs
            vn += vc
            rows.append(
                {
                    "drop": tag,
                    "region": region,
                    "val_rmse": round(float(np.sqrt(vs / vc)), 4),
                    "test_blocks_rmse": round(float(np.sqrt(ts / tc)), 4),
                    "best_iteration": int(booster.best_iteration),
                }
            )
        pooled[tag] = float(np.sqrt(vsse / vn))
        print(f"drop={tag} pooled val rmse {pooled[tag]:.4f}")

    pd.DataFrame(rows).to_csv(rep / "lgbm_ablation.csv", index=False)

    best = min(pooled, key=pooled.get)
    booster, names = fits[best]
    booster.save_model(str(out / "lgbm.txt"))
    (rep / "lgbm_choice.txt").write_text(
        f"chosen drop={best} pooled val rmse {pooled[best]:.4f} C\n", encoding="utf-8"
    )
    imp = pd.DataFrame({"feature": names, "gain": booster.feature_importance("gain")})
    imp = imp.sort_values("gain", ascending=False)
    imp.to_csv(rep / "lgbm_importance.csv", index=False)

    for region in regions:
        predict_region(region, booster, names)

    for region in regions:
        print(f"--- {region} ---")
        for split in ("train", "val", "test_blocks", "all_land"):
            c, t = counts[region][split]
            print(f"{split:12s} {c} cells, {t} hours")
        vb, tb = base[region]
        print(f"bilinear val rmse {vb:.4f} C   test_blocks rmse {tb:.4f} C")

    print("--- ablation ---")
    print(pd.DataFrame(rows).to_string(index=False))
    print("--- pooled val rmse ---")
    for tag, v in pooled.items():
        print(f"drop={tag:10s} {v:.4f} C")
    print(f"chosen drop={best} on pooled val rmse {pooled[best]:.4f} C")
    print("--- top 8 features ---")
    print(imp.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
