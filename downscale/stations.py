import argparse
import datetime as dt

import lightgbm as lgb
import meteostat
import numpy as np
import pandas as pd
import torch
from meteostat import Parameter

from downscale.baselines import build_x, store_path
from downscale.config import ROOT, load_config
from downscale.data import region_list
from downscale.features import static_feats, time_feats
from downscale.predict import load_model, run_model
from downscale.prepare import open_store

LABEL = {"uae": "UAE region", "bangladesh": "Bangladesh region"}
MIN_COVER = 0.7
MIN_STATIONS = 3


def in_bbox(region):
    b = load_config()["regions"][region]["bbox"]
    north, west, south, east = b
    sql = (
        "select id, country, latitude, longitude, elevation from stations "
        f"where latitude between {south} and {north} "
        f"and longitude between {west} and {east}"
    )
    return meteostat.stations.query(sql)


def station_name(sid):
    m = meteostat.stations.meta(sid)
    return m.name if m is not None and m.name else sid


def station_hours(ids, year):
    ids = list(ids)
    out = []
    for k in range(0, len(ids), 10):
        ts = meteostat.hourly(
            ids[k : k + 10],
            dt.datetime(year, 1, 1),
            dt.datetime(year, 12, 31, 23),
            parameters=[Parameter.TEMP],
        )
        df = ts.fetch()
        if df is not None and not df.empty:
            out.append(df.reset_index().pivot(index="time", columns="station", values="temp"))
    if not out:
        return pd.DataFrame()
    return pd.concat(out, axis=1)


def nearest_cell(ds, lat, lon):
    i = int(np.argmin(np.abs(ds.lat.values - lat)))
    j = int(np.argmin(np.abs(ds.lon.values - lon)))
    return i, j


def cell_preds(region, cells, name, booster, lnames, model, names, norm, dev, batch):
    cfg = load_config()
    ds = open_store(store_path(region))
    idx = pd.DatetimeIndex(ds.time.values)
    te = np.where(np.asarray(idx.year) == cfg["years"]["test"])[0]
    ii = np.array([c[0] for c in cells])
    jj = np.array([c[1] for c in cells])
    off = cfg["regions"][region]["utc_offset"]
    st = static_feats(ds, lnames)
    out = {"bilinear": [], "lgbm": [], name: [], "era5land": []}
    stamps = []
    for sel in np.array_split(te, 12):
        tc = ds.t2m_coarse.isel(time=sel).values[:, ii, jj].astype("float32")
        tl = ds.t2m_land.isel(time=sel).values[:, ii, jj].astype("float32")
        tf = time_feats(ds.time.values[sel], off)
        x = build_x(lnames, tc, tf, st, ii, jj, sel.size)
        p = booster.predict(x, num_iteration=booster.best_iteration).astype("float32")
        full = run_model(model, ds, names, norm, sel, off, dev, batch)
        out["bilinear"].append(tc)
        out["lgbm"].append(p.reshape(tc.shape) + tc)
        out[name].append(full[:, ii, jj])
        out["era5land"].append(tl)
        stamps.append(ds.time.values[sel])
    elev = ds.elev.values[ii, jj]
    ds.close()
    time = pd.DatetimeIndex(np.concatenate(stamps))
    return {k: np.concatenate(v) for k, v in out.items()}, time, elev


def score(pred, obs):
    d = pred - obs
    good = np.isfinite(d)
    if good.sum() == 0:
        return float("nan"), float("nan"), float("nan"), 0
    v = d[good].astype("float64")
    return float(np.sqrt(np.mean(v**2))), float(np.mean(np.abs(v))), float(np.mean(v)), int(good.sum())


def run_region(region, name, booster, lnames, model, names, norm, dev, batch):
    cfg = load_config()
    year = cfg["years"]["test"]
    meta = in_bbox(region)
    print(f"{region} {len(meta)} stations in bbox", flush=True)
    if meta.empty:
        return [], 0

    obs = station_hours(meta.id.to_numpy(), year)
    if obs.empty:
        print(f"{region} no hourly temperature returned")
        return [], 0
    hours = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    obs = obs.reindex(hours)
    cover = obs.notna().sum() / len(hours)

    ds = open_store(store_path(region))
    land = ds.land.values > 0.5
    keep = []
    for _, r in meta.iterrows():
        if r.id not in obs.columns or cover[r.id] < MIN_COVER:
            continue
        i, j = nearest_cell(ds, r.latitude, r.longitude)
        if not land[i, j]:
            continue
        keep.append((r, i, j))
    ds.close()
    print(f"{region} {len(keep)} stations with >={MIN_COVER:.0%} coverage on a land cell", flush=True)
    if len(keep) < MIN_STATIONS:
        print(f"{region} has fewer than {MIN_STATIONS} usable stations, treat its station numbers as anecdotal")
    if not keep:
        return [], 0

    cells = [(i, j) for _, i, j in keep]
    preds, time, elev = cell_preds(region, cells, name, booster, lnames, model, names, norm, dev, batch)
    o = obs.reindex(time)

    rows = []
    for q, (r, i, j) in enumerate(keep):
        y = o[r.id].to_numpy(dtype="float32")
        for method in ("bilinear", "lgbm", name, "era5land"):
            rm, ma, bi, n = score(preds[method][:, q], y)
            rows.append(
                {
                    "region": region,
                    "label": LABEL[region],
                    "method": method,
                    "id": r.id,
                    "name": station_name(r.id),
                    "lat": round(float(r.latitude), 4),
                    "lon": round(float(r.longitude), 4),
                    "station_elev": float(r.elevation) if pd.notna(r.elevation) else float("nan"),
                    "cell_elev": round(float(elev[q]), 1),
                    "elev_diff": round(float(r.elevation) - float(elev[q]), 1)
                    if pd.notna(r.elevation)
                    else float("nan"),
                    "n_hours": n,
                    "rmse": round(rm, 4),
                    "mae": round(ma, 4),
                    "bias": round(bi, 4),
                }
            )
    return rows, len(keep)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="unet_main")
    p.add_argument("--batch", type=int, default=24)
    a = p.parse_args()
    cfg = load_config()
    rep = ROOT / cfg["paths"]["reports"]
    rep.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, names, norm = load_model(a.name, dev)
    booster = lgb.Booster(model_file=str(ROOT / cfg["paths"]["outputs"] / "lgbm.txt"))
    lnames = booster.feature_name()

    rows = []
    for region in region_list():
        r, _ = run_region(region, a.name, booster, lnames, model, names, norm, dev, a.batch)
        rows += r
    if not rows:
        print("no usable stations in either region")
        return

    df = pd.DataFrame(rows)
    df.to_csv(rep / "stations.csv", index=False)
    g = (
        df.groupby(["region", "label", "method"])
        .agg(stations=("id", "nunique"), median_rmse=("rmse", "median"), mean_bias=("bias", "mean"))
        .reset_index()
        .round(4)
    )
    g.to_csv(rep / "stations_summary.csv", index=False)
    print(g.to_string(index=False))


if __name__ == "__main__":
    main()
