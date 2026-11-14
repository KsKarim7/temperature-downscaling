import matplotlib

matplotlib.use("Agg")

import argparse
import datetime as dt

import lightgbm as lgb
import matplotlib.pyplot as plt
import meteostat
import numpy as np
import pandas as pd
import torch
from meteostat import Parameter
from scipy.stats import spearmanr

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


EXCLUDE = {
    "42423": "named Jorhat, a town at 26.75N 94.21E in eastern Assam, "
    "but listed at 26.07N 90.17E, about 400 km west"
}
MAX_ELEV_MISMATCH = 500.0


def bad_metadata(r, cell_elev, north, west, south, east):
    if r.id in EXCLUDE:
        return EXCLUDE[r.id]
    if not (south <= r.latitude <= north and west <= r.longitude <= east):
        return f"listed at {r.latitude:.4f}N {r.longitude:.4f}E, outside the region bbox"
    if pd.notna(r.elevation) and abs(float(r.elevation) - cell_elev) > MAX_ELEV_MISMATCH:
        return (
            f"listed elevation {float(r.elevation):.0f} m against {cell_elev:.0f} m "
            f"for its cell in the dem, a mismatch over {MAX_ELEV_MISMATCH:.0f} m"
        )
    return ""


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
        return [], []

    obs = station_hours(meta.id.to_numpy(), year)
    if obs.empty:
        print(f"{region} no hourly temperature returned")
        return [], []
    hours = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    obs = obs.reindex(hours)
    cover = obs.notna().sum() / len(hours)

    ds = open_store(store_path(region))
    land = ds.land.values > 0.5
    north, west, south, east = cfg["regions"][region]["bbox"]
    keep = []
    dropped = []
    for _, r in meta.iterrows():
        if r.id not in obs.columns or cover[r.id] < MIN_COVER:
            continue
        i, j = nearest_cell(ds, r.latitude, r.longitude)
        if not land[i, j]:
            continue
        why = bad_metadata(r, float(ds.elev.values[i, j]), north, west, south, east)
        if why:
            dropped.append(
                {
                    "region": region,
                    "id": r.id,
                    "name": station_name(r.id),
                    "lat": round(float(r.latitude), 4),
                    "lon": round(float(r.longitude), 4),
                    "reason": why,
                }
            )
            continue
        keep.append((r, i, j))
    ds.close()
    for d in dropped:
        print(f"{region} excluded {d['id']} {d['name']}: {d['reason']}", flush=True)
    print(f"{region} {len(keep)} stations with >={MIN_COVER:.0%} coverage on a land cell", flush=True)
    if len(keep) < MIN_STATIONS:
        print(f"{region} has fewer than {MIN_STATIONS} usable stations, treat its station numbers as anecdotal")
    if not keep:
        return [], []

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
    return rows, dropped


def is_airport(name):
    s = str(name).lower()
    return any(k in s for k in ("airport", "intl", "international"))


def neighbourhood_land(land, i, j):
    ny, nx = land.shape
    a = max(i - 1, 0)
    b = min(i + 2, ny)
    c = max(j - 1, 0)
    d = min(j + 2, nx)
    return float(land[a:b, c:d].mean())


def drivers(name):
    cfg = load_config()
    rep = ROOT / cfg["paths"]["reports"]
    df = pd.read_csv(rep / "stations.csv")
    rows = []
    for region in sorted(df.region.unique()):
        ds = open_store(store_path(region))
        land = (ds.land.values > 0.5).astype("float32")
        sub = df[df.region == region]
        for sid in sub.id.unique():
            s = sub[sub.id == sid]
            e = s[s.method == "era5land"].iloc[0]
            u = s[s.method == name].iloc[0]
            i, j = nearest_cell(ds, e.lat, e.lon)
            rows.append(
                {
                    "id": sid,
                    "name": e["name"],
                    "region": region,
                    "label": LABEL[region],
                    "era5land_bias": round(float(e.bias), 4),
                    "unet_bias": round(float(u.bias), 4),
                    "lc_built": round(float(ds.lc_built.values[i, j]), 4),
                    "lc_water": round(float(ds.lc_water.values[i, j]), 4),
                    "land_frac_3x3": round(neighbourhood_land(land, i, j), 4),
                    "elev_diff": e.elev_diff,
                    "airport": is_airport(e["name"]),
                }
            )
        ds.close()
    out = pd.DataFrame(rows)
    out.to_csv(rep / "stations_bias_drivers.csv", index=False)
    return out


def report_drivers(out, fdir):
    print("station counts are small, so read every number below as suggestive, not significant")
    for region in sorted(out.region.unique()):
        s = out[out.region == region]
        print(f"--- {LABEL[region]}, n = {len(s)} stations ---")
        for col in ("lc_built", "land_frac_3x3", "elev_diff"):
            v = s[["era5land_bias", col]].dropna()
            rho, p = spearmanr(v.era5land_bias, v[col])
            flag = "" if p < 0.05 else "  (not significant)"
            print(f"era5land_bias vs {col:14s} rho {rho:+.3f}  p {p:.3f}  n {len(v)}{flag}")
        for a, b in ((s.airport, "airport"), (s.land_frac_3x3 < 0.7, "coastal")):
            g1 = s[a].era5land_bias
            g0 = s[~a].era5land_bias
            m1 = f"{g1.mean():+.3f} c" if len(g1) else "no stations"
            m0 = f"{g0.mean():+.3f} c" if len(g0) else "no stations"
            print(f"mean era5land_bias {b} {m1} (n {len(g1)})  not {b} {m0} (n {len(g0)})")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for ax, col, xl in zip(
        axes,
        ("lc_built", "land_frac_3x3"),
        ("built-up fraction of cell (0 to 1)", "land fraction of 3x3 cell neighbourhood (0 to 1)"),
    ):
        for region, c in zip(sorted(out.region.unique()), ("#4c72b0", "#c44e52")):
            s = out[out.region == region]
            ax.scatter(s[col], s.era5land_bias, s=34, color=c, label=LABEL[region], alpha=0.85)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xlabel(xl)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ERA5-Land bias against station (C)")
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(fdir / "station_bias_drivers.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="unet_main")
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--drivers", action="store_true")
    a = p.parse_args()
    cfg = load_config()
    if a.drivers:
        rep = ROOT / cfg["paths"]["reports"]
        fdir = rep / "figures"
        fdir.mkdir(parents=True, exist_ok=True)
        report_drivers(drivers(a.name), fdir)
        return
    rep = ROOT / cfg["paths"]["reports"]
    rep.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, names, norm = load_model(a.name, dev)
    booster = lgb.Booster(model_file=str(ROOT / cfg["paths"]["outputs"] / "lgbm.txt"))
    lnames = booster.feature_name()

    rows = []
    drops = []
    for region in region_list():
        r, d = run_region(region, a.name, booster, lnames, model, names, norm, dev, a.batch)
        rows += r
        drops += d
    if not rows:
        print("no usable stations in either region")
        return

    pd.DataFrame(drops, columns=["region", "id", "name", "lat", "lon", "reason"]).to_csv(
        rep / "stations_excluded.csv", index=False
    )
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
