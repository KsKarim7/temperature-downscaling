import matplotlib

matplotlib.use("Agg")

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygeohash as pgh
import torch

from downscale.baselines import build_x, score_cells, store_path
from downscale.config import ROOT, load_config
from downscale.data import region_list
from downscale.features import static_feats, time_feats
from downscale.predict import load_model, run_model, score_model
from downscale.prepare import open_store
from downscale.splits import split_masks

LCS = ["lc_tree", "lc_shrub", "lc_grass", "lc_crop", "lc_built", "lc_bare", "lc_water"]
NICE = {
    "lc_tree": "tree",
    "lc_shrub": "shrub",
    "lc_grass": "grass",
    "lc_crop": "crop",
    "lc_built": "built",
    "lc_bare": "bare",
    "lc_water": "water",
}
LABEL = {"uae": "UAE region", "bangladesh": "Bangladesh region"}
EXPECT = {
    ("uae", "bilinear"): 1.1883,
    ("uae", "lgbm"): 0.8277,
    ("uae", "unet_main"): 0.7815,
    ("bangladesh", "bilinear"): 1.4736,
    ("bangladesh", "lgbm"): 0.9437,
    ("bangladesh", "unet_main"): 0.9167,
}
MIN_CELLS = 20


def add(acc, key, d):
    v = d[np.isfinite(d)]
    if v.size == 0:
        return
    a = acc.setdefault(key, np.zeros(4))
    a[0] += float(np.sum(v.astype("float64") ** 2))
    a[1] += float(np.sum(np.abs(v.astype("float64"))))
    a[2] += float(np.sum(v.astype("float64")))
    a[3] += v.size


def stats(a):
    n = a[3]
    return np.sqrt(a[0] / n), a[1] / n, a[2] / n, int(n)


def month_chunks(times, year):
    idx = pd.DatetimeIndex(times)
    out = []
    for m in range(1, 13):
        sel = np.where((np.asarray(idx.year) == year) & (np.asarray(idx.month) == m))[0]
        if sel.size:
            out.append((m, sel))
    return out


def lc_classes(ds, ii, jj):
    f = np.stack([ds[c].values[ii, jj] for c in LCS])
    tot = np.nansum(f, axis=0)
    k = np.nanargmax(np.nan_to_num(f, nan=-1.0), axis=0)
    out = np.array([NICE[LCS[q]] for q in k], dtype=object)
    out[tot <= 0] = "none"
    return out


def geohashes(ds, ii, jj, precision):
    lat = ds.lat.values[ii]
    lon = ds.lon.values[jj]
    return np.array([pgh.encode(float(a), float(o), precision=precision) for a, o in zip(lat, lon)], dtype=object)


def method_pred(method, ds, region, sel, ii, jj, tc, booster, lnames, model, names, norm, dev, batch):
    if method == "bilinear":
        return tc
    off = load_config()["regions"][region]["utc_offset"]
    if method == "lgbm":
        st = static_feats(ds, lnames)
        tf = time_feats(ds.time.values[sel], off)
        x = build_x(lnames, tc, tf, st, ii, jj, sel.size)
        p = booster.predict(x, num_iteration=booster.best_iteration).astype("float32")
        return p.reshape(tc.shape) + tc
    full = run_model(model, ds, names, norm, sel, off, dev, batch)
    return full[:, ii, jj]


def collect(region, name, booster, lnames, model, names, norm, dev, batch):
    cfg = load_config()
    ds = open_store(store_path(region))
    masks = split_masks(region)
    land = masks["all_land"][1]
    ii, jj = np.where(land)
    tsel = masks["test_blocks"][1][ii, jj]
    cls = lc_classes(ds, ii, jj)
    gh = geohashes(ds, ii, jj, cfg["geohash_precision"])
    off = cfg["regions"][region]["utc_offset"]
    methods = ["bilinear", "lgbm", name]

    acc = {}
    for m, sel in month_chunks(ds.time.values, cfg["years"]["test"]):
        tc = ds.t2m_coarse.isel(time=sel).values[:, ii, jj].astype("float32")
        tl = ds.t2m_land.isel(time=sel).values[:, ii, jj].astype("float32")
        hours = (pd.DatetimeIndex(ds.time.values[sel]).hour.to_numpy() + off) % 24
        for method in methods:
            pred = method_pred(
                method, ds, region, sel, ii, jj, tc, booster, lnames, model, names, norm, dev, batch
            )
            d = pred - tl
            add(acc, ("all_land", method), d)
            dt = d[:, tsel]
            add(acc, ("test_blocks", method), dt)
            add(acc, ("month", method, m), dt)
            for h in range(24):
                add(acc, ("hour", method, h), dt[hours == h])
            for c in np.unique(cls[tsel]):
                add(acc, ("lc", method, c), dt[:, cls[tsel] == c])
            for g in np.unique(gh[tsel]):
                add(acc, ("gh", method, g), dt[:, gh[tsel] == g])
        print(f"{region} 2024-{m:02d} done", flush=True)
    ds.close()
    counts = {
        "lc": {c: int(np.sum(cls[tsel] == c)) for c in np.unique(cls[tsel])},
        "gh": {g: int(np.sum(gh[tsel] == g)) for g in np.unique(gh[tsel])},
    }
    return acc, counts


def val_rmse(region, booster, lnames, model, names, norm, dev, batch, name):
    def r(pair):
        sse, n = pair
        return float(np.sqrt(sse / n)) if n else float("nan")

    return {
        "bilinear": r(score_cells(region, None, [], "val")),
        "lgbm": r(score_cells(region, booster, lnames, "val")),
        name: r(score_model(region, model, names, norm, "val", dev, batch)),
    }


def skill(rows, keys):
    base = {}
    for r in rows:
        if r["method"] == "bilinear":
            base[tuple(r[k] for k in keys)] = r["rmse_c"]
    for r in rows:
        b = base.get(tuple(r[k] for k in keys))
        r["skill"] = round(1.0 - r["rmse_c"] / b, 4) if b else float("nan")


def tables(region, name, acc, counts, vr):
    methods = ["bilinear", "lgbm", name]
    over, hour, month, lc, gh = [], [], [], [], []
    for split in ("val", "test_blocks", "all_land"):
        for m in methods:
            if split == "val":
                row = {"rmse_c": round(vr[m], 4), "mae_c": float("nan"), "bias_c": float("nan"), "n": 0}
            else:
                rm, ma, bi, n = stats(acc[(split, m)])
                row = {"rmse_c": round(rm, 4), "mae_c": round(ma, 4), "bias_c": round(bi, 4), "n": n}
            over.append({"region": region, "label": LABEL[region], "method": m, "split": split, **row,
                         "val_rmse_c": round(vr[m], 4)})
    skill(over, ["split"])

    for m in methods:
        for h in range(24):
            rm, ma, bi, n = stats(acc[("hour", m, h)])
            hour.append({"region": region, "label": LABEL[region], "method": m, "local_hour": h,
                         "rmse_c": round(rm, 4), "mae_c": round(ma, 4), "bias_c": round(bi, 4), "n": n})
    skill(hour, ["local_hour"])

    for m in methods:
        for mo in sorted({k[2] for k in acc if k[0] == "month"}):
            rm, ma, bi, n = stats(acc[("month", m, mo)])
            month.append({"region": region, "label": LABEL[region], "method": m, "month": mo,
                          "rmse_c": round(rm, 4), "mae_c": round(ma, 4), "bias_c": round(bi, 4), "n": n})
    skill(month, ["month"])

    keep = [c for c, k in counts["lc"].items() if k >= MIN_CELLS]
    for m in methods:
        for c in sorted(keep):
            rm, ma, bi, n = stats(acc[("lc", m, c)])
            lc.append({"region": region, "label": LABEL[region], "method": m, "landcover": c,
                       "cells": counts["lc"][c], "rmse_c": round(rm, 4), "mae_c": round(ma, 4),
                       "bias_c": round(bi, 4), "n": n})
    skill(lc, ["landcover"])

    for m in methods:
        for g in sorted(counts["gh"]):
            rm, ma, bi, n = stats(acc[("gh", m, g)])
            b = pgh.get_bounding_box(g)
            gh.append({"region": region, "label": LABEL[region], "method": m, "geohash": g,
                       "cells": counts["gh"][g], "min_lat": b.min_lat, "max_lat": b.max_lat,
                       "min_lon": b.min_lon, "max_lon": b.max_lon, "rmse_c": round(rm, 4),
                       "mae_c": round(ma, 4), "bias_c": round(bi, 4), "n": n})
    skill(gh, ["geohash"])
    return over, hour, month, lc, gh


def check(over, name):
    for r in over:
        if r["split"] != "test_blocks":
            continue
        want = EXPECT.get((r["region"], r["method"]))
        if want is None:
            continue
        if abs(r["rmse_c"] - want) > 5e-4:
            raise SystemExit(
                f"mismatch {r['region']} {r['method']} test_blocks rmse {r['rmse_c']} expected {want}"
            )
    print("overall test_blocks rmse matches predict.py")


def fig_hour(hour, regions, name, fdir):
    fig, axes = plt.subplots(1, len(regions), figsize=(5.2 * len(regions), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    df = pd.DataFrame(hour)
    for ax, region in zip(axes, regions):
        for m in ("bilinear", "lgbm", name):
            s = df[(df.region == region) & (df.method == m)].sort_values("local_hour")
            ax.plot(s.local_hour, s.rmse_c, marker="o", ms=3, label=m)
        ax.set_title(LABEL[region], fontsize=11)
        ax.set_xlabel("local hour")
        ax.set_xticks(range(0, 24, 3))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test-block RMSE (C)")
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(fdir / "rmse_by_hour.png", dpi=150)
    plt.close(fig)


def fig_val_test(over, regions, name, fdir):
    df = pd.DataFrame(over)
    methods = ["bilinear", "lgbm", name]
    fig, axes = plt.subplots(1, len(regions), figsize=(5.2 * len(regions), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(methods))
    for ax, region in zip(axes, regions):
        s = df[df.region == region].set_index(["method", "split"])
        v = [s.loc[(m, "val"), "rmse_c"] for m in methods]
        t = [s.loc[(m, "test_blocks"), "rmse_c"] for m in methods]
        ax.bar(x - 0.2, v, 0.4, label="val blocks", color="#4c72b0")
        ax.bar(x + 0.2, t, 0.4, label="test blocks", color="#c44e52")
        for q, (a, b) in enumerate(zip(v, t)):
            ax.text(q, max(a, b) + 0.02, f"{b - a:+.3f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title(LABEL[region], fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("RMSE (C)")
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(fdir / "val_vs_test.png", dpi=150)
    plt.close(fig)


def fig_geohash(gh, region, name, fdir):
    df = pd.DataFrame([r for r in gh if r["region"] == region and r["method"] == name])
    ds = open_store(store_path(region))
    land = ds.land.values > 0.5
    lat = ds.lat.values
    lon = ds.lon.values
    ds.close()
    vals = df.rmse_c.to_numpy()
    norm = plt.Normalize(vals.min(), vals.max())
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.contour(lon, lat, land.astype(float), levels=[0.5], colors="k", linewidths=0.8)
    for _, r in df.iterrows():
        ax.add_patch(
            plt.Rectangle(
                (r.min_lon, r.min_lat),
                r.max_lon - r.min_lon,
                r.max_lat - r.min_lat,
                facecolor=cmap(norm(r.rmse_c)),
                edgecolor="white",
                linewidth=0.4,
                alpha=0.85,
            )
        )
    ax.set_xlim(lon.min() - 0.1, lon.max() + 0.1)
    ax.set_ylim(lat.min() - 0.1, lat.max() + 0.1)
    ax.set_xlabel("longitude (deg E)")
    ax.set_ylabel("latitude (deg N)")
    ax.set_title(f"{name} test-block RMSE per geohash cell, {LABEL[region]}", fontsize=11)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="RMSE (C)")
    fig.tight_layout()
    fig.savefig(fdir / f"geohash_rmse_{region}.png", dpi=150)
    plt.close(fig)


def fig_example(region, name, model, names, norm, dev, fdir):
    cfg = load_config()
    off = cfg["regions"][region]["utc_offset"]
    ds = open_store(store_path(region))
    idx = pd.DatetimeIndex(ds.time.values)
    want = pd.Timestamp("2024-07-15 14:00") - np.timedelta64(int(off), "h")
    k = int(np.flatnonzero(idx == want)[0])
    sel = np.array([k])
    land = ds.land.values > 0.5
    pred = run_model(model, ds, names, norm, sel, off, dev, 1)[0]
    tc = ds.t2m_coarse.isel(time=sel).values[0]
    tl = ds.t2m_land.isel(time=sel).values[0]
    lat = ds.lat.values
    lon = ds.lon.values
    ds.close()
    for a in (tc, pred, tl):
        a[~land] = np.nan
    err = pred - tl
    ext = [lon.min(), lon.max(), lat.min(), lat.max()]
    lo = float(np.nanmin([np.nanmin(tc), np.nanmin(pred), np.nanmin(tl)]))
    hi = float(np.nanmax([np.nanmax(tc), np.nanmax(pred), np.nanmax(tl)]))
    e = float(np.nanmax(np.abs(err)))
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.2))
    titles = ["ERA5 coarse input", name, "ERA5-Land truth", f"{name} minus ERA5-Land"]
    for ax, a, t in zip(axes, (tc, pred, tl, err), titles):
        div = a is err
        im = ax.imshow(
            a,
            origin="upper" if lat[0] > lat[-1] else "lower",
            extent=ext,
            cmap="RdBu_r" if div else "inferno",
            vmin=-e if div else lo,
            vmax=e if div else hi,
        )
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("longitude (deg E)")
        fig.colorbar(im, ax=ax, label="error (C)" if div else "2 m temperature (C)", fraction=0.046)
    axes[0].set_ylabel("latitude (deg N)")
    fig.suptitle(f"{LABEL[region]}, 2024-07-15 14:00 local time", fontsize=12)
    fig.tight_layout()
    fig.savefig(fdir / f"example_{region}.png", dpi=150)
    plt.close(fig)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--name", default="unet_main")
    p.add_argument("--batch", type=int, default=24)
    a = p.parse_args()
    cfg = load_config()
    rep = ROOT / cfg["paths"]["reports"]
    fdir = rep / "figures"
    fdir.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, names, norm = load_model(a.name, dev)
    booster = lgb.Booster(model_file=str(ROOT / cfg["paths"]["outputs"] / "lgbm.txt"))
    lnames = booster.feature_name()
    regions = region_list()

    over, hour, month, lc, gh = [], [], [], [], []
    for region in regions:
        acc, counts = collect(region, a.name, booster, lnames, model, names, norm, dev, a.batch)
        vr = val_rmse(region, booster, lnames, model, names, norm, dev, a.batch, a.name)
        o, h, m, l, g = tables(region, a.name, acc, counts, vr)
        over += o
        hour += h
        month += m
        lc += l
        gh += g

    check(over, a.name)
    pd.DataFrame(over).to_csv(rep / "metrics_overall.csv", index=False)
    pd.DataFrame(hour).to_csv(rep / "metrics_by_hour.csv", index=False)
    pd.DataFrame(month).to_csv(rep / "metrics_by_month.csv", index=False)
    pd.DataFrame(lc).to_csv(rep / "metrics_by_landcover.csv", index=False)
    pd.DataFrame(gh).to_csv(rep / "metrics_by_geohash.csv", index=False)

    fig_hour(hour, regions, a.name, fdir)
    fig_val_test(over, regions, a.name, fdir)
    for region in regions:
        fig_geohash(gh, region, a.name, fdir)
        fig_example(region, a.name, model, names, norm, dev, fdir)
    print(f"wrote 5 csvs and {2 + 2 * len(regions)} figures")


if __name__ == "__main__":
    main()
