import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from downscale.baselines import store_path
from downscale.config import ROOT, load_config
from downscale.data import region_list
from downscale.evaluate import LCS, LABEL
from downscale.features import time_feats
from downscale.predict import load_model
from downscale.prepare import open_store
from downscale.unet import pad_to

NONTREE = [c for c in LCS if c != "lc_tree"]
MONTHS = {1: "january", 7: "july"}
BUILT_MIN = 0.1
PLACEBO_MAX = 0.02


def statics(ds, names):
    return {c: ds[c].values.astype("float32") for c in names if c in ds}


def apply_shift(st, donors, shift):
    out = {k: v.copy() for k, v in st.items()}
    moved = np.zeros_like(st["lc_tree"])
    for name, mask in donors.items():
        d = np.where(mask, np.minimum(shift, st[name]), 0.0).astype("float32")
        out[name] = st[name] - d
        moved = moved + d
    out["lc_tree"] = st["lc_tree"] + moved
    return out, moved


def built_donors(st, land):
    return {"lc_built": land & (st["lc_built"] >= BUILT_MIN)}


def placebo_donors(st, land, shift):
    cells = land & (st["lc_built"] < PLACEBO_MAX)
    names = [c for c in NONTREE if c in st]
    big = np.argmax(np.stack([st[c] for c in names]), axis=0)
    use = np.where(st["lc_bare"] >= shift, names.index("lc_bare"), big)
    return {c: cells & (use == k) for k, c in enumerate(names)}


def stack(ds, names, norm, sel, off, st):
    tf = time_feats(ds.time.values[sel], off)
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
            v = st[c][None]
        x[:, k] = np.nan_to_num((v - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def predict(model, ds, names, norm, sel, off, st, dev, batch):
    out = []
    for s in np.array_split(sel, max(1, len(sel) // batch)):
        x = stack(ds, names, norm, s, off, st)
        t = torch.from_numpy(x).to(dev)
        t, h, w = pad_to(t, 8)
        with torch.no_grad():
            p = model(t)[:, 0, :h, :w].cpu().numpy()
        out.append(p)
    return np.concatenate(out)


def deltas(model, ds, names, norm, sel, off, st, st2, dev, batch):
    a = predict(model, ds, names, norm, sel, off, st, dev, batch)
    b = predict(model, ds, names, norm, sel, off, st2, dev, batch)
    return b - a


def region_rows(region, model, names, norm, shift, dev, batch):
    cfg = load_config()
    ds = open_store(store_path(region))
    land = ds.land.values > 0.5
    st = statics(ds, names)
    off = cfg["regions"][region]["utc_offset"]
    idx = pd.DatetimeIndex(ds.time.values)
    year = np.asarray(idx.year) == cfg["years"]["test"]

    groups = {
        "built": built_donors(st, land),
        "placebo": placebo_donors(st, land, shift),
    }
    rows = []
    for month, mname in MONTHS.items():
        sel = np.where(year & (np.asarray(idx.month) == month))[0]
        hours = (idx.hour.to_numpy()[sel] + off) % 24
        for group, donors in groups.items():
            st2, moved = apply_shift(st, donors, shift)
            cells = moved > 0
            if not cells.any():
                continue
            d = deltas(model, ds, names, norm, sel, off, st, st2, dev, batch)
            v = d[:, cells]
            for h in range(24):
                m = hours == h
                if not m.any():
                    continue
                rows.append(
                    {
                        "region": region,
                        "label": LABEL[region],
                        "month": mname,
                        "group": group,
                        "local_hour": h,
                        "n_cells": int(cells.sum()),
                        "mean_delta_c": round(float(np.nanmean(v[m])), 4),
                    }
                )
    ds.close()
    return rows


def summarise(rows):
    df = pd.DataFrame(rows)
    out = (
        df.groupby(["region", "label", "month", "group"])
        .agg(
            n_cells=("n_cells", "max"),
            mean_delta_c=("mean_delta_c", "mean"),
            min_hourly_delta_c=("mean_delta_c", "min"),
            max_hourly_delta_c=("mean_delta_c", "max"),
        )
        .reset_index()
        .round(4)
    )
    return df, out


def figure(df, fdir):
    regions = sorted(df.region.unique())
    fig, axes = plt.subplots(1, len(regions), figsize=(5.6 * len(regions), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    colour = {"july": "#c44e52", "january": "#4c72b0"}
    for ax, region in zip(axes, regions):
        for month in ("july", "january"):
            for group, style in (("built", "-"), ("placebo", "--")):
                s = df[
                    (df.region == region) & (df.month == month) & (df.group == group)
                ].sort_values("local_hour")
                if s.empty:
                    continue
                ax.plot(
                    s.local_hour,
                    s.mean_delta_c,
                    style,
                    color=colour[month],
                    label=f"{month} {group}",
                    lw=1.6,
                )
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_title(LABEL[region], fontsize=11)
        ax.set_xlabel("local hour")
        ax.set_xticks(range(0, 24, 3))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean predicted change (C)")
    axes[0].legend(fontsize=8)
    fig.suptitle("effect of moving built-up fraction to tree on the u-net prediction", fontsize=11)
    fig.tight_layout()
    fig.savefig(fdir / "whatif.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shift", type=float, default=0.2)
    p.add_argument("--name", default="unet_main")
    p.add_argument("--batch", type=int, default=24)
    a = p.parse_args()
    cfg = load_config()
    rep = ROOT / cfg["paths"]["reports"]
    fdir = rep / "figures"
    fdir.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, names, norm = load_model(a.name, dev)

    rows = []
    for region in region_list():
        rows += region_rows(region, model, names, norm, a.shift, dev, a.batch)
        print(f"{region} done", flush=True)

    df, out = summarise(rows)
    df.to_csv(rep / "whatif.csv", index=False)
    out.to_csv(rep / "whatif_summary.csv", index=False)
    figure(df, fdir)

    o = pd.read_csv(rep / "metrics_overall.csv")
    print()
    print(out.to_string(index=False))
    print()
    print(f"only {a.name} is used here: the chosen lightgbm config dropped the landcover features,")
    print("so lightgbm has no land cover to vary and cannot answer this question at all.")
    print()
    for region in sorted(out.region.unique()):
        s = o[(o.region == region) & (o.method == a.name) & (o.split == "test_blocks")]
        rmse = float(s.rmse_c.iloc[0])
        for _, r in out[out.region == region].iterrows():
            print(
                f"{LABEL[region]} {r['month']:8s} {r['group']:8s} "
                f"mean {r['mean_delta_c']:+.4f} c over {int(r['n_cells'])} cells, "
                f"{abs(r['mean_delta_c']) / rmse:.1%} of the {a.name} test_blocks rmse {rmse:.4f} c"
            )
    print()
    print("this is a sensitivity test of a statistical model trained at about 10 km, not a")
    print("physical simulation. it shows what the model learned to associate with land cover,")
    print("not a causal cooling effect.")


if __name__ == "__main__":
    main()
