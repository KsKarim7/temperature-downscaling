import argparse
import os
import time
from math import ceil, floor

import cdsapi
import numpy as np
import pandas as pd
import requests
import xarray as xr

from downscale.config import ROOT, load_config

HOURS = [f"{h:02d}:00" for h in range(24)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
DEM_BASE = "https://copernicus-dem-90m.s3.amazonaws.com"
WC_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
DATASETS = {
    "era5": "reanalysis-era5-single-levels",
    "era5land": "reanalysis-era5-land",
}
QUARTERS = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
QUEUE_WAIT = 600
QUEUE_TRIES = 18
NET_WAITS = [60, 120, 300]


def pad_bbox(bbox, pad):
    n, w, s, e = bbox
    return [n + pad, w - pad, s - pad, e + pad]


def dem_url(lat, lon):
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    name = f"Copernicus_DSM_COG_30_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
    return f"{DEM_BASE}/{name}/{name}.tif"


def dem_urls(bbox):
    n, w, s, e = bbox
    out = []
    for lat in range(floor(s), ceil(n)):
        for lon in range(floor(w), ceil(e)):
            out.append(dem_url(lat, lon))
    return out


def worldcover_urls(bbox):
    n, w, s, e = bbox
    out = []
    lat = floor(s / 3) * 3
    while lat < n:
        lon = floor(w / 3) * 3
        while lon < e:
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            tile = f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"
            out.append(f"{WC_BASE}/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
            lon += 3
        lat += 3
    return out


def is_queue_full(err):
    return "queued request" in str(err).lower()


def is_too_big(err):
    s = str(err).lower()
    keys = ("too large", "too big", "cost limit", "size limit", "exceeds", "exceeded the limit")
    return not is_queue_full(err) and any(k in s for k in keys)


def make_req(kind, area, year, months):
    req = {
        "variable": ["2m_temperature"],
        "year": [str(year)],
        "month": [f"{m:02d}" for m in months],
        "day": DAYS,
        "time": HOURS,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    }
    if kind == "era5":
        req["product_type"] = ["reanalysis"]
    return req


def retrieve(c, dataset, req, target):
    queue_tries = 0
    net_tries = 0
    while True:
        try:
            c.retrieve(dataset, req, str(target))
            return "ok"
        except Exception as err:
            if is_too_big(err):
                return "too_big"
            if is_queue_full(err):
                queue_tries += 1
                if queue_tries > QUEUE_TRIES:
                    raise SystemExit(
                        f"queue still full after {QUEUE_TRIES * QUEUE_WAIT // 3600} hours, "
                        "stopping. check for another download process and try again later"
                    )
                print("queue full, waiting 10 min")
                time.sleep(QUEUE_WAIT)
                continue
            net_tries += 1
            if net_tries > len(NET_WAITS):
                raise
            w = NET_WAITS[net_tries - 1]
            print(f"{err}")
            print(f"retrying in {w // 60} min")
            time.sleep(w)


def split_months(src, prefix, out_dir):
    ds = xr.open_dataset(src)
    keys = np.asarray(pd.DatetimeIndex(ds.valid_time.values).strftime("%Y_%m"))
    made = []
    for key in sorted(set(keys.tolist())):
        path = out_dir / f"{prefix}_{key}.nc"
        part = path.with_suffix(".part")
        ds.isel(valid_time=np.where(keys == key)[0]).to_netcdf(part)
        os.replace(part, path)
        made.append(path)
    ds.close()
    return made


def have(out_dir, prefix, year, months):
    return [out_dir / f"{prefix}_{year}_{m:02d}.nc" for m in months]


def fetch(c, region, kind, year, months, area, out_dir):
    want = have(out_dir, kind, year, months)
    if all(p.exists() and p.stat().st_size > 0 for p in want):
        return
    span = f"{year}-{months[0]:02d}" if len(months) == 1 else f"{year}-{months[0]:02d}..{months[-1]:02d}"
    print(f"{region} {kind} {span} requesting")
    tmp = out_dir / f"{kind}_{year}_{months[0]:02d}_{months[-1]:02d}.part"
    status = retrieve(c, DATASETS[kind], make_req(kind, area, year, months), tmp)
    if status == "too_big":
        if len(months) == 1:
            print(f"{region} {kind} {span} rejected as too large")
            return
        print(f"{region} {kind} {span} too large, falling back to monthly")
        for m in months:
            fetch(c, region, kind, year, [m], area, out_dir)
        return
    made = split_months(tmp, kind, out_dir)
    tmp.unlink()
    print(f"{region} {kind} {span} done, {len(made)} monthly files")


def periods(kind, years, month):
    if month:
        return [(y, [month]) for y in years]
    if kind == "era5":
        return [(y, list(range(1, 13))) for y in years]
    return [(y, list(q)) for y in years for q in QUARTERS]


def get_dem(region, bbox, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for url in dem_urls(bbox):
        path = out_dir / url.split("/")[-1]
        if path.exists() and path.stat().st_size > 0:
            continue
        r = requests.get(url, timeout=300)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        part = path.with_suffix(".part")
        part.write_bytes(r.content)
        os.replace(part, path)
        print(f"{region} dem {path.stem} done")


def report(raw, regions, years, bboxes):
    for region in regions:
        for kind in ("era5", "era5land"):
            got, missing = [], []
            for year in years:
                for m in range(1, 13):
                    p = raw / region / f"{kind}_{year}_{m:02d}.nc"
                    key = f"{year}-{m:02d}"
                    if p.exists() and p.stat().st_size > 0:
                        got.append(key)
                    else:
                        missing.append(key)
            print(f"{region} {kind}: {len(got)} of {len(got) + len(missing)} months")
            if missing:
                print("  missing " + ", ".join(missing))
        d = raw / region / "dem"
        n = len(list(d.glob("*.tif"))) if d.exists() else 0
        print(f"{region} dem: {n} of {len(dem_urls(bboxes[region]))} tiles, sea tiles are skipped")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True, choices=["uae", "bangladesh", "all"])
    p.add_argument("--part", default="all", choices=["all", "era5", "era5land", "dem"])
    p.add_argument("--year", type=int)
    p.add_argument("--month", type=int)
    a = p.parse_args()

    cfg = load_config()
    regions = ["uae", "bangladesh"] if a.region == "all" else [a.region]
    years = [a.year] if a.year else [cfg["years"]["train"], cfg["years"]["test"]]
    raw = ROOT / cfg["paths"]["raw"]
    c = cdsapi.Client() if a.part != "dem" else None

    for region in regions:
        bbox = cfg["regions"][region]["bbox"]
        out = raw / region
        out.mkdir(parents=True, exist_ok=True)
        if a.part in ("all", "era5"):
            area = pad_bbox(bbox, cfg["coarse_pad"])
            for year, months in periods("era5", years, a.month):
                fetch(c, region, "era5", year, months, area, out)
        if a.part in ("all", "era5land"):
            for year, months in periods("era5land", years, a.month):
                fetch(c, region, "era5land", year, months, bbox, out)
        if a.part in ("all", "dem"):
            get_dem(region, bbox, out / "dem")

    report(raw, regions, years, {r: cfg["regions"][r]["bbox"] for r in regions})


if __name__ == "__main__":
    main()
