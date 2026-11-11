import argparse
import os
from math import ceil, floor
from pathlib import Path

import cdsapi
import requests

from downscale.config import ROOT, load_config

HOURS = [f"{h:02d}:00" for h in range(24)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
DEM_BASE = "https://copernicus-dem-90m.s3.amazonaws.com"
WC_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"


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


def get_hourly(c, region, dataset, prefix, area, years, months, out_dir, failed):
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        for month in months:
            path = out_dir / f"{prefix}_{year}_{month:02d}.nc"
            if path.exists() and path.stat().st_size > 0:
                continue
            req = {
                "variable": ["2m_temperature"],
                "year": str(year),
                "month": f"{month:02d}",
                "day": DAYS,
                "time": HOURS,
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": area,
            }
            if dataset == "reanalysis-era5-single-levels":
                req["product_type"] = ["reanalysis"]
            part = path.with_suffix(".part")
            try:
                c.retrieve(dataset, req, str(part))
                os.replace(part, path)
            except Exception as err:
                print(f"{region} {prefix} {year}-{month:02d} failed: {err}")
                failed.append(f"{region} {prefix} {year}-{month:02d}")
                continue
            print(f"{region} {prefix} {year}-{month:02d} done")


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
    months = [a.month] if a.month else list(range(1, 13))
    raw = ROOT / cfg["paths"]["raw"]
    c = cdsapi.Client() if a.part != "dem" else None
    failed = []

    for region in regions:
        bbox = cfg["regions"][region]["bbox"]
        out = raw / region
        if a.part in ("all", "era5"):
            area = pad_bbox(bbox, cfg["coarse_pad"])
            get_hourly(
                c,
                region,
                "reanalysis-era5-single-levels",
                "era5",
                area,
                years,
                months,
                out,
                failed,
            )
        if a.part in ("all", "era5land"):
            get_hourly(
                c,
                region,
                "reanalysis-era5-land",
                "era5land",
                bbox,
                years,
                months,
                out,
                failed,
            )
        if a.part in ("all", "dem"):
            get_dem(region, bbox, out / "dem")

    if failed:
        print("failed months:")
        for f in failed:
            print(f)


if __name__ == "__main__":
    main()
