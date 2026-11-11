import argparse

import numpy as np
import pandas as pd

from downscale.config import ROOT, load_config
from downscale.prepare import open_store

LC_NAMES = ["lc_tree", "lc_shrub", "lc_grass", "lc_crop", "lc_built", "lc_bare", "lc_water"]


def check_region(region):
    cfg = load_config()
    path = ROOT / cfg["paths"]["processed"] / f"{region}.zarr"
    ds = open_store(path)
    res = cfg["fine_res"]

    lat = ds.lat.values
    lon = ds.lon.values
    land = ds.land.values > 0.5
    sea = ~land
    t_land = ds.t2m_land.values
    t_coarse = ds.t2m_coarse.values
    land3 = np.broadcast_to(land, t_land.shape)
    elev = ds.elev.values

    r = {}
    r["region"] = region
    r["months"] = ds.attrs.get("months")
    r["utc_offset"] = ds.attrs.get("utc_offset")
    r["crs"] = str(ds.rio.crs)
    r["crs_ok"] = ds.rio.crs is not None and ds.rio.crs.to_epsg() == 4326
    r["shape"] = (int(lat.size), int(lon.size))

    r["dlat"] = float(np.abs(np.median(np.diff(lat))))
    r["dlon"] = float(np.abs(np.median(np.diff(lon))))
    r["spacing_ok"] = abs(r["dlat"] - res) < 1e-6 and abs(r["dlon"] - res) < 1e-6
    r["lat_descending"] = bool(np.all(np.diff(lat) < 0))
    r["lat_descending_ok"] = r["lat_descending"]
    r["lon_ascending_ok"] = bool(np.all(np.diff(lon) > 0))

    idx = pd.DatetimeIndex(ds.time.values)
    r["time_start"] = str(idx[0])
    r["time_end"] = str(idx[-1])
    r["n_hours"] = int(idx.size)
    missing = pd.date_range(idx[0], idx[-1], freq="h").difference(idx)
    r["missing_hours"] = int(missing.size)
    r["missing_hours_ok"] = missing.size == 0

    r["nan_frac_land_over_land"] = float(np.isnan(t_land[land3]).mean())
    r["nan_frac_land_over_land_ok"] = r["nan_frac_land_over_land"] == 0.0
    r["nan_frac_coarse"] = float(np.isnan(t_coarse).mean())
    r["nan_frac_coarse_ok"] = r["nan_frac_coarse"] == 0.0

    r["t2m_land_min"] = float(np.nanmin(t_land))
    r["t2m_land_max"] = float(np.nanmax(t_land))
    r["t2m_land_mean"] = float(np.nanmean(t_land))
    r["t2m_coarse_min"] = float(np.nanmin(t_coarse))
    r["t2m_coarse_max"] = float(np.nanmax(t_coarse))
    r["t2m_coarse_mean"] = float(np.nanmean(t_coarse))
    r["celsius_ok"] = -60.0 < r["t2m_land_min"] and r["t2m_land_max"] < 70.0
    r["bias_coarse_minus_land"] = float(np.nanmean(t_coarse[land3] - t_land[land3]))

    r["land_cells"] = int(land.sum())
    r["land_fraction"] = float(land.mean())

    lc_sum = sum(ds[k].values for k in LC_NAMES)
    r["lc_sum_max"] = float(lc_sum.max())
    r["lc_sum_ok"] = r["lc_sum_max"] <= 1.0 + 1e-3
    r["lc_water_sea"] = float(ds.lc_water.values[sea].mean())
    r["lc_water_land"] = float(ds.lc_water.values[land].mean())
    r["lc_water_ok"] = r["lc_water_sea"] > r["lc_water_land"]

    r["elev_min"] = float(elev.min())
    r["elev_max"] = float(elev.max())
    r["elev_mean_sea"] = float(elev[sea].mean())
    r["elev_mean_land"] = float(elev[land].mean())
    r["elev_sea_ok"] = abs(r["elev_mean_sea"]) < 20.0

    ds.close()
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True, choices=["uae", "bangladesh", "all"])
    a = p.parse_args()
    cfg = load_config()
    regions = ["uae", "bangladesh"] if a.region == "all" else [a.region]
    bad = 0
    for region in regions:
        path = ROOT / cfg["paths"]["processed"] / f"{region}.zarr"
        if not path.exists():
            continue
        r = check_region(region)
        print(f"--- {region} ---")
        for k, v in r.items():
            if k.endswith("_ok"):
                print(f"{k:32s} {'pass' if v else 'FAIL'}")
                bad += 0 if v else 1
            else:
                print(f"{k:32s} {v}")
    if bad:
        print(f"{bad} checks failed")


if __name__ == "__main__":
    main()
