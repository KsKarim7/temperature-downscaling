import argparse

import numpy as np
import pandas as pd
import rasterio
import rioxarray
import xarray as xr
from affine import Affine
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds

from downscale.config import ROOT, load_config
from downscale.download import worldcover_urls

LC_CODES = {
    "lc_tree": 10,
    "lc_shrub": 20,
    "lc_grass": 30,
    "lc_crop": 40,
    "lc_built": 50,
    "lc_bare": 60,
    "lc_water": 80,
}


def open_store(path):
    ds = xr.open_zarr(path, consolidated=True)
    if "spatial_ref" in ds.data_vars:
        ds = ds.set_coords("spatial_ref")
    return ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")


def write_store(ds, path):
    ds.to_zarr(path, mode="w", consolidated=True, zarr_format=2)


def target_transform(lat, lon, res):
    west = float(np.min(lon)) - res / 2
    north = float(np.max(lat)) + res / 2
    return from_origin(west, north, res, res)


def month_keys(d, prefix):
    return {p.stem.split("_", 1)[1] for p in d.glob(f"{prefix}_*.nc") if p.stat().st_size > 0}


def open_months(d, prefix, keys):
    files = [d / f"{prefix}_{k}.nc" for k in keys]
    ds = xr.open_mfdataset(files, combine="by_coords", engine="netcdf4")
    ds = ds.rename({"valid_time": "time", "latitude": "lat", "longitude": "lon"})
    drop = [c for c in ("number", "expver") if c in ds.variables]
    return ds.drop_vars(drop)


def check_times(t_land, t_coarse):
    if np.array_equal(t_land, t_coarse):
        return
    a = pd.Series(pd.DatetimeIndex(t_land).strftime("%Y-%m")).value_counts()
    b = pd.Series(pd.DatetimeIndex(t_coarse).strftime("%Y-%m")).value_counts()
    only = set(a.index) ^ set(b.index)
    differ = {m for m in set(a.index) & set(b.index) if a[m] != b[m]}
    bad = ", ".join(sorted(only | differ))
    raise SystemExit(f"era5 and era5-land timestamps differ for months: {bad}")


def dem_mosaic(dem_dir):
    srcs = [rasterio.open(p) for p in sorted(dem_dir.glob("*.tif"))]
    arr, tf = merge(srcs, nodata=0)
    for s in srcs:
        s.close()
    a = arr[0].astype("float32")
    a[~np.isfinite(a)] = 0.0
    return a, tf


def warp(src, src_tf, shape, dst_tf, how=Resampling.average):
    out = np.zeros(shape, "float32")
    reproject(
        src,
        out,
        src_transform=src_tf,
        src_crs="EPSG:4326",
        dst_transform=dst_tf,
        dst_crs="EPSG:4326",
        resampling=how,
    )
    return out


def landcover(bbox, lat, lon, res, cache):
    if cache.exists():
        return xr.open_dataset(cache)
    n, w, s, e = bbox
    dst_tf = target_transform(lat, lon, res)
    shape = (lat.size, lon.size)
    num = {k: np.zeros(shape, "float32") for k in LC_CODES}
    den = np.zeros(shape, "float32")
    for url in worldcover_urls(bbox):
        with rasterio.open("/vsicurl/" + url) as src:
            win = from_bounds(w, s, e, n, src.transform)
            win = win.intersection(Window(0, 0, src.width, src.height))
            win = win.round_offsets().round_lengths()
            oh = max(1, int(win.height) // 16)
            ow = max(1, int(win.width) // 16)
            arr = src.read(1, window=win, out_shape=(oh, ow), resampling=Resampling.nearest)
            tf = src.window_transform(win) * Affine.scale(win.width / ow, win.height / oh)
        den += warp(np.ones_like(arr, dtype="float32"), tf, shape, dst_tf, Resampling.sum)
        for k, code in LC_CODES.items():
            num[k] += warp((arr == code).astype("float32"), tf, shape, dst_tf, Resampling.sum)
        print(f"landcover {url.split('_v200_')[1][:7]} read")
    good = den > 0
    out = {}
    for k in LC_CODES:
        f = np.zeros(shape, "float32")
        f[good] = num[k][good] / den[good]
        out[k] = (("lat", "lon"), f)
    ds = xr.Dataset(out, coords={"lat": lat, "lon": lon})
    ds.to_netcdf(cache)
    return ds


def build(region, cfg):
    raw = ROOT / cfg["paths"]["raw"] / region
    proc = ROOT / cfg["paths"]["processed"]
    proc.mkdir(parents=True, exist_ok=True)
    keys = sorted(month_keys(raw, "era5") & month_keys(raw, "era5land"))
    if not keys:
        print(f"{region} has no month with both era5 and era5-land")
        return
    print(f"{region} months: {len(keys)}")

    land_ds = open_months(raw, "era5land", keys)
    coarse_ds = open_months(raw, "era5", keys)
    check_times(land_ds.time.values, coarse_ds.time.values)

    land_ds = land_ds.sortby("lon").sortby("lat", ascending=False)
    lat = land_ds.lat.values.astype("float64")
    lon = land_ds.lon.values.astype("float64")
    land_ds = land_ds.assign_coords(lat=lat, lon=lon)

    t_land = (land_ds.t2m - 273.15).astype("float32")
    cs = coarse_ds.sortby("lat").sortby("lon")
    clat = cs.lat.values.astype("float64")
    clon = cs.lon.values.astype("float64")
    cs = cs.assign_coords(lat=clat, lon=clon)
    t_coarse = (cs.t2m.interp(lat=lat, lon=lon) - 273.15).astype("float32")

    res = cfg["fine_res"]
    dst_tf = target_transform(lat, lon, res)
    shape = (lat.size, lon.size)
    mosaic, mtf = dem_mosaic(raw / "dem")

    elev = warp(mosaic, mtf, shape, dst_tf)
    elev2 = warp(mosaic**2, mtf, shape, dst_tf)
    elev_std = np.sqrt(np.maximum(elev2 - elev**2, 0.0)).astype("float32")

    ctf = target_transform(clat, clon, cfg["coarse_res"])
    elev_c = warp(mosaic, mtf, (clat.size, clon.size), ctf)
    da_c = xr.DataArray(elev_c, coords={"lat": clat, "lon": clon}, dims=("lat", "lon"))
    elev_anom = (elev - da_c.interp(lat=lat, lon=lon).values).astype("float32")

    lc = landcover(cfg["regions"][region]["bbox"], lat, lon, res, proc / f"{region}_landcover.nc")
    land = t_land.notnull().all("time").astype("float32").compute()
    lon2d, lat2d = np.meshgrid(lon, lat)

    out = xr.Dataset({"t2m_land": t_land, "t2m_coarse": t_coarse})
    out["elev"] = (("lat", "lon"), elev)
    out["elev_std"] = (("lat", "lon"), elev_std)
    out["elev_anom"] = (("lat", "lon"), elev_anom)
    for k in LC_CODES:
        out[k] = (("lat", "lon"), lc[k].values.astype("float32"))
    out["land"] = land
    out["lat2d"] = (("lat", "lon"), lat2d.astype("float32"))
    out["lon2d"] = (("lat", "lon"), lon2d.astype("float32"))

    out.attrs["region"] = region
    out.attrs["utc_offset"] = cfg["regions"][region]["utc_offset"]
    out.attrs["months"] = [k.replace("_", "-") for k in keys]

    out = out.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs("EPSG:4326")
    out = out.chunk({"time": 24 * 31})
    path = proc / f"{region}.zarr"
    write_store(out, path)
    print(f"{region} written {path.name} {shape[0]}x{shape[1]} {t_land.sizes['time']} hours")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True, choices=["uae", "bangladesh", "all"])
    a = p.parse_args()
    cfg = load_config()
    regions = ["uae", "bangladesh"] if a.region == "all" else [a.region]
    for region in regions:
        build(region, cfg)


if __name__ == "__main__":
    main()
