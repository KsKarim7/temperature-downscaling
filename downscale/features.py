import numpy as np
import pandas as pd

CHANNELS = [
    "t2m_coarse",
    "elev",
    "elev_std",
    "elev_anom",
    "lc_tree",
    "lc_shrub",
    "lc_grass",
    "lc_crop",
    "lc_built",
    "lc_bare",
    "lc_water",
    "lat2d",
    "lon2d",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]

GROUPS = {
    "landcover": ["lc_tree", "lc_shrub", "lc_grass", "lc_crop", "lc_built", "lc_bare", "lc_water"],
    "terrain": ["elev", "elev_std", "elev_anom"],
    "position": ["lat2d", "lon2d"],
}

TIME_CHANNELS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]


def channels(drop=()):
    skip = set()
    for g in drop:
        skip.update(GROUPS[g])
    return [c for c in CHANNELS if c not in skip]


def time_feats(times, utc_offset):
    idx = pd.DatetimeIndex(times)
    h = (idx.hour.to_numpy() + utc_offset) % 24
    doy = idx.dayofyear.to_numpy()
    return {
        "hour_sin": np.sin(2 * np.pi * h / 24).astype("float32"),
        "hour_cos": np.cos(2 * np.pi * h / 24).astype("float32"),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25).astype("float32"),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25).astype("float32"),
    }


def static_feats(ds, names):
    return {c: ds[c].values.astype("float32") for c in names if c in ds}


def residual(t_land, t_coarse):
    return t_land - t_coarse
