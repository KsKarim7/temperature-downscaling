import numpy as np
import pandas as pd
import pygeohash as pgh

from downscale.evaluate import add, stats
from downscale.features import time_feats


def one(d):
    acc = {}
    add(acc, "k", np.asarray(d, dtype="float32"))
    return stats(acc["k"])


def test_rmse_mae_bias_known():
    rm, ma, bi, n = one([1.0, -1.0, 1.0, -1.0])
    assert rm == 1.0 and ma == 1.0 and bi == 0.0 and n == 4


def test_bias_signed():
    rm, ma, bi, n = one([2.0, 4.0])
    assert np.isclose(rm, np.sqrt(10.0))
    assert ma == 3.0 and bi == 3.0 and n == 2


def test_mae_differs_from_rmse():
    rm, ma, bi, n = one([0.0, 0.0, 0.0, 4.0])
    assert ma == 1.0 and rm == 2.0 and bi == 1.0


def test_nan_cells_excluded():
    rm, ma, bi, n = one([3.0, np.nan, -3.0, np.nan])
    assert n == 2 and rm == 3.0 and ma == 3.0 and bi == 0.0


def test_all_nan_adds_nothing():
    acc = {}
    add(acc, "k", np.array([np.nan, np.nan], dtype="float32"))
    assert acc == {}


def test_accumulation_matches_single_pass():
    a = np.array([1.0, -2.0, 3.0, 4.5, -0.5], dtype="float32")
    acc = {}
    add(acc, "k", a[:2])
    add(acc, "k", a[2:])
    rm, ma, bi, n = stats(acc["k"])
    assert n == 5
    assert np.isclose(rm, np.sqrt(np.mean(a.astype("float64") ** 2)))
    assert np.isclose(ma, np.mean(np.abs(a.astype("float64"))))
    assert np.isclose(bi, np.mean(a.astype("float64")))


def test_geohash_grouping_three_points():
    pts = [(25.25, 55.33), (25.30, 55.35), (23.60, 58.50)]
    g = [pgh.encode(a, o, precision=4) for a, o in pts]
    assert all(len(x) == 4 for x in g)
    assert g[0] == g[1]
    assert g[2] != g[0]


def test_geohash_bounds_contain_point():
    a, o = 24.4, 54.5
    g = pgh.encode(a, o, precision=4)
    b = pgh.get_bounding_box(g)
    assert b.min_lat <= a <= b.max_lat
    assert b.min_lon <= o <= b.max_lon


def test_local_hour_offset_4():
    t = pd.date_range("2024-07-15 00:00", periods=24, freq="h")
    f = time_feats(t.values, 4)
    h = np.round(np.arctan2(f["hour_sin"], f["hour_cos"]) / (2 * np.pi) * 24).astype(int) % 24
    assert list(h) == [(q + 4) % 24 for q in range(24)]


def test_local_hour_offset_6_wraps():
    t = pd.DatetimeIndex(["2024-07-15 20:00", "2024-07-15 18:00", "2024-07-15 00:00"])
    f = time_feats(t.values, 6)
    h = np.round(np.arctan2(f["hour_sin"], f["hour_cos"]) / (2 * np.pi) * 24).astype(int) % 24
    assert list(h) == [2, 0, 6]


def test_local_hour_matches_plain_formula():
    t = pd.date_range("2024-01-01", periods=50, freq="7h")
    for off in (4, 6):
        want = (t.hour.to_numpy() + off) % 24
        f = time_feats(t.values, off)
        got = np.round(np.arctan2(f["hour_sin"], f["hour_cos"]) / (2 * np.pi) * 24).astype(int) % 24
        assert list(got) == list(want)
