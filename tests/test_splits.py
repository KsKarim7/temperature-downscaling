import numpy as np
import pandas as pd

from downscale.config import load_config
from downscale.splits import spatial_blocks, val_days


def land_half():
    land = np.zeros((20, 20), dtype=bool)
    land[:10, :] = True
    return land


def test_sea_blocks_never_test():
    land = land_half()
    m = spatial_blocks(land, 5, 1.0, 7)
    assert not m[10:, :].any()
    assert m[:10, :].all()


def test_block_under_half_land_never_test():
    land = np.zeros((10, 10), dtype=bool)
    land[0:2, 0:5] = True
    m = spatial_blocks(land, 5, 1.0, 7)
    assert not m[0:5, 0:5].any()


def test_block_at_half_land_can_be_test():
    land = np.zeros((5, 5), dtype=bool)
    land[0:3, :] = True
    m = spatial_blocks(land, 5, 1.0, 7)
    assert m.all()


def test_same_seed_same_mask():
    land = land_half()
    a = spatial_blocks(land, 5, 0.25, 7)
    b = spatial_blocks(land, 5, 0.25, 7)
    assert np.array_equal(a, b)


def test_fraction_of_candidate_blocks():
    land = land_half()
    m = spatial_blocks(land, 5, 0.25, 7)
    assert m.sum() == 50


def test_train_and_test_cells_disjoint():
    land = land_half()
    test = spatial_blocks(land, 5, 0.25, 7)
    train = land & ~test
    assert not (train & test & land).any()
    assert np.array_equal(train | (land & test), land)


def test_val_days_only_train_year():
    cfg = load_config()
    times = pd.date_range("2023-01-01", "2024-12-31 23:00", freq="h").values
    v = val_days(times, 0.1, 7)
    years = sorted(pd.DatetimeIndex(times[v]).year.unique().tolist())
    assert years == [cfg["years"]["train"]]


def test_val_days_are_whole_days():
    times = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h").values
    v = val_days(times, 0.1, 7)
    day = pd.DatetimeIndex(times).normalize()
    counts = pd.Series(v).groupby(day).sum()
    assert set(counts.unique().tolist()) <= {0, 24}
    assert v.sum() == round(365 * 0.1) * 24


def test_val_days_same_seed():
    times = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h").values
    assert np.array_equal(val_days(times, 0.1, 7), val_days(times, 0.1, 7))
