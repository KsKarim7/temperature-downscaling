import numpy as np
import pandas as pd

from downscale.config import load_config
from downscale.splits import spatial_blocks, val_blocks, val_days


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


def old_spatial_blocks(land, block_size, test_fraction, seed):
    ny, nx = land.shape
    cand = []
    for by in range(0, ny, block_size):
        for bx in range(0, nx, block_size):
            if land[by : by + block_size, bx : bx + block_size].mean() >= 0.5:
                cand.append((by, bx))
    mask = np.zeros(land.shape, dtype=bool)
    if not cand:
        return mask
    rng = np.random.default_rng(seed)
    k = int(round(len(cand) * test_fraction))
    for i in rng.choice(len(cand), size=k, replace=False):
        by, bx = cand[i]
        mask[by : by + block_size, bx : bx + block_size] = True
    return mask


def patchy_land(seed=3):
    rng = np.random.default_rng(seed)
    return rng.random((40, 45)) < 0.7


def test_test_blocks_unchanged_from_old_function():
    for seed in (7, 11, 42):
        for frac in (0.25, 0.4):
            land = patchy_land(seed)
            assert np.array_equal(
                spatial_blocks(land, 5, frac, 7), old_spatial_blocks(land, 5, frac, 7)
            )


def test_test_blocks_unchanged_on_half_land():
    land = land_half()
    assert np.array_equal(spatial_blocks(land, 5, 0.25, 7), old_spatial_blocks(land, 5, 0.25, 7))


def test_val_blocks_never_overlap_test_blocks():
    land = patchy_land()
    test = spatial_blocks(land, 5, 0.25, 7)
    val = val_blocks(land, 5, 0.15, 7, test)
    assert not (test & val).any()


def test_train_val_test_blocks_disjoint_and_cover_land():
    land = patchy_land()
    test = spatial_blocks(land, 5, 0.25, 7)
    val = val_blocks(land, 5, 0.15, 7, test)
    tr = land & ~test & ~val
    va = land & val
    te = land & test
    assert not (tr & va).any()
    assert not (tr & te).any()
    assert not (va & te).any()
    assert np.array_equal(tr | va | te, land)


def test_val_blocks_only_on_eligible_blocks():
    land = np.zeros((10, 10), dtype=bool)
    land[0:2, 0:5] = True
    land[5:10, 5:10] = True
    test = spatial_blocks(land, 5, 0.0, 7)
    val = val_blocks(land, 5, 1.0, 7, test)
    assert not val[0:5, 0:5].any()


def test_val_blocks_fraction_of_all_eligible():
    land = np.ones((20, 20), dtype=bool)
    test = spatial_blocks(land, 5, 0.25, 7)
    val = val_blocks(land, 5, 0.25, 7, test)
    assert test.sum() == 4 * 25
    assert val.sum() == 4 * 25


def test_val_blocks_same_seed():
    land = patchy_land()
    test = spatial_blocks(land, 5, 0.25, 7)
    a = val_blocks(land, 5, 0.15, 7, test)
    b = val_blocks(land, 5, 0.15, 7, test)
    assert np.array_equal(a, b)
