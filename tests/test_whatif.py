import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch import nn

from downscale.whatif import NONTREE, apply_shift, built_donors, deltas, placebo_donors


class PickChannel(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        return x[:, self.k : self.k + 1]


NAMES = ["t2m_coarse", "lc_built", "lc_tree", "lc_bare", "lc_crop"]
NORM = {c: [0.0, 1.0] for c in NAMES}


def fake_ds(ny=8, nx=8, nt=4):
    rng = np.random.default_rng(0)
    built = np.full((ny, nx), 0.01, "float32")
    built[:2, :] = 0.5
    tree = np.full((ny, nx), 0.1, "float32")
    bare = np.full((ny, nx), 0.6, "float32")
    crop = np.full((ny, nx), 0.29, "float32")
    time = pd.date_range("2024-07-01", periods=nt, freq="h")
    return xr.Dataset(
        {
            "t2m_coarse": (("time", "lat", "lon"), rng.normal(300, 1, (nt, ny, nx)).astype("float32")),
            "lc_built": (("lat", "lon"), built),
            "lc_tree": (("lat", "lon"), tree),
            "lc_bare": (("lat", "lon"), bare),
            "lc_crop": (("lat", "lon"), crop),
        },
        coords={"time": time, "lat": np.arange(ny, dtype="float32"), "lon": np.arange(nx, dtype="float32")},
    )


def st_of(ds):
    return {c: ds[c].values.astype("float32") for c in NAMES if c in ds}


def test_delta_equals_the_shift_on_modified_cells_only():
    ds = fake_ds()
    st = st_of(ds)
    land = np.ones(st["lc_built"].shape, bool)
    shift = 0.2
    st2, moved = apply_shift(st, built_donors(st, land), shift)
    model = PickChannel(NAMES.index("lc_tree")).eval()
    sel = np.arange(ds.sizes["time"])
    d = deltas(model, ds, NAMES, NORM, sel, 0, st, st2, torch.device("cpu"), 2)
    cells = moved > 0
    assert cells.sum() == 16
    assert np.allclose(d[:, cells], shift, atol=1e-5)
    assert np.allclose(d[:, ~cells], 0.0, atol=1e-6)


def test_shift_is_capped_by_the_donor_value():
    st = st_of(fake_ds())
    st["lc_built"] = np.full_like(st["lc_built"], 0.12)
    land = np.ones(st["lc_built"].shape, bool)
    st2, moved = apply_shift(st, built_donors(st, land), 0.2)
    assert np.allclose(moved, 0.12)
    assert np.allclose(st2["lc_built"], 0.0)


def test_fractions_keep_their_total():
    ds = fake_ds()
    st = st_of(ds)
    land = np.ones(st["lc_built"].shape, bool)
    st2, _ = apply_shift(st, built_donors(st, land), 0.2)
    before = sum(st[c] for c in NAMES if c.startswith("lc_"))
    after = sum(st2[c] for c in NAMES if c.startswith("lc_"))
    assert np.allclose(before, after, atol=1e-6)


def test_built_group_selects_only_built_cells():
    st = st_of(fake_ds())
    land = np.ones(st["lc_built"].shape, bool)
    _, moved = apply_shift(st, built_donors(st, land), 0.2)
    assert (moved[:2, :] > 0).all()
    assert (moved[2:, :] == 0).all()


def test_placebo_group_avoids_built_cells_and_never_takes_from_tree():
    st = st_of(fake_ds())
    land = np.ones(st["lc_built"].shape, bool)
    donors = placebo_donors(st, land, 0.2)
    assert "lc_tree" not in donors
    assert set(donors) <= set(NONTREE)
    st2, moved = apply_shift(st, donors, 0.2)
    assert (moved[:2, :] == 0).all()
    assert (moved[2:, :] > 0).all()
    assert np.allclose(st2["lc_tree"][2:, :], st["lc_tree"][2:, :] + 0.2)


def test_placebo_takes_from_bare_when_bare_is_large_enough():
    st = st_of(fake_ds())
    land = np.ones(st["lc_built"].shape, bool)
    st2, _ = apply_shift(st, placebo_donors(st, land, 0.2), 0.2)
    assert np.allclose(st2["lc_bare"][2:, :], st["lc_bare"][2:, :] - 0.2)
    assert np.allclose(st2["lc_crop"][2:, :], st["lc_crop"][2:, :])


def test_placebo_falls_back_to_the_largest_non_tree_class():
    st = st_of(fake_ds())
    st["lc_bare"] = np.full_like(st["lc_bare"], 0.05)
    st["lc_crop"] = np.full_like(st["lc_crop"], 0.84)
    land = np.ones(st["lc_built"].shape, bool)
    st2, _ = apply_shift(st, placebo_donors(st, land, 0.2), 0.2)
    assert np.allclose(st2["lc_crop"][2:, :], st["lc_crop"][2:, :] - 0.2)
    assert np.allclose(st2["lc_bare"][2:, :], st["lc_bare"][2:, :])


def test_delta_is_zero_when_nothing_moves():
    ds = fake_ds()
    st = st_of(ds)
    land = np.zeros(st["lc_built"].shape, bool)
    st2, moved = apply_shift(st, built_donors(st, land), 0.2)
    assert moved.sum() == 0
    model = PickChannel(NAMES.index("lc_tree")).eval()
    sel = np.arange(ds.sizes["time"])
    d = deltas(model, ds, NAMES, NORM, sel, 0, st, st2, torch.device("cpu"), 2)
    assert np.allclose(d, 0.0, atol=1e-6)


def test_normalisation_scale_is_applied_to_the_shift():
    ds = fake_ds()
    st = st_of(ds)
    land = np.ones(st["lc_built"].shape, bool)
    st2, moved = apply_shift(st, built_donors(st, land), 0.2)
    norm = dict(NORM, lc_tree=[0.0, 2.0])
    model = PickChannel(NAMES.index("lc_tree")).eval()
    sel = np.arange(ds.sizes["time"])
    d = deltas(model, ds, NAMES, norm, sel, 0, st, st2, torch.device("cpu"), 2)
    assert np.allclose(d[:, moved > 0], 0.1, atol=1e-5)
