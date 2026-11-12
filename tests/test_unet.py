import numpy as np
import torch

from downscale.data import STATIC, compute_norm
from downscale.features import TIME_CHANNELS
from downscale.train import masked_loss
from downscale.unet import Unet, pad_to


def test_output_shape_32():
    m = Unet(17).eval()
    x = torch.randn(2, 17, 32, 32)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 1, 32, 32)


def test_output_shape_48x64():
    m = Unet(10).eval()
    x = torch.randn(1, 10, 48, 64)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 1, 48, 64)


def test_pad_to_multiple_of_eight():
    x = torch.randn(1, 3, 41, 56)
    p, h, w = pad_to(x, 8)
    assert p.shape[-2] % 8 == 0 and p.shape[-1] % 8 == 0
    assert (h, w) == (41, 56)
    assert p[..., :h, :w].shape == x.shape


def test_masked_loss_ignores_masked_pixels():
    y = torch.zeros(1, 1, 4, 4)
    m = torch.zeros(1, 1, 4, 4)
    m[..., :2, :] = 1.0
    a = torch.zeros(1, 1, 4, 4)
    a[..., 2:, :] = 5.0
    b = torch.zeros(1, 1, 4, 4)
    b[..., 2:, :] = -99.0
    for kind in ("l1", "mse"):
        assert float(masked_loss(a, y, m, kind)) == float(masked_loss(b, y, m, kind))
        assert float(masked_loss(a, y, m, kind)) == 0.0


def test_masked_loss_value():
    y = torch.zeros(1, 1, 2, 2)
    p = torch.full((1, 1, 2, 2), 3.0)
    m = torch.zeros(1, 1, 2, 2)
    m[..., 0, 0] = 1.0
    assert float(masked_loss(p, y, m, "l1")) == 3.0
    assert float(masked_loss(p, y, m, "mse")) == 9.0


def fake_region():
    cells = np.zeros((4, 4), bool)
    cells[:2, :] = True
    st = {c: np.where(cells, 100.0, 9999.0).astype("float32") for c in STATIC}
    tc = np.where(cells, 20.0, 9999.0).astype("float32")[None].repeat(3, 0)
    tf = {c: np.zeros(3, "float32") for c in TIME_CHANNELS}
    return {"tc": tc, "st": st, "tf": tf, "cells": cells}


def test_norm_uses_train_cells_only():
    d = fake_region()
    norm = compute_norm([d])
    assert norm["t2m_coarse"][0] == 20.0
    assert norm["elev"][0] == 100.0


def test_norm_zero_variance_channel_gets_unit_std():
    d = fake_region()
    norm = compute_norm([d])
    assert norm["elev"][1] == 1.0
    assert norm["t2m_coarse"][1] == 1.0
