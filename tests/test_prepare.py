import numpy as np
import pytest
from affine import Affine
from rasterio.enums import Resampling

from downscale.prepare import target_transform, warp


def test_target_transform_edges():
    lat = np.array([26.5, 26.4, 26.3])
    lon = np.array([51.0, 51.1])
    tf = target_transform(lat, lon, 0.1)
    assert tf.c == pytest.approx(50.95)
    assert tf.f == pytest.approx(26.55)
    assert tf.a == pytest.approx(0.1)
    assert tf.e == pytest.approx(-0.1)


def test_target_transform_uses_extremes():
    lat = np.array([22.5, 26.5, 24.0])
    lon = np.array([56.5, 51.0, 53.0])
    tf = target_transform(lat, lon, 0.1)
    assert tf.c == pytest.approx(50.95)
    assert tf.f == pytest.approx(26.55)


def test_warp_average_constant():
    src = np.full((40, 40), 7.5, "float32")
    src_tf = Affine(0.025, 0, 50.0, 0, -0.025, 26.0)
    dst_tf = Affine(0.1, 0, 50.0, 0, -0.1, 26.0)
    out = warp(src, src_tf, (10, 10), dst_tf, Resampling.average)
    assert out.shape == (10, 10)
    assert np.allclose(out, 7.5)


def test_warp_sum_counts_pixels():
    src = np.ones((40, 40), "float32")
    src_tf = Affine(0.025, 0, 50.0, 0, -0.025, 26.0)
    dst_tf = Affine(0.1, 0, 50.0, 0, -0.1, 26.0)
    out = warp(src, src_tf, (10, 10), dst_tf, Resampling.sum)
    assert np.allclose(out, 16.0)
