import numpy as np
import pandas as pd
import xarray as xr

from downscale.download import is_queue_full, is_too_big, periods, split_months

QUEUE_ERR = (
    "Maximum number queued requests for this dataset has been reached. "
    "Please configure your scripts to request data sequentially"
)


def make_nc(path, start, end):
    t = pd.date_range(start, end, freq="h")
    ds = xr.Dataset(
        {"t2m": (("valid_time", "latitude", "longitude"), np.ones((t.size, 2, 2), "float32"))},
        coords={"valid_time": t.values, "latitude": [26.5, 26.4], "longitude": [51.0, 51.1]},
    )
    ds.to_netcdf(path)
    ds.close()


def test_split_months_writes_one_file_per_month(tmp_path):
    src = tmp_path / "combined.nc"
    make_nc(src, "2023-01-30", "2023-02-02 23:00")
    made = split_months(src, "era5", tmp_path)
    assert [p.name for p in made] == ["era5_2023_01.nc", "era5_2023_02.nc"]
    with xr.open_dataset(tmp_path / "era5_2023_01.nc") as a:
        assert a.sizes["valid_time"] == 48
        assert str(pd.DatetimeIndex(a.valid_time.values)[0]) == "2023-01-30 00:00:00"
    with xr.open_dataset(tmp_path / "era5_2023_02.nc") as b:
        assert b.sizes["valid_time"] == 48
        assert str(pd.DatetimeIndex(b.valid_time.values)[-1]) == "2023-02-02 23:00:00"


def test_split_months_keeps_every_hour(tmp_path):
    src = tmp_path / "combined.nc"
    make_nc(src, "2023-03-01", "2023-05-31 23:00")
    made = split_months(src, "era5land", tmp_path)
    assert len(made) == 3
    total = 0
    for p in made:
        with xr.open_dataset(p) as ds:
            total += ds.sizes["valid_time"]
    assert total == pd.date_range("2023-03-01", "2023-05-31 23:00", freq="h").size


def test_split_months_leaves_no_part_files(tmp_path):
    src = tmp_path / "combined.nc"
    make_nc(src, "2023-01-01", "2023-01-02 23:00")
    split_months(src, "era5", tmp_path)
    assert list(tmp_path.glob("*.part")) == []


def test_queue_message_detected():
    assert is_queue_full(QUEUE_ERR)
    assert not is_too_big(QUEUE_ERR)


def test_too_large_message_detected():
    err = "the request is too large, cost limit exceeded"
    assert is_too_big(err)
    assert not is_queue_full(err)


def test_periods_are_year_and_quarter():
    assert periods("era5", [2023, 2024], None) == [
        (2023, list(range(1, 13))),
        (2024, list(range(1, 13))),
    ]
    q = periods("era5land", [2023], None)
    assert q == [(2023, [1, 2, 3]), (2023, [4, 5, 6]), (2023, [7, 8, 9]), (2023, [10, 11, 12])]


def test_periods_single_month():
    assert periods("era5", [2023], 7) == [(2023, [7])]
