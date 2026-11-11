from downscale.config import load_config
from downscale.download import dem_urls, pad_bbox, worldcover_urls

UAE = load_config()["regions"]["uae"]["bbox"]


def test_worldcover_tiles():
    urls = worldcover_urls(UAE)
    tiles = [u.split("_v200_")[1].split("_Map")[0] for u in urls]
    assert tiles == ["N21E051", "N21E054", "N24E051", "N24E054"]
    assert all(u.startswith("https://esa-worldcover.s3.eu-central-1.amazonaws.com") for u in urls)


def test_dem_tile_list():
    urls = dem_urls(UAE)
    assert len(urls) == 30
    names = [u.split("/")[-1] for u in urls]
    assert names[0] == "Copernicus_DSM_COG_30_N22_00_E051_00_DEM.tif"
    assert names[-1] == "Copernicus_DSM_COG_30_N26_00_E056_00_DEM.tif"
    assert "Copernicus_DSM_COG_30_N24_00_E054_00_DEM.tif" in names
    assert len(set(names)) == len(names)


def test_dem_covers_every_degree():
    urls = dem_urls(UAE)
    lats = {u.split("_30_")[1][:3] for u in urls}
    lons = {u.split("_00_")[1][:4] for u in urls}
    assert lats == {"N22", "N23", "N24", "N25", "N26"}
    assert lons == {"E051", "E052", "E053", "E054", "E055", "E056"}


def test_pad_bbox():
    assert pad_bbox(UAE, 0.5) == [27.0, 50.5, 22.0, 57.0]
