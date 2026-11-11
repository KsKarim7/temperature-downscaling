from downscale.baselines import shares


def test_shares_are_proportional():
    assert shares([1000, 3000], 400) == [100, 300]


def test_shares_sum_to_cap():
    out = shares([11400000, 26000000], 3_000_000)
    assert sum(out) == 3_000_000
    assert out[1] > out[0]


def test_shares_take_everything_when_under_cap():
    avail = [500, 1500]
    out = shares(avail, 3_000_000)
    assert all(o >= a for o, a in zip(out, avail))


def test_shares_single_region_gets_cap():
    assert shares([42], 300_000) == [300_000]


def test_shares_empty_split():
    assert shares([0, 0], 3_000_000) == [0, 0]
