import pytest

from rlasset.execution_data import decode_unadjusted


def payload(key="day"):
    return {"data": {"sh510300": {key: [
        ["2026-01-02", "4", "4.1", "4.2", "3.9", "123", "", ""],
        ["2026-01-05", "4.1", "4.2", "4.3", "4", "234", "", ""],
    ]}}}


def test_only_raw_day_prices_are_accepted_and_volume_is_shares():
    df = decode_unadjusted(payload(), "510300")
    assert df.open.iloc[0] == 4
    assert df.volume.iloc[0] == 12300
    assert df.tic.iloc[0] == "510300"


def test_adjusted_fallback_is_rejected():
    with pytest.raises(ValueError, match="unadjusted"):
        decode_unadjusted(payload("hfqday"), "510300")


def test_bad_bar_is_rejected():
    data = payload()
    data["data"]["sh510300"]["day"][0][2] = "0"
    with pytest.raises(ValueError):
        decode_unadjusted(data, "510300")
