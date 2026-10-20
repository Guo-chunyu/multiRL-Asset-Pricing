import numpy as np
import pandas as pd
import pytest


def test_reject_missing_duplicate_and_invalid_prices():
    from rlasset.data import MarketPanel, synthetic_frame
    df = synthetic_frame(periods=60, assets=2)
    for bad in (df.iloc[1:], pd.concat([df, df.iloc[:1]])):
        with pytest.raises(ValueError):
            MarketPanel.from_frame(bad)
    df.loc[0, "open"] = 0
    with pytest.raises(ValueError):
        MarketPanel.from_frame(df)


def test_panel_sorts_dates_and_preserves_ticker_codes():
    from rlasset.data import MarketPanel, synthetic_frame
    df = synthetic_frame(periods=60, assets=2)
    a = MarketPanel.from_frame(df)
    b = MarketPanel.from_frame(df.sample(frac=1, random_state=1))
    np.testing.assert_array_equal(a.close, b.close)
    assert a.close.shape == (60, 2)
    assert a.dates.is_monotonic_increasing


def test_time_splits_never_reuse_a_reward_or_cross_a_partition():
    from rlasset.data import chronological_splits
    splits = chronological_splits(600, lookback=20)
    last_end = -1
    reward_indices = []
    for start, end in splits.values():
        assert start > last_end and end > start
        reward_indices.extend(range(start + 1, end + 1))
        last_end = end
    assert len(reward_indices) == len(set(reward_indices))
    assert splits["test"][1] == 599
    with pytest.raises(ValueError):
        chronological_splits(30, lookback=20)


def test_synthetic_is_reproducible_and_identified_as_synthetic():
    from rlasset.data import synthetic_frame
    a = synthetic_frame(periods=80, assets=3, seed=7)
    pd.testing.assert_frame_equal(a, synthetic_frame(periods=80, assets=3, seed=7))
    assert all(a.tic.str.startswith("SYN"))
    assert not a.equals(synthetic_frame(periods=80, assets=3, seed=8))


def test_custom_partitions_reject_overlap_and_future_bounds():
    from rlasset.data import validate_splits
    valid = {"train": (20, 199), "meta": (200, 299), "validation": (300, 349), "test": (350, 399)}
    assert validate_splits(valid, 500, 20) == valid
    with pytest.raises(ValueError):
        validate_splits({**valid, "meta": (199, 299)}, 500, 20)
    with pytest.raises(ValueError):
        validate_splits({**valid, "test": (350, 500)}, 500, 20)
