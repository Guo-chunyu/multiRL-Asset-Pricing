"""Paired time-series uncertainty and seed aggregation, without pseudo-replication."""
import json

import numpy as np
import pandas as pd
import pytest


def test_identical_paired_paths_have_exactly_zero_excess_and_interval():
    from rlasset.statistics import paired_block_bootstrap

    path = np.array([.02, -.01, .03, -.02] * 10)
    result = paired_block_bootstrap(path, path, seed=11)
    assert result["annualized_mean_excess_return"] == 0
    assert result["annualized_mean_excess_ci"] == [0, 0]
    assert result["cumulative_return_difference"] == 0
    assert result["n_observations"] == 40
    assert result["n_bootstrap"] == 2000
    json.dumps(result, allow_nan=False)


def test_constant_daily_excess_has_the_correct_annualized_unit():
    from rlasset.statistics import paired_block_bootstrap

    benchmark = np.tile([.01, -.01, .005, -.005], 10)
    result = paired_block_bootstrap(benchmark + .001, benchmark, seed=19)
    assert result["mean_daily_excess_return"] == pytest.approx(.001)
    assert result["annualized_mean_excess_return"] == pytest.approx(.252)
    np.testing.assert_allclose(result["annualized_mean_excess_ci"], [.252, .252], atol=1e-14)


def test_moving_blocks_preserve_local_pairwise_cancellation():
    from rlasset.statistics import paired_block_bootstrap

    # Every contiguous two-day block sums to zero; independent day sampling
    # would instead create a nonzero-width distribution of sample means.
    strategy = np.array([.01, -.01, .01, -.01, .01, -.01])
    result = paired_block_bootstrap(strategy, np.zeros(6), block_length=2, seed=7)
    assert result["annualized_mean_excess_ci"] == [0., 0.]


def test_overlapping_nonwrapping_blocks_match_hand_enumerated_extremes():
    from rlasset.statistics import paired_block_bootstrap

    # Available blocks: [0,.01], [.01,.02], [.02,.03]. Resampling two
    # blocks produces daily means between .005 and .025, each extreme 1/9.
    result = paired_block_bootstrap([0, .01, .02, .03], [0] * 4,
                                    block_length=2, n_bootstrap=2000, seed=11)
    assert result["annualized_mean_excess_return"] == pytest.approx(3.78)
    assert result["annualized_mean_excess_ci"] == pytest.approx([1.26, 6.30])


def test_cumulative_difference_is_difference_of_compounded_paths_not_mean_excess():
    from rlasset.statistics import paired_block_bootstrap

    result = paired_block_bootstrap([.1, -.1], [0, 0], block_length=1, seed=7)
    assert result["annualized_mean_excess_return"] == pytest.approx(0)
    assert result["strategy_total_return"] == pytest.approx(-.01)
    assert result["benchmark_total_return"] == 0
    assert result["cumulative_return_difference"] == pytest.approx(-.01)


def test_bootstrap_rng_is_reproducible_and_does_not_depend_on_global_numpy_seed():
    from rlasset.statistics import paired_block_bootstrap

    returns = np.linspace(-.03, .04, 47)
    first = paired_block_bootstrap(returns, np.zeros(47), seed=123)
    np.random.seed(999)
    second = paired_block_bootstrap(returns, np.zeros(47), seed=123)
    assert first == second


def test_indexed_inputs_reject_misaligned_duplicate_or_unsorted_dates():
    from rlasset.statistics import paired_block_bootstrap

    dates = pd.bdate_range("2024-01-01", periods=30)
    strategy = pd.Series(np.zeros(30), index=dates)
    for bad in [
        pd.Series(np.zeros(30), index=dates + pd.Timedelta(days=1)),
        pd.Series(np.zeros(30), index=dates[::-1]),
        pd.Series(np.zeros(30), index=[dates[0]] * 30),
    ]:
        with pytest.raises(ValueError, match="index|date"):
            paired_block_bootstrap(strategy, bad)
    assert paired_block_bootstrap(strategy, strategy)["n_observations"] == 30


@pytest.mark.parametrize("strategy,benchmark,kwargs", [
    ([], [], {}), ([0] * 20, [0] * 21, {}), ([[0] * 20], [[0] * 20], {}),
    ([np.nan] * 20, [0] * 20, {}), ([0] * 20, [np.inf] * 20, {}),
    ([-1.01] * 20, [0] * 20, {}), ([0] * 19, [0] * 19, {"block_length": 10}),
    ([0] * 20, [0] * 20, {"block_length": 0}),
    ([0] * 20, [0] * 20, {"block_length": 1.5}),
    ([0] * 20, [0] * 20, {"n_bootstrap": 1}),
    ([0] * 20, [0] * 20, {"n_bootstrap": 20.5}),
    ([0] * 20, [0] * 20, {"confidence": 1}),
    ([0] * 20, [0] * 20, {"confidence": 0}),
    ([0] * 20, [0] * 20, {"periods_per_year": np.inf}),
])
def test_invalid_statistics_inputs_are_rejected(strategy, benchmark, kwargs):
    from rlasset.statistics import paired_block_bootstrap

    with pytest.raises(ValueError):
        paired_block_bootstrap(strategy, benchmark, **kwargs)


def test_seed_aggregation_keeps_dates_as_observations_and_reports_dispersion_separately():
    from rlasset.statistics import aggregate_seed_returns

    # Different realizations of training randomness on the same two market days.
    result = aggregate_seed_returns([[.1, -.1], [-.1, .1]])
    assert result["n_observations"] == 2
    assert result["n_seeds"] == 2
    assert result["mean_daily_returns"] == [0., 0.]
    assert result["seed_total_returns"] == pytest.approx([-.01, -.01])
    assert result["seed_dispersion"]["total_return_std"] == pytest.approx(0)
    assert result["total_return_of_mean_daily_returns"] == 0
    assert result["mean_of_seed_total_returns"] == pytest.approx(-.01)


def test_seed_return_dispersion_is_sample_std_not_a_market_confidence_interval():
    from rlasset.statistics import aggregate_seed_returns

    result = aggregate_seed_returns([[0, .1], [0, -.1], [0, 0]])
    assert result["seed_dispersion"]["total_return_std"] == pytest.approx(.1)
    assert result["seed_dispersion"]["annualized_mean_return_std"] == pytest.approx(12.6)
    one = aggregate_seed_returns([[0, .1]])
    assert one["seed_dispersion"]["total_return_std"] is None


def test_repeating_identical_seeds_does_not_manufacture_more_market_evidence():
    from rlasset.statistics import compare_seeded_returns

    path = np.linspace(-.02, .03, 40)
    benchmark = np.zeros(40)
    one = compare_seeded_returns(path[None, :], benchmark, seed=31)
    repeated = compare_seeded_returns(np.tile(path, (3, 1)), benchmark, seed=31)
    assert one["comparison"]["n_observations"] == repeated["comparison"]["n_observations"] == 40
    np.testing.assert_allclose(one["comparison"]["annualized_mean_excess_ci"],
                               repeated["comparison"]["annualized_mean_excess_ci"], atol=1e-15)
    assert repeated["n_seeds"] == 3
    assert repeated["seed_dispersion"]["total_return_std"] == pytest.approx(0)


def test_seeded_comparison_bootstraps_daily_mean_excess_without_flattening():
    from rlasset.statistics import compare_seeded_returns

    paths = np.array([[.001] * 40, [.003] * 40, [.005] * 40])
    result = compare_seeded_returns(paths, np.zeros(40), seed=7)
    assert result["comparison"]["annualized_mean_excess_return"] == pytest.approx(.756)
    assert result["comparison"]["annualized_mean_excess_ci"] == pytest.approx([.756, .756])
    assert result["seed_dispersion"]["annualized_mean_return_std"] == pytest.approx(.504)


@pytest.mark.parametrize("data", [[], [0, .1], np.empty((0, 20)), np.empty((2, 0)), [[np.nan] * 20]])
def test_seed_aggregation_rejects_missing_or_malformed_panels(data):
    from rlasset.statistics import aggregate_seed_returns

    with pytest.raises(ValueError):
        aggregate_seed_returns(data)
