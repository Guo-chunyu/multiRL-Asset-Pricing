"""Exploratory paired time-series intervals; training seeds are not markets.

Callers must align unique trading dates before passing NumPy arrays. A row of a
seed panel is one training seed and a column is one common market date. Never
flatten that panel or join separated walk-forward folds into one bootstrap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _returns(values, name, ndim):
    array = np.asarray(values, dtype=float)
    if array.ndim != ndim or not array.size or any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional return array")
    if not np.isfinite(array).all() or np.any(array < -1):
        raise ValueError(f"{name} must contain finite simple returns no smaller than -1")
    return array


def _periods(value):
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError("periods_per_year must be finite and positive")
    return float(value)


def _total_returns(returns, axis=None):
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.prod(1.0 + returns, axis=axis) - 1.0
    if not np.isfinite(result).all():
        raise ValueError("Compounded returns overflow; cannot report finite statistics")
    return result


def paired_block_bootstrap(
    strategy_returns,
    benchmark_returns,
    block_length: int = 10,
    n_bootstrap: int = 2000,
    seed: int = 7,
    confidence: float = .95,
    periods_per_year: float = 252,
) -> dict:
    """Percentile moving-block CI for annualized *arithmetic* mean excess.

    Paired observations form ``d[t] = strategy[t] - benchmark[t]``. Draw from
    every overlapping, non-wrapping block of ``block_length`` consecutive dates
    with replacement; concatenate and truncate the last draw to the original
    series length. The same resampling indices therefore apply to both paths.

    The reported CI concerns ``periods_per_year * mean(d)``. It is neither a
    CAGR interval nor a probability of outperforming in future. Cumulative
    return difference is a separate descriptive point estimate. At least two
    full blocks are required to avoid a mechanically degenerate one-block CI.
    The caller must enforce date alignment; matching pandas Series also have
    their indices checked here. No multiple-comparison correction is applied.
    """
    for item in (strategy_returns, benchmark_returns):
        if isinstance(item, pd.Series) and (not item.index.is_unique or not item.index.is_monotonic_increasing):
            raise ValueError("Return date index must be unique and increasing")
    if isinstance(strategy_returns, pd.Series) and isinstance(benchmark_returns, pd.Series):
        if not strategy_returns.index.equals(benchmark_returns.index):
            raise ValueError("Strategy and benchmark date indices must match exactly")
    strategy = _returns(strategy_returns, "strategy_returns", 1)
    benchmark = _returns(benchmark_returns, "benchmark_returns", 1)
    if strategy.shape != benchmark.shape:
        raise ValueError("Strategy and benchmark returns must have identical lengths")
    for value, label, minimum in ((block_length, "block_length", 1), (n_bootstrap, "n_bootstrap", 2)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{label} must be an integer >= {minimum}")
    if strategy.size < 2 * block_length:
        raise ValueError("At least two full blocks of observations are required")
    if not np.isscalar(confidence) or not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    periods = _periods(periods_per_year)
    n = strategy.size
    block_length, n_bootstrap = int(block_length), int(n_bootstrap)
    rng = np.random.default_rng(seed)
    excess = strategy - benchmark
    draws_per_resample = (n + block_length - 1) // block_length
    offsets = np.arange(block_length)
    estimates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        starts = rng.integers(0, n - block_length + 1, size=draws_per_resample)
        indices = (starts[:, None] + offsets).ravel()[:n]
        estimates[i] = periods * excess[indices].mean()
    if not np.isfinite(estimates).all():
        raise ValueError("Annualized excess estimates are not finite")
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(estimates, [alpha, 1 - alpha])
    strategy_total = float(_total_returns(strategy))
    benchmark_total = float(_total_returns(benchmark))
    daily_mean = float(excess.mean())
    return {
        "method": "paired_overlapping_moving_block_percentile",
        "n_observations": int(n),
        "block_length": block_length,
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": int(seed),
        "confidence": float(confidence),
        "periods_per_year": periods,
        "mean_daily_excess_return": daily_mean,
        "annualized_mean_excess_return": periods * daily_mean,
        "annualized_mean_excess_ci": [float(lower), float(upper)],
        "ci_contains_zero": bool(lower <= 0 <= upper),
        "strategy_total_return": strategy_total,
        "benchmark_total_return": benchmark_total,
        "cumulative_return_difference": strategy_total - benchmark_total,
        "interpretation": (
            "Exploratory percentile interval for annualized arithmetic mean paired "
            "daily excess, conditional on these realized paths. Not a CAGR interval, "
            "future-win probability, corrected multiple-comparison test, or proof "
            "of performance on a previously unseen market."
        ),
    }


def aggregate_seed_returns(returns_by_seed, periods_per_year: float = 252) -> dict:
    """Average aligned seed paths per date; keep seed dispersion separate.

    Input shape is ``[n_seeds, n_common_dates]``. The averaged path is a summary
    of algorithm outcomes, not an additional traded strategy. Its compounded
    return generally differs from the mean of each seed's compounded return.
    Seed standard deviations describe optimization randomness, not uncertainty
    across independent market histories. Undefined one-seed deviations are None.
    """
    returns = _returns(returns_by_seed, "returns_by_seed", 2)
    periods = _periods(periods_per_year)
    mean_daily = returns.mean(axis=0)
    total_by_seed = _total_returns(returns, axis=1)
    annual_mean_by_seed = periods * returns.mean(axis=1)
    if not np.isfinite(annual_mean_by_seed).all():
        raise ValueError("Annualized seed estimates are not finite")
    n_seeds, n_dates = returns.shape
    return {
        "n_seeds": int(n_seeds),
        "n_observations": int(n_dates),
        "mean_daily_returns": mean_daily.tolist(),
        "seed_total_returns": total_by_seed.tolist(),
        "seed_annualized_mean_returns": annual_mean_by_seed.tolist(),
        "mean_of_seed_total_returns": float(total_by_seed.mean()),
        "total_return_of_mean_daily_returns": float(_total_returns(mean_daily)),
        "seed_dispersion": {
            "total_return_std": float(total_by_seed.std(ddof=1)) if n_seeds > 1 else None,
            "annualized_mean_return_std": float(annual_mean_by_seed.std(ddof=1)) if n_seeds > 1 else None,
        },
        "aggregation": "equal mean across seeds for each already aligned market date",
    }


def compare_seeded_returns(
    returns_by_seed,
    benchmark_returns,
    block_length: int = 10,
    n_bootstrap: int = 2000,
    seed: int = 7,
    confidence: float = .95,
    periods_per_year: float = 252,
) -> dict:
    """Bootstrap the per-date seed mean against one aligned benchmark path.

    Repeating a seed does not add market dates. No bootstrap re-training is
    performed, so the interval is conditional on the provided trained models.
    For walk-forward studies call this independently for each test fold.
    """
    aggregate = aggregate_seed_returns(returns_by_seed, periods_per_year)
    comparison = paired_block_bootstrap(
        aggregate["mean_daily_returns"], benchmark_returns,
        block_length=block_length, n_bootstrap=n_bootstrap, seed=seed,
        confidence=confidence, periods_per_year=periods_per_year,
    )
    return {**aggregate, "comparison": comparison}
