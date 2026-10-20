"""Performance metrics for a time-ordered sequence of net simple returns."""

from math import isfinite, sqrt

import numpy as np


def performance_metrics(returns: np.ndarray, turnover=None, periods_per_year=252) -> dict:
    """Return JSON-safe metrics; undefined or nonfinite ratios are None.

    annual_return is geometric, volatility uses sample standard deviation,
    Sharpe uses arithmetic mean and zero risk-free return. Sortino uses the
    root mean square of negative returns over *all* periods with target zero.
    max_drawdown is a positive loss magnitude and includes initial wealth 1.
    Input returns must already include execution fees and other modeled costs.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 1 or returns.size == 0:
        raise ValueError("returns must be a nonempty one-dimensional sequence")
    if not np.all(np.isfinite(returns)) or np.any(returns <= -1):
        raise ValueError("returns must be finite and strictly greater than -1")
    if not isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be finite and positive")
    mean_turnover = None
    if turnover is not None:
        turnover = np.asarray(turnover, dtype=float)
        if turnover.shape != returns.shape or not np.all(np.isfinite(turnover)) or np.any(turnover < 0):
            raise ValueError("turnover must be finite, nonnegative, and aligned with returns")
        mean_turnover = float(turnover.mean())

    def finite(value):
        return float(value) if np.isfinite(value) else None

    with np.errstate(over="ignore", invalid="ignore"):
        log_wealth = np.concatenate(([0.0], np.cumsum(np.log1p(returns))))
        # Log drawdown avoids overflow from constructing a long wealth curve.
        drawdowns = -np.expm1(log_wealth - np.maximum.accumulate(log_wealth))
        max_drawdown = float(drawdowns.max())
        annual_return = finite(np.expm1(log_wealth[-1] * periods_per_year / returns.size))
        volatility = float(returns.std(ddof=1)) if returns.size > 1 else 0.0
        mean_return = float(returns.mean())
        downside = float(np.sqrt(np.mean(np.minimum(returns, 0) ** 2)))
        sharpe = finite(mean_return / volatility * sqrt(periods_per_year)) if volatility > 1e-12 else None
        sortino = finite(mean_return / downside * sqrt(periods_per_year)) if downside > 1e-12 else None
        calmar = finite(annual_return / max_drawdown) if annual_return is not None and max_drawdown > 1e-12 else None
        return {
            "total_return": finite(np.expm1(log_wealth[-1])),
            "annual_return": annual_return,
            "annual_volatility": finite(volatility * sqrt(periods_per_year)),
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "sortino": sortino,
            "calmar": calmar,
            "mean_turnover": mean_turnover,
        }
