"""Accounting-based checks for reported portfolio performance."""

import importlib
import json

import numpy as np
import pytest


def metrics(returns, **kwargs):
    return importlib.import_module("rlasset.metrics").performance_metrics(returns, **kwargs)


def test_initial_loss_is_counted_in_maximum_drawdown():
    # Omitting initial wealth would incorrectly report no drawdown here.
    result = metrics(np.array([-0.1, 0.1]), periods_per_year=2)
    assert result["max_drawdown"] == pytest.approx(0.1)
    assert result["total_return"] == pytest.approx(-0.01)
    assert result["annual_return"] == pytest.approx(-0.01)
    assert result["annual_volatility"] == pytest.approx(0.2)
    assert result["sharpe"] == pytest.approx(0)
    assert result["calmar"] == pytest.approx(-0.1)


def test_zero_return_series_has_json_safe_undefined_ratios():
    result = metrics(np.zeros(5))
    assert result["total_return"] == result["annual_volatility"] == result["max_drawdown"] == 0
    assert result["sharpe"] is result["sortino"] is result["calmar"] is None
    json.dumps(result, allow_nan=False)


def test_risk_ratios_use_sample_volatility_and_target_zero_downside_rms():
    result = metrics(np.array([0.02, -0.01]), periods_per_year=2,
                     turnover=np.array([0.1, 0.3]))
    assert result["annual_volatility"] == pytest.approx(0.03)
    assert result["sharpe"] == pytest.approx(1 / 3)
    assert result["sortino"] == pytest.approx(1.0)
    assert result["mean_turnover"] == pytest.approx(0.2)


def test_single_positive_return_has_no_estimable_volatility_or_downside():
    result = metrics(np.array([0.01]), periods_per_year=1)
    assert result["annual_return"] == pytest.approx(0.01)
    assert result["sharpe"] is None
    assert result["sortino"] is None
    assert result["calmar"] is None


@pytest.mark.parametrize("returns", [[], [np.nan], [np.inf], [-1.0], [-1.1], [[0.1, 0.2]]])
def test_invalid_returns_are_rejected(returns):
    with pytest.raises(ValueError):
        metrics(np.asarray(returns))


@pytest.mark.parametrize("kwargs", [
    {"turnover": [0.1]}, {"turnover": [-0.1, 0.2]}, {"turnover": [0.1, np.nan]},
    {"periods_per_year": 0}, {"periods_per_year": np.inf},
])
def test_invalid_turnover_or_annualization_is_rejected(kwargs):
    with pytest.raises(ValueError):
        metrics(np.array([0.01, 0.02]), **kwargs)
