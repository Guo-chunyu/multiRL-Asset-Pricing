"""Guard against treating additive chart adjustment as tradable returns."""
from dataclasses import replace

import numpy as np
import pytest

from rlasset.data import MarketPanel, synthetic_frame


def panels():
    raw = MarketPanel.from_frame(synthetic_frame(periods=12, assets=2))
    factors = np.linspace(1.1, 1.5, len(raw.dates))[:, None]
    adjusted = replace(raw, **{
        field: np.round(getattr(raw, field) * factors, 3)
        for field in ("open", "high", "low", "close")
    })
    return raw, adjusted


def test_multiplicative_prices_pass_intraday_check_but_not_total_return_certification():
    from rlasset.data_audit import audit_adjusted_prices
    raw, adjusted = panels()
    result = audit_adjusted_prices(raw, adjusted)
    assert result["passed"]
    assert result["total_return_verified"] is False
    assert result["violating_bars"] == 0
    assert result["bars"] == 24


def test_provider_additive_adjustment_is_rejected_even_with_correct_dates_and_volumes():
    from rlasset.data_audit import audit_adjusted_prices
    raw, _ = panels()
    adjusted = replace(raw, **{
        field: getattr(raw, field) * 0.37 + 0.25
        for field in ("open", "high", "low", "close")
    })
    result = audit_adjusted_prices(raw, adjusted, tolerance_bps=0, raw_price_decimals=None,
                                   adjusted_price_decimals=None)
    assert not result["passed"]
    assert result["violating_bars"] == 24
    assert result["intraday_abs_difference_bps"]["max"] > 0
    assert result["examples"][0]["date"] == str(raw.dates[0].date())


def test_actual_tencent_sample_fails_default_precision_tolerance():
    from rlasset.data_audit import audit_adjusted_prices
    raw, _ = panels()
    raw = replace(raw, **{field: np.full_like(raw.close, value) for field, value in
                          {"open": 4.007, "high": 4.240, "low": 3.876, "close": 4.233}.items()})
    adjusted = replace(raw, **{field: np.full_like(raw.close, value) for field, value in
                               {"open": 1.735, "high": 1.821, "low": 1.686, "close": 1.818}.items()})
    result = audit_adjusted_prices(raw, adjusted)
    assert not result["passed"]
    assert result["intraday_abs_difference_bps"]["mean"] == pytest.approx(85.6268101427693)


def test_high_low_factor_distortion_is_detected_when_open_close_are_unchanged():
    from rlasset.data_audit import audit_adjusted_prices
    raw, _ = panels()
    adjusted = replace(raw, high=raw.high * 1.1)
    result = audit_adjusted_prices(raw, adjusted)
    assert result["intraday_abs_difference_bps"]["max"] == 0
    assert not result["passed"]


def test_three_decimal_rounding_is_not_misclassified_as_additive_adjustment():
    from rlasset.data_audit import audit_adjusted_prices
    raw, _ = panels()
    raw = replace(raw, **{field: getattr(raw, field) / 50 for field in ("open", "high", "low", "close")})
    adjusted = replace(raw, **{field: np.round(getattr(raw, field) * 0.37123, 3)
                              for field in ("open", "high", "low", "close")})
    result = audit_adjusted_prices(raw, adjusted, tolerance_bps=0, raw_price_decimals=None)
    assert result["passed"]
    assert result["intraday_abs_difference_bps"]["max"] > 5


@pytest.mark.parametrize("kind", ["dates", "tickers", "shape"])
def test_misaligned_panels_are_refused(kind):
    from rlasset.data_audit import audit_adjusted_prices
    raw, adjusted = panels()
    if kind == "dates":
        adjusted = replace(adjusted, dates=adjusted.dates[::-1])
    elif kind == "tickers":
        adjusted = replace(adjusted, tickers=adjusted.tickers[::-1])
    else:
        adjusted = replace(adjusted, close=adjusted.close[:-1])
    with pytest.raises(ValueError, match="align|shape"):
        audit_adjusted_prices(raw, adjusted)


def test_volume_units_are_checked_with_explicit_rounding_tolerance():
    from rlasset.data_audit import audit_adjusted_prices
    raw, adjusted = panels()
    assert audit_adjusted_prices(raw, replace(adjusted, volume=raw.volume + 0.1), volume_atol=0.2)["passed"]
    result = audit_adjusted_prices(raw, replace(adjusted, volume=raw.volume / 100))
    assert not result["passed"]
    assert result["volume_mismatches"] == 24


def test_pnl_gate_requires_fresh_passed_audit_and_external_corporate_action_verification():
    from rlasset.data_audit import audit_adjusted_prices, require_audited_adjusted_pnl
    raw, adjusted = panels()
    with pytest.raises(ValueError, match="audit"):
        require_audited_adjusted_pnl(raw, adjusted)
    report = audit_adjusted_prices(raw, adjusted)
    with pytest.raises(ValueError, match="corporate"):
        require_audited_adjusted_pnl(raw, adjusted, report)
    require_audited_adjusted_pnl(raw, adjusted, report, corporate_actions_verified=True)
    changed = replace(adjusted, close=adjusted.close * 1.00001)
    with pytest.raises(ValueError, match="snapshot"):
        require_audited_adjusted_pnl(raw, changed, report, corporate_actions_verified=True)
    bad = replace(adjusted, high=adjusted.high * 1.05)
    with pytest.raises(ValueError, match="failed"):
        require_audited_adjusted_pnl(raw, bad, audit_adjusted_prices(raw, bad), corporate_actions_verified=True)


@pytest.mark.parametrize("kwargs", [{"tolerance_bps": -1}, {"tolerance_bps": np.nan},
                                    {"raw_price_decimals": -1}, {"adjusted_price_decimals": 1.5},
                                    {"volume_rtol": -1}, {"volume_atol": np.inf}])
def test_invalid_audit_tolerances_fail(kwargs):
    from rlasset.data_audit import audit_adjusted_prices
    raw, adjusted = panels()
    with pytest.raises(ValueError):
        audit_adjusted_prices(raw, adjusted, **kwargs)
