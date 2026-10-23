"""Check price-adjustment semantics before using adjusted OHLC for P&L.

A same-day multiplicative adjustment preserves tradable intraday returns. Some
providers instead add accumulated cash distributions: their chart prices cannot
be used as though close/open were an ETF return. Passing this necessary check
does NOT verify dividends, splits, overnight total returns or point-in-time data.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .data import MarketPanel


_FIELDS = ("open", "high", "low", "close")


def _fingerprint(panel: MarketPanel) -> str:
    digest = hashlib.sha256(json.dumps(panel.tickers).encode("utf-8"))
    digest.update(np.asarray(panel.dates.asi8, dtype="<i8").tobytes())
    for field in (*_FIELDS, "volume"):
        digest.update(np.asarray(getattr(panel, field), dtype="<f8").tobytes())
    return digest.hexdigest()


def _rounding_error(decimals):
    if decimals is None:
        return 0.0
    if isinstance(decimals, bool) or not isinstance(decimals, (int, np.integer)) or not 0 <= decimals <= 15:
        raise ValueError("Price decimals must be an integer in [0, 15], or None for exact values")
    return 0.5 * 10.0 ** -int(decimals)


def audit_adjusted_prices(
    raw: MarketPanel,
    adjusted: MarketPanel,
    *,
    tolerance_bps: float = 5.0,
    raw_price_decimals: int | None = 3,
    adjusted_price_decimals: int | None = 3,
    volume_rtol: float = 1e-8,
    volume_atol: float = 1e-6,
) -> dict:
    """Return a JSON-serializable audit bound to both exact panel snapshots.

    Every bar must admit a common positive factor across open/high/low/close,
    allowing half a last decimal unit on both observed prices and the explicit
    relative tolerance. Factors may change BETWEEN dates; the caller must verify
    those changes independently against distributions/splits. No dates are joined,
    discarded, or filled. Volumes must already be in the same physical units.

    ``passed`` certifies only alignment, volume, and intraday consistency. The
    deliberately separate ``total_return_verified=False`` prevents this necessary
    test being interpreted as sufficient economic verification.
    """
    for value in (tolerance_bps, volume_rtol, volume_atol):
        if not np.isfinite(value) or value < 0:
            raise ValueError("Audit tolerances must be finite and nonnegative")
    raw_error = _rounding_error(raw_price_decimals)
    adjusted_error = _rounding_error(adjusted_price_decimals)
    if raw.tickers != adjusted.tickers or not raw.dates.equals(adjusted.dates):
        raise ValueError("Raw and adjusted date/ticker panels must align exactly")
    shape = (len(raw.dates), len(raw.tickers))
    if not all(shape) or not raw.dates.is_unique or not raw.dates.is_monotonic_increasing:
        raise ValueError("Aligned dates must be nonempty, unique and chronological")
    for panel in (raw, adjusted):
        for field in (*_FIELDS, "volume"):
            values = np.asarray(getattr(panel, field))
            if values.shape != shape:
                raise ValueError("All price and volume shapes must align with dates and tickers")
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError("Prices and volumes must be finite and positive")

    raw_prices = np.stack([getattr(raw, field) for field in _FIELDS])
    adjusted_prices = np.stack([getattr(adjusted, field) for field in _FIELDS])
    if (raw_prices <= raw_error).any() or (adjusted_prices <= adjusted_error).any():
        raise ValueError("Price precision is insufficient for a positive adjustment-factor interval")
    # True prices lie in observed +/- half a tick. Intersect their factor
    # intervals rather than rejecting legitimate three-decimal quantization.
    factor_lower = (adjusted_prices - adjusted_error) / (raw_prices + raw_error)
    factor_upper = (adjusted_prices + adjusted_error) / (raw_prices - raw_error)
    lower = factor_lower.max(axis=0)
    upper = factor_upper.min(axis=0)
    residual_bps = np.maximum(lower / upper - 1, 0) * 10000
    violations = residual_bps > tolerance_bps + 1e-8
    volume_ok = np.isclose(raw.volume, adjusted.volume, rtol=volume_rtol, atol=volume_atol)
    intraday_diff_bps = np.abs(adjusted.close / adjusted.open - raw.close / raw.open) * 10000
    examples = []
    for t, i in np.argwhere(violations)[:12]:
        examples.append({
            "date": str(raw.dates[t].date()), "ticker": raw.tickers[i],
            "raw_ohlc": {field: float(getattr(raw, field)[t, i]) for field in _FIELDS},
            "adjusted_ohlc": {field: float(getattr(adjusted, field)[t, i]) for field in _FIELDS},
            "intraday_abs_difference_bps": float(intraday_diff_bps[t, i]),
            "factor_inconsistency_after_rounding_bps": float(residual_bps[t, i]),
        })
    per_ticker = {
        ticker: {
            "violating_bars": int(violations[:, i].sum()),
            "volume_mismatches": int((~volume_ok[:, i]).sum()),
            "mean_intraday_abs_difference_bps": float(intraday_diff_bps[:, i].mean()),
            "max_intraday_abs_difference_bps": float(intraday_diff_bps[:, i].max()),
        }
        for i, ticker in enumerate(raw.tickers)
    }
    return {
        "schema_version": 1,
        "passed": bool(not violations.any() and volume_ok.all()),
        "total_return_verified": False,
        "raw_snapshot_sha256": _fingerprint(raw),
        "adjusted_snapshot_sha256": _fingerprint(adjusted),
        "dates": len(raw.dates), "tickers": list(raw.tickers), "bars": int(np.prod(shape)),
        "violating_bars": int(violations.sum()), "volume_mismatches": int((~volume_ok).sum()),
        "intraday_abs_difference_bps": {
            "mean": float(intraday_diff_bps.mean()), "max": float(intraday_diff_bps.max()),
        },
        "max_factor_inconsistency_after_rounding_bps": float(residual_bps.max()),
        "parameters": {
            "tolerance_bps": float(tolerance_bps), "raw_price_decimals": raw_price_decimals,
            "adjusted_price_decimals": adjusted_price_decimals,
            "volume_rtol": float(volume_rtol), "volume_atol": float(volume_atol),
        },
        "per_ticker": per_ticker, "examples": examples,
        "limitations": "Necessary intraday consistency check only. Verify corporate actions, overnight return factors, data precision and point-in-time availability independently. Low-resolution prices can conceal distortions smaller than their rounding intervals.",
    }


def require_audited_adjusted_pnl(
    raw: MarketPanel,
    adjusted: MarketPanel,
    audit: dict | None = None,
    *,
    corporate_actions_verified: bool = False,
) -> None:
    """Refuse adjusted-price P&L without a matching successful audit and review.

    ``corporate_actions_verified`` is an explicit caller attestation that the
    interday factor chain was checked against dividend/split events. This function
    cannot perform that check from two price panels and does not imply it did.
    """
    if not isinstance(audit, dict) or audit.get("schema_version") != 1:
        raise ValueError("An explicit adjusted-price audit is required before P&L")
    if (audit.get("raw_snapshot_sha256") != _fingerprint(raw)
            or audit.get("adjusted_snapshot_sha256") != _fingerprint(adjusted)):
        raise ValueError("Audit does not match the current price snapshot")
    if audit.get("passed") is not True:
        raise ValueError("Adjusted-price semantic audit failed; these ratios cannot be used for P&L")
    if corporate_actions_verified is not True:
        raise ValueError("Independent corporate-action and overnight-factor verification is required")
