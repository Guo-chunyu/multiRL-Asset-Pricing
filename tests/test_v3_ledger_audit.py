"""Independent ledger reconstruction must catch self-consistent-looking errors."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def auditor():
    path = Path(__file__).parents[1] / "scripts" / "audit_research_v3.py"
    spec = importlib.util.spec_from_file_location("v3_independent_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit_ledger


def sample_ledger(tmp_path):
    from rlasset.corporate_actions import CorporateActions, CorporateActionPortfolioEnv
    from rlasset.data import MarketPanel
    dates = pd.bdate_range("2024-01-01", periods=8)
    rows = []
    for i, date in enumerate(dates):
        for ticker in ("AAA", "BBB"):
            price = 20. if ticker == "BBB" else (10. if i < 3 else 9. if i < 6 else 4.5)
            rows.append(dict(date=date, tic=ticker, open=price, high=price, low=price, close=price, volume=10000))
    frame = pd.DataFrame(rows)
    dividends = pd.DataFrame([dict(ticker="AAA", ex_date="2024-01-04", pay_date="2024-01-08", cash_per_share=1.)])
    splits = pd.DataFrame([dict(ticker="AAA", effective_date="2024-01-09", ratio=2.)])
    dividends.to_csv(tmp_path / "cash.csv", index=False)
    splits.to_csv(tmp_path / "splits.csv", index=False)
    panel = MarketPanel.from_frame(frame)
    actions = CorporateActions.from_csv(panel, tmp_path / "cash.csv", tmp_path / "splits.csv")
    env = CorporateActionPortfolioEnv(panel, 1, 7, lookback=1, cost_bps=5, actions=actions)
    for _ in range(6):
        env.step_weights([0., .5, .5])
    return pd.DataFrame(env.history), frame, dividends, splits


def test_reconstructs_fees_cash_receivables_splits_nav_and_returns(tmp_path):
    audit = auditor()
    ledger, prices, dividends, splits = sample_ledger(tmp_path)
    result = audit(ledger, prices, dividends, splits, cost_bps=5)
    assert result["days"] == 6
    assert result["max_absolute_error"] < 1e-10
    assert result["dividend_entitlements"] == 1
    assert result["split_sessions"] == 1


@pytest.mark.parametrize("column", ["cash", "receivable", "portfolio_value", "net_return", "cost_fraction", "dividend_entitlement", "dividend_paid"])
def test_tampered_accounting_is_rejected(tmp_path, column):
    audit = auditor()
    ledger, prices, dividends, splits = sample_ledger(tmp_path)
    ledger.loc[2, column] += .01
    with pytest.raises(ValueError, match=column):
        audit(ledger, prices, dividends, splits, cost_bps=5)


def test_wrong_event_amount_cannot_hide_behind_a_consistent_reported_nav(tmp_path):
    audit = auditor()
    ledger, prices, dividends, splits = sample_ledger(tmp_path)
    dividends.loc[0, "cash_per_share"] = 2.
    with pytest.raises(ValueError):
        audit(ledger, prices, dividends, splits, cost_bps=5)


def test_gapped_or_shifted_dates_are_rejected(tmp_path):
    audit = auditor()
    ledger, prices, dividends, splits = sample_ledger(tmp_path)
    with pytest.raises(ValueError, match="consecutive"):
        audit(ledger.drop(index=3), prices, dividends, splits, cost_bps=5)
    ledger.loc[2, "decision_date"] = ledger.loc[2, "date"]
    with pytest.raises(ValueError, match="decision_date"):
        audit(ledger, prices, dividends, splits, cost_bps=5)
