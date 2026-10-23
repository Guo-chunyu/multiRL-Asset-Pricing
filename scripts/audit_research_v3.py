"""Independently reconstruct raw-price v3 ledgers and verify frozen artifacts.

This auditor does not import the environment or its corporate-action accountant.
It re-reads event CSVs and reconstructs cash flows from share changes at raw opens.
Use --validation-only before selection is frozen; test scans otherwise require
the frozen selection file to exist before any evaluation ledger is opened.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector(value):
    return np.asarray(json.loads(value) if isinstance(value, str) else value, dtype=float)


def audit_ledger(ledger, prices, dividends, splits, *, cost_bps=5):
    """Reconstruct every account transition, raising on any inconsistency."""
    ledger = ledger.reset_index(drop=True)
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices.date).dt.normalize()
    tickers = sorted(prices.tic.astype(str).unique())
    dates = pd.DatetimeIndex(sorted(prices.date.unique()))
    opens = prices.pivot(index="date", columns="tic", values="open").reindex(index=dates, columns=tickers).to_numpy()
    closes = prices.pivot(index="date", columns="tic", values="close").reindex(index=dates, columns=tickers).to_numpy()
    ledger_dates = pd.DatetimeIndex(pd.to_datetime(ledger.date))
    locations = dates.get_indexer(ledger_dates)
    if len(ledger) < 2 or (locations < 0).any() or not np.all(np.diff(locations) == 1):
        raise ValueError("Ledger must contain consecutive observed market dates")
    decisions = pd.DatetimeIndex(pd.to_datetime(ledger.decision_date))
    if not decisions[1:].equals(ledger_dates[:-1]):
        raise ValueError("decision_date must be the previous observed close")
    events, conversions = {}, {}
    for event in dividends.to_dict("records"):
        day = pd.Timestamp(event["ex_date"]).normalize()
        asset = tickers.index(str(event["ticker"]))
        payment = int(dates.searchsorted(pd.Timestamp(event["pay_date"]).normalize()))
        events.setdefault(day, []).append((asset, payment, float(event["cash_per_share"])))
    for event in splits.to_dict("records"):
        day = pd.Timestamp(event["effective_date"]).normalize()
        asset = tickers.index(str(event["ticker"]))
        conversions.setdefault(day, []).append((asset, float(event["ratio"])))

    errors = []
    def check(name, actual, expected):
        actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError(f"Invalid {name}")
        error = float(np.max(np.abs(actual - expected)))
        errors.append(error)
        if not np.allclose(actual, expected, rtol=1e-10, atol=1e-10):
            raise ValueError(f"{name} mismatch at {current_date}: absolute error={error}")

    current_date = str(ledger_dates[0].date())
    q = np.zeros(len(tickers))
    cash, nav, receivable = 1., 1., 0.
    pending = []
    for name, expected in (("cash", cash), ("portfolio_value", nav), ("receivable", 0.), ("net_return", 0.)):
        check(name, ledger.iloc[0][name], expected)
    check("shares", vector(ledger.iloc[0].shares), q)
    entitlement_days = split_days = 0
    for i in range(1, len(ledger)):
        row, t = ledger.iloc[i], locations[i]
        current_date = str(dates[t].date())
        ratio = np.ones(len(tickers))
        for asset, conversion in conversions.get(dates[t], []):
            ratio[asset] *= conversion
        q = q * ratio
        check("split_ratio", vector(row.split_ratio), ratio)
        split_days += int(not np.all(ratio == 1))
        entitlement = 0.
        for asset, pay_index, amount in events.get(dates[t], []):
            earned = q[asset] * amount
            entitlement += earned
            pending.append((pay_index, earned))
        entitlement_days += int(entitlement > 0)
        paid = sum(amount for pay_index, amount in pending if pay_index <= t)
        pending = [(pay_index, amount) for pay_index, amount in pending if pay_index > t]
        receivable = sum(amount for _, amount in pending)
        check("dividend_entitlement", row.dividend_entitlement, entitlement)
        check("dividend_paid", row.dividend_paid, paid)
        check("receivable", row.receivable, receivable)
        investable_open = cash + paid + float(q @ opens[t])
        open_nav = investable_open + receivable
        after_q = vector(row.shares)
        if (after_q < -1e-12).any() or row.cash < -1e-12:
            raise ValueError("Negative shares or cash")
        trade = after_q - q
        notional = float(np.abs(trade * opens[t]).sum())
        fee = float(cost_bps / 10000 * notional)
        expected_cash = cash + paid - float(trade @ opens[t]) - fee
        check("cash", row.cash, expected_cash)
        target = vector(row.target_weights)
        check("target_weights", np.r_[expected_cash, after_q * opens[t]] / (investable_open - fee), target)
        if "rebalance_executed" in ledger and not bool(row.rebalance_executed):
            check("hold_shares", trade, np.zeros(len(tickers)))
        next_nav = expected_cash + float(after_q @ closes[t]) + receivable
        check("portfolio_value", row.portfolio_value, next_nav)
        check("net_return", row.net_return, next_nav / nav - 1)
        check("cost_fraction", row.cost_fraction, fee / open_nav)
        check("turnover", row.turnover, notional / open_nav)
        check("gap_growth", row.gap_growth, open_nav / nav)
        check("intraday_growth", row.intraday_growth, next_nav / (open_nav - fee))
        check("weights", vector(row.weights), np.r_[expected_cash + receivable, after_q * closes[t]] / next_nav)
        q, cash, nav = after_q, expected_cash, next_nav
    return {
        "days": len(ledger) - 1, "max_absolute_error": max(errors),
        "dividend_entitlements": entitlement_days, "split_sessions": split_days,
        "first_decision": str(ledger_dates[0].date()), "last_valuation": str(ledger_dates[-1].date()),
    }


def audit_run(data_dir, run_dir, *, validation_only=False):
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    frozen_path = run_dir / "frozen_selection.json"
    if not validation_only and not frozen_path.exists():
        raise ValueError("Frozen selection is required before opening evaluation ledgers")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    project = Path(__file__).resolve().parents[1]
    for name, checksum in manifest["identity"]["data"].items():
        if digest(data_dir / name) != checksum:
            raise ValueError(f"Data hash mismatch: {name}")
    for name, checksum in manifest["identity"]["sources"].items():
        path = project / ("docs" if name.endswith(".md") else "src/rlasset") / name
        if digest(path) != checksum:
            raise ValueError(f"Source hash mismatch: {name}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))["folds"] if frozen_path.exists() else None
    prices = pd.read_csv(data_dir / "ashare_broad_raw.csv", dtype={"tic": str})
    dividends = pd.read_csv(data_dir / "cash_actions.csv", dtype={"ticker": str})
    splits = pd.read_csv(data_dir / "splits.csv", dtype={"ticker": str})
    results, models = {}, {}
    for fold in manifest["folds"]:
        fold_dir = run_dir / fold["name"]
        validation_scores = {}
        for candidate in manifest["identity"]["candidates"]:
            validation_scores[candidate] = []
            for seed in manifest["identity"]["seeds"]:
                directory = fold_dir / f"seed_{seed}"
                path = directory / f"{candidate}.zip"
                record = json.loads((directory / f"{candidate}.json").read_text(encoding="utf-8"))
                checksum = digest(path)
                if checksum != record["model_sha256"]:
                    raise ValueError("Saved model hash mismatch")
                if frozen is not None and checksum != frozen[fold["name"]]["model_hashes"][f"{seed}/{candidate}"]:
                    raise ValueError("Frozen model hash mismatch")
                models[str(path)] = checksum
                ledger_path = directory / f"validation_{candidate}.csv"
                ledger = pd.read_csv(ledger_path)
                results[str(ledger_path)] = audit_ledger(ledger, prices, dividends, splits)
                if [results[str(ledger_path)]["first_decision"], results[str(ledger_path)]["last_valuation"]] != fold["dates"]["validation"]:
                    raise ValueError("Validation dates differ from frozen plan")
                nav = ledger.portfolio_value.to_numpy()
                total = float(nav[-1] / nav[0] - 1)
                drawdown = float(np.max(1 - nav / np.maximum.accumulate(nav)))
                for field, expected in (("total_return", total), ("max_drawdown", drawdown)):
                    if not np.isclose(record["validation_metrics"][field], expected, rtol=1e-10, atol=1e-10):
                        raise ValueError("Validation selection metrics do not match ledger")
                validation_scores[candidate].append(total - .5 * drawdown)
        if frozen is not None:
            choice = max(validation_scores, key=lambda name: np.mean(validation_scores[name]))
            if choice != frozen[fold["name"]]["candidate"]:
                raise ValueError("Frozen choice is not the validation winner")
        if not validation_only:
            for folder in run_dir.glob("evaluation_*bps"):
                cost_bps = float(folder.name.removeprefix("evaluation_").removesuffix("bps"))
                for path in sorted((folder / fold["name"]).glob("*.csv")):
                    results[str(path)] = audit_ledger(pd.read_csv(path), prices, dividends, splits, cost_bps=cost_bps)
                    if [results[str(path)]["first_decision"], results[str(path)]["last_valuation"]] != fold["dates"]["test"]:
                        raise ValueError("Evaluation dates differ from frozen plan")
    return {
        "verified_utc": datetime.now(timezone.utc).isoformat(), "validation_only": validation_only,
        "ledgers": len(results), "transitions": sum(row["days"] for row in results.values()),
        "max_absolute_error": max(row["max_absolute_error"] for row in results.values()),
        "model_count": len(models), "model_sha256": models, "ledger_results": results,
        "interpretation": "Independent cash/share/receivable reconstruction from raw prices and event CSVs. This validates accounting and validation-only selection, not predictability, market-data completeness or real-world fills.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/v3")
    parser.add_argument("--run", default="runs/research_v3")
    parser.add_argument("--output", default="runs/verification_v3.json")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    result = audit_run(args.data_dir, args.run, validation_only=args.validation_only)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("ledgers", "transitions", "max_absolute_error", "model_count")}))


if __name__ == "__main__":
    main()
