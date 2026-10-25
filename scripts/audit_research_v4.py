"""Independent cash-flow and causal risk-target verification for the v4 study.

Risk reconstruction uses raw closes and the event CSVs, without importing the
research environments, their corporate-action helper, or their covariance code.
The v3 cash/share auditor is reused as an independent accounting implementation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector(value):
    return np.asarray(json.loads(value) if isinstance(value, str) else value, dtype=float)


def independent_signal(prices, dividends, splits):
    """Rebuild a feature-only reinvestment index from raw market/event records."""
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame.date).dt.normalize()
    frame["tic"] = frame.tic.astype(str)
    dates = pd.DatetimeIndex(sorted(frame.date.unique()))
    tickers = sorted(frame.tic.unique())
    close = frame.pivot(index="date", columns="tic", values="close").reindex(
        index=dates, columns=tickers).to_numpy(float)
    if not np.isfinite(close).all() or np.any(close <= 0):
        raise ValueError("Risk reference requires a complete positive close matrix")
    cash, ratios = np.zeros_like(close), np.ones_like(close)
    for row in dividends.to_dict("records"):
        date = pd.Timestamp(row["ex_date"]).normalize()
        if date < dates[0] or date > dates[-1]:
            continue
        location = dates.get_indexer([date])[0]
        if location < 0:
            raise ValueError("Dividend ex-date missing from risk reference")
        cash[location, tickers.index(str(row["ticker"]))] += float(row["cash_per_share"])
    for row in splits.to_dict("records"):
        date = pd.Timestamp(row["effective_date"]).normalize()
        if date < dates[0] or date > dates[-1]:
            continue
        location = dates.get_indexer([date])[0]
        if location < 0:
            raise ValueError("Split date missing from risk reference")
        ratios[location, tickers.index(str(row["ticker"]))] *= float(row["ratio"])
    signal = np.empty_like(close)
    signal[0] = close[0]
    # Sequential update deliberately differs from the environment's cumprod.
    for day in range(1, len(dates)):
        gain = ratios[day] * close[day] + ratios[day] * cash[day]
        signal[day] = signal[day - 1] * gain / close[day - 1]
    return dates, tickers, signal


def _missing(value):
    return (value is None or (isinstance(value, str) and value in ("", "null", "None"))
            or (np.isscalar(value) and not isinstance(value, str) and bool(pd.isna(value))))


def audit_risk_targets(ledger, prices, dividends, splits, *, budget_bounds=None, rule=None, reference=None):
    """Check proposed risks at decision closes, separately from actual hold weights.

    ``budget_bounds`` applies to single learned policies, not rule benchmarks or
    mixtures. The original scalar risk-budget action cannot be recovered when
    exposure is capped at 100%; the check then only establishes the upper bound.
    """
    columns = ("requested_exposure", "estimated_risky_vol", "requested_target_vol", "requested_risky_mix")
    if not set(columns) <= set(ledger.columns):
        raise ValueError("Required risk diagnostic fields are missing")
    if reference is None:
        reference = {"signal": independent_signal(prices, dividends, splits), "covariance": {}}
    dates, tickers, signal = reference["signal"]
    cache = reference["covariance"]
    errors, requests, executed = [], 0, 0

    def check(label, actual, expected):
        actual, expected = np.asarray(actual, float), np.asarray(expected, float)
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError(f"Invalid risk {label}")
        error = float(np.max(np.abs(actual - expected)))
        errors.append(error)
        if not np.allclose(actual, expected, rtol=1e-9, atol=1e-10):
            raise ValueError(f"Risk {label} mismatch: {error}")

    for row in ledger.iloc[1:].itertuples(index=False):
        values = [getattr(row, field) for field in columns]
        missing = [_missing(value) for value in values]
        if all(missing):
            continue
        if any(missing):
            raise ValueError("Partially missing risk diagnostics")
        requests += 1
        decision = pd.Timestamp(row.decision_date).normalize()
        day = dates.get_indexer([decision])[0]
        if day < 60:
            raise ValueError("Insufficient decision history for risk covariance")
        exposure = float(row.requested_exposure)
        if not np.isfinite(exposure) or not 0 <= exposure <= 1 + 1e-10:
            raise ValueError("Risk exposure must be in [0, 1]")
        mix = vector(row.requested_risky_mix)
        if mix.shape != (len(tickers),) or not np.isfinite(mix).all():
            raise ValueError("Invalid requested risky mix")
        if np.any(mix < -1e-10) or np.any(mix > .5 + 1e-9):
            raise ValueError("Requested risky mix exceeds its cap")
        check("mix simplex", mix.sum(), 1. if exposure > 1e-12 else 0.)
        if day not in cache:
            matrices = []
            for horizon in (20, 60):
                history = np.log(signal[day - horizon:day + 1])
                log_returns = np.diff(history, axis=0)
                sample = np.atleast_2d(np.cov(log_returns, rowvar=False, ddof=1))
                matrices.append(.75 * sample + .25 * np.diag(np.diag(sample)))
            cache[day] = matrices
        sigma = max(.01, *[np.sqrt(252 * max(float(mix @ matrix @ mix), 0.)) for matrix in cache[day]])
        check("estimated risky volatility", row.estimated_risky_vol, sigma)
        check("requested target volatility", row.requested_target_vol, exposure * sigma)
        if rule is not None:
            if rule not in ("vol_target10", "invvol_target10"):
                raise ValueError("Unknown risk rule")
            preferences = np.ones(len(tickers))
            if rule == "invvol_target10":
                returns = np.diff(np.log(signal[day - 60:day + 1]), axis=0)
                preferences = 1 / np.maximum(np.std(returns, axis=0, ddof=1), 1e-6)
            expected_mix = preferences / preferences.sum()
            remaining = np.ones(len(tickers), dtype=bool)
            while np.any(expected_mix > .5 + 1e-12):
                remaining[expected_mix > .5] = False
                expected_mix[~remaining] = .5
                expected_mix[remaining] = ((1 - .5 * np.count_nonzero(~remaining))
                                           * preferences[remaining] / preferences[remaining].sum())
            check("rule risky mix", mix, expected_mix)
            check("rule exposure", exposure, min(1., .10 / sigma))
        if budget_bounds is not None:
            lower, upper = budget_bounds
            if exposure * sigma > upper + 1e-9 or (exposure < 1 - 1e-9 and exposure * sigma < lower - 1e-9):
                raise ValueError("Requested target is outside the registered risk budget")
        is_executed = row.rebalance_executed
        if isinstance(is_executed, str):
            if is_executed not in ("True", "False"):
                raise ValueError("Invalid rebalance execution flag")
            is_executed = is_executed == "True"
        if is_executed:
            executed += 1
            check("executed target", vector(row.target_weights), np.r_[1 - exposure, exposure * mix])
    return {"requests": requests, "executed_requests": executed,
            "max_absolute_error": max(errors, default=0.)}


def verify_record_seal(record):
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if record.get("record_sha256") != hashlib.sha256(canonical).hexdigest():
        raise ValueError("Training/validation record seal mismatch")


def verify_selection(frozen, learned, rules):
    """Recompute both winners without calling the research selection function."""
    scores = {name: float(np.mean([row["total_return"] - .5 * row["max_drawdown"] for row in rows]))
              for name, rows in learned.items()}
    deployed = {**scores, **{name: row["total_return"] - .5 * row["max_drawdown"]
                             for name, row in rules.items()}}
    for field, expected in (("scores", scores), ("deploy_scores", deployed)):
        actual = frozen[field]
        if set(actual) != set(expected) or any(not np.isclose(actual[key], expected[key], rtol=1e-10, atol=1e-10)
                                               for key in expected):
            raise ValueError("Frozen selection scores differ from independent validation metrics")
    selected, deploy_selected = max(scores, key=scores.get), max(deployed, key=deployed.get)
    if (frozen["candidate"] != selected or frozen["deploy_candidate"] != deploy_selected
            or frozen["deploy_kind"] != ("rule" if deploy_selected in rules else "rl")
            or frozen["selected_on"] != "validation"
            or frozen["criterion"] != "mean_seed(total_return - 0.5 * max_drawdown)"):
        raise ValueError("Frozen selection does not match validation winners")
    return {"candidate": selected, "deploy_candidate": deploy_selected}


def verify_evaluation_inventory(folder, candidates, seeds, baselines):
    expected = {f"seed_{seed}_{name}.csv" for seed in seeds for name in (*candidates, "joint_mixture")}
    expected.update(f"{name}.csv" for name in baselines)
    actual = {path.name for path in Path(folder).glob("*.csv")}
    if actual != expected:
        raise ValueError(f"Evaluation inventory mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    return sorted(expected)


def _accounting_auditor():
    path = Path(__file__).with_name("audit_research_v3.py")
    spec = importlib.util.spec_from_file_location("v4_accounting_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit_ledger


def audit_run(data_dir, run_dir, *, validation_only=False):
    """Require every declared artifact and verify both learned and rule choices."""
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    manifest_path, frozen_path = run_dir / "manifest.json", run_dir / "frozen_selection.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.exists() else None
    if not validation_only and frozen is None:
        raise ValueError("Frozen selection is required before evaluation audit")
    if frozen is not None and frozen["manifest_sha256"] != digest(manifest_path):
        raise ValueError("Frozen manifest hash mismatch")
    project = Path(__file__).resolve().parents[1]
    identity = manifest["identity"]
    for name, checksum in identity["data"].items():
        if digest(data_dir / name) != checksum:
            raise ValueError(f"Data hash mismatch: {name}")
    for name, checksum in identity["sources"].items():
        path = project / ("docs" if name.endswith(".md") else "src/rlasset") / name
        if digest(path) != checksum:
            raise ValueError(f"Source hash mismatch: {name}")
    prices = pd.read_csv(data_dir / "ashare_broad_raw.csv", dtype={"tic": str})
    dividends = pd.read_csv(data_dir / "cash_actions.csv", dtype={"ticker": str})
    splits = pd.read_csv(data_dir / "splits.csv", dtype={"ticker": str})
    reference = {"signal": independent_signal(prices, dividends, splits), "covariance": {}}
    dates, tickers, _ = reference["signal"]
    if manifest["sessions"] != len(dates) or manifest["tickers"] != tickers:
        raise ValueError("Manifest market dimensions changed")
    candidates, seeds, baselines = identity["candidates"], identity["seeds"], identity["baselines"]
    rule_names = ("vol_target10", "invvol_target10")
    results, models, selections = {}, {}, {}
    audit_account = _accounting_auditor()

    def ledger_audit(path, fold, period, candidate, cost=5):
        ledger = pd.read_csv(path)
        account = audit_account(ledger, prices, dividends, splits, cost_bps=cost)
        if [account["first_decision"], account["last_valuation"]] != fold["dates"][period]:
            raise ValueError("Ledger dates differ from frozen manifest")
        for index, row in enumerate(ledger.iloc[1:].itertuples(index=False), start=1):
            should_trade = index == 1 if candidate == "buy_hold" else (index - 1) % 5 == 0
            if (row.decision_phase != (index - 1) % 5 or row.rebalance_phase != index % 5
                    or row.rebalance_executed != should_trade):
                raise ValueError("Rebalance schedule differs from the five-session protocol")
        risk_enabled = candidate.startswith(("risk_", "joint_")) or candidate in baselines
        risk = {"requests": 0, "executed_requests": 0, "max_absolute_error": 0.}
        if risk_enabled:
            risk = audit_risk_targets(ledger, prices, dividends, splits,
                budget_bounds=(.05, .2) if candidate in candidates else None,
                rule=candidate if candidate in rule_names else None, reference=reference)
        results[str(path)] = {**account, "risk": risk}
        nav = ledger.portfolio_value.to_numpy(float)
        metrics = {"total_return": float(nav[-1] / nav[0] - 1),
                   "max_drawdown": float(np.max(1 - nav / np.maximum.accumulate(nav))),
                   "return_days": len(ledger) - 1,
                   "mean_cash_weight": float(np.mean([vector(row)[0] for row in ledger.weights.iloc[1:]]))}
        return metrics

    def load_record(path, ledger_path, candidate, fold, *, seed=None):
        record = json.loads(path.read_text(encoding="utf-8"))
        verify_record_seal(record)
        if record["candidate"] != candidate or record["fold"] != fold["name"]:
            raise ValueError("Training/validation record identity mismatch")
        if record["validation_sha256"] != digest(ledger_path):
            raise ValueError("Validation ledger hash mismatch")
        if seed is None:
            if record["kind"] != "rule":
                raise ValueError("Invalid rule validation identity")
        else:
            requested = identity["steps"]
            rollout = next((value for value in range(min(100, requested), 1, -1)
                            if requested % value == 0), 64) if candidate.endswith("_ppo") else 1
            expected = math.ceil(requested / rollout) * rollout
            if (record["seed"] != seed or record["requested_steps"] != requested
                    or record["actual_steps"] != expected):
                raise ValueError("Training record seed or step budget mismatch")
            model_path = path.with_suffix(".zip")
            if record["model_sha256"] != digest(model_path):
                raise ValueError("Model hash mismatch")
            models[str(model_path)] = record["model_sha256"]
        values = ledger_audit(ledger_path, fold, "validation", candidate)
        for field, value in values.items():
            if not np.isclose(record["validation_metrics"][field], value, rtol=1e-10, atol=1e-10):
                raise ValueError(f"Validation metric {field} differs from independent ledger")
        return record, values

    for fold in manifest["folds"]:
        name, learned, rules = fold["name"], {}, {}
        selected = frozen["folds"][name] if frozen is not None else None
        expected_keys = {f"{seed}/{candidate}" for seed in seeds for candidate in candidates}
        if selected is not None:
            for field in ("model_hashes", "validation_hashes", "record_hashes"):
                if set(selected[field]) != expected_keys:
                    raise ValueError("Frozen artifact inventory differs from registered candidates")
            for field in ("baseline_validation_hashes", "baseline_record_hashes"):
                if set(selected[field]) != set(rule_names):
                    raise ValueError("Frozen rule artifact inventory mismatch")
        for candidate in candidates:
            learned[candidate] = []
            for seed in seeds:
                directory = run_dir / name / f"seed_{seed}"
                path, ledger_path = directory / f"{candidate}.json", directory / f"validation_{candidate}.csv"
                record, values = load_record(path, ledger_path, candidate, fold, seed=seed)
                learned[candidate].append(values)
                if selected is not None:
                    key = f"{seed}/{candidate}"
                    for field, actual in (("model_hashes", record["model_sha256"]),
                                          ("validation_hashes", digest(ledger_path)), ("record_hashes", digest(path))):
                        if selected[field][key] != actual:
                            raise ValueError(f"Frozen {field} mismatch")
        rule_directory = run_dir / name / "validation_baselines"
        if selected is not None or rule_directory.exists():
            for candidate in rule_names:
                path, ledger_path = rule_directory / f"{candidate}.json", rule_directory / f"{candidate}.csv"
                _, values = load_record(path, ledger_path, candidate, fold)
                rules[candidate] = values
                if selected is not None:
                    if (selected["baseline_validation_hashes"][candidate] != digest(ledger_path)
                            or selected["baseline_record_hashes"][candidate] != digest(path)):
                        raise ValueError("Frozen rule validation hash mismatch")
        if selected is not None:
            selections[name] = verify_selection(selected, learned, rules)
        if not validation_only:
            for cost in identity["cost_bps"]:
                directory = run_dir / f"evaluation_{cost:g}bps" / name
                filenames = verify_evaluation_inventory(directory, candidates, seeds, baselines)
                for filename in filenames:
                    candidate = filename[:-4]
                    if candidate.startswith("seed_"):
                        candidate = candidate.split("_", 2)[2]
                    ledger_audit(directory / filename, fold, "test", candidate, cost)
    return {"verified_utc": datetime.now(timezone.utc).isoformat(), "validation_only": validation_only,
            "selection_verified": frozen is not None, "selections": selections,
            "ledgers": len(results), "transitions": sum(row["days"] for row in results.values()),
            "max_absolute_error": max((row["max_absolute_error"] for row in results.values()), default=0.),
            "risk_requests": sum(row["risk"]["requests"] for row in results.values()),
            "risk_executed_requests": sum(row["risk"]["executed_requests"] for row in results.values()),
            "risk_max_absolute_error": max((row["risk"]["max_absolute_error"] for row in results.values()), default=0.),
            "model_count": len(models), "model_sha256": models, "ledger_results": results,
            "interpretation": "Independent accounting, causal risk-target diagnostics, fixed-rule targets and validation-only selection. This checks proposed targets, not original actions at the full-exposure cap; it does not establish predictive alpha or attainable realized volatility/fills."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/v3")
    parser.add_argument("--run", default="runs/research_v4")
    parser.add_argument("--output", default="runs/verification_v4.json")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    result = audit_run(args.data_dir, args.run, validation_only=args.validation_only)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "ledgers", "transitions", "max_absolute_error", "risk_requests", "risk_max_absolute_error", "model_count")}))


if __name__ == "__main__":
    main()
