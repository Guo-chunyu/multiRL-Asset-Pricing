"""Independent v4 risk diagnostics must expose corrupted or future-driven targets."""
import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def auditor():
    path = Path(__file__).parents[1] / "scripts" / "audit_research_v4.py"
    spec = importlib.util.spec_from_file_location("v4_independent_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def risk_fixture():
    dates = pd.bdate_range("2020-01-01", periods=130)
    log_returns = np.where(np.arange(130) % 2, .01, -.01)
    prices = 100 * np.exp(np.cumsum(log_returns))
    frame = pd.DataFrame([
        {"date": date, "tic": ticker, "open": price, "close": price,
         "high": price, "low": price, "volume": 1000.}
        for date, price in zip(dates, prices) for ticker in ("AAA", "BBB")
    ])
    # Identical risky returns and 25% diagonal shrinkage: variance = .875 * s².
    sigma = np.sqrt(252 * .875 * (20 / 19) * .01 ** 2)
    exposure = .1 / sigma
    target = [1 - exposure, .5 * exposure, .5 * exposure]
    ledger = pd.DataFrame([
        {"date": dates[120], "decision_date": dates[120], "target_weights": [1, 0, 0],
         "rebalance_executed": False, "requested_exposure": np.nan,
         "estimated_risky_vol": np.nan, "requested_target_vol": np.nan,
         "requested_risky_mix": None},
        {"date": dates[121], "decision_date": dates[120], "target_weights": target,
         "rebalance_executed": True, "requested_exposure": exposure,
         "estimated_risky_vol": sigma, "requested_target_vol": .1,
         "requested_risky_mix": [.5, .5]},
    ])
    dividends = pd.DataFrame(columns=["ticker", "ex_date", "pay_date", "cash_per_share"])
    splits = pd.DataFrame(columns=["ticker", "effective_date", "ratio"])
    return ledger, frame, dividends, splits


def test_reconstructs_causal_log_covariance_and_risk_target_independently():
    module = auditor()
    ledger, prices, dividends, splits = risk_fixture()
    result = module.audit_risk_targets(ledger, prices, dividends, splits, budget_bounds=(.05, .2))
    assert result["requests"] == 1
    assert result["executed_requests"] == 1
    assert result["max_absolute_error"] < 1e-12


@pytest.mark.parametrize("column", ["requested_exposure", "estimated_risky_vol", "requested_target_vol"])
def test_corrupted_risk_diagnostic_cannot_pass_accounting_only(column):
    module = auditor()
    ledger, prices, dividends, splits = risk_fixture()
    ledger.loc[1, column] += .02
    with pytest.raises(ValueError, match="risk|exposure|target"):
        module.audit_risk_targets(ledger, prices, dividends, splits)


def test_risky_mix_cap_and_final_target_must_match_the_request():
    module = auditor()
    ledger, prices, dividends, splits = risk_fixture()
    ledger.at[1, "requested_risky_mix"] = [.6, .4]
    with pytest.raises(ValueError, match="mix|cap"):
        module.audit_risk_targets(ledger, prices, dividends, splits)
    ledger, prices, dividends, splits = risk_fixture()
    ledger.at[1, "target_weights"] = [1., 0., 0.]
    with pytest.raises(ValueError, match="target"):
        module.audit_risk_targets(ledger, prices, dividends, splits)


def test_hold_day_requests_are_audited_without_confusing_them_with_actual_hold_weights():
    module = auditor()
    ledger, prices, dividends, splits = risk_fixture()
    ledger.loc[1, "rebalance_executed"] = False
    ledger.at[1, "target_weights"] = [1., 0., 0.]
    result = module.audit_risk_targets(ledger, prices, dividends, splits)
    assert result["requests"] == 1
    assert result["executed_requests"] == 0


def test_future_prices_do_not_enter_risk_diagnostics():
    module = auditor()
    ledger, prices, dividends, splits = risk_fixture()
    later = pd.to_datetime(prices.date) > pd.Timestamp(ledger.loc[1, "decision_date"])
    prices.loc[later, ["open", "close", "high", "low"]] *= 40
    assert module.audit_risk_targets(ledger, prices, dividends, splits)["requests"] == 1


def test_independent_signal_removes_split_and_dividend_price_jumps():
    module = auditor()
    _, prices, _, _ = risk_fixture()
    dates = pd.DatetimeIndex(sorted(prices.date.unique()))
    for column in ("open", "close", "high", "low"):
        prices[column] = np.where(prices.date < dates[60], 100.,
                                  np.where(prices.date < dates[90], 50., 40.))
    dividends = pd.DataFrame([
        {"ticker": ticker, "ex_date": dates[90], "pay_date": dates[92], "cash_per_share": 10.}
        for ticker in ("AAA", "BBB")
    ])
    splits = pd.DataFrame([
        {"ticker": ticker, "effective_date": dates[60], "ratio": 2.}
        for ticker in ("AAA", "BBB")
    ])
    _, tickers, signal = module.independent_signal(prices, dividends, splits)
    assert tickers == ["AAA", "BBB"]
    np.testing.assert_allclose(signal, 100.)


def test_missing_risk_fields_and_out_of_budget_requests_are_rejected():
    module = auditor()
    ledger, prices, dividends, splits = risk_fixture()
    with pytest.raises(ValueError, match="required|missing"):
        module.audit_risk_targets(ledger.drop(columns="requested_exposure"), prices, dividends, splits)
    with pytest.raises(ValueError, match="budget"):
        module.audit_risk_targets(ledger, prices, dividends, splits, budget_bounds=(.15, .2))


def test_internal_record_seal_detects_metric_change_even_before_external_freeze(tmp_path):
    module = auditor()
    record = {"candidate": "base_ppo", "validation_metrics": {"total_return": .1}}
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    record["record_sha256"] = hashlib.sha256(payload).hexdigest()
    module.verify_record_seal(record)
    record["validation_metrics"]["total_return"] = .2
    with pytest.raises(ValueError, match="record"):
        module.verify_record_seal(record)


def test_independent_selection_allows_rule_to_replace_a_weaker_rl_winner():
    module = auditor()
    learned = {"base_ppo": [{"total_return": .15, "max_drawdown": .1}] * 3,
               "joint_ppo": [{"total_return": .14, "max_drawdown": .02}] * 3}
    rules = {"vol_target10": {"total_return": .16, "max_drawdown": .02},
             "invvol_target10": {"total_return": .13, "max_drawdown": .03}}
    frozen = {"candidate": "joint_ppo", "scores": {"base_ppo": .10, "joint_ppo": .13},
              "deploy_candidate": "vol_target10", "deploy_kind": "rule",
              "deploy_scores": {"base_ppo": .10, "joint_ppo": .13,
                                "vol_target10": .15, "invvol_target10": .115},
              "selected_on": "validation",
              "criterion": "mean_seed(total_return - 0.5 * max_drawdown)"}
    result = module.verify_selection(frozen, learned, rules)
    assert result["candidate"] == "joint_ppo"
    assert result["deploy_candidate"] == "vol_target10"
    frozen["deploy_candidate"] = "joint_ppo"
    with pytest.raises(ValueError, match="selection"):
        module.verify_selection(frozen, learned, rules)


def test_incomplete_evaluation_inventory_does_not_silently_pass(tmp_path):
    module = auditor()
    for name in ("seed_7_base_ppo.csv", "seed_7_joint_mixture.csv", "equal_50.csv"):
        (tmp_path / name).write_text("placeholder", encoding="utf-8")
    module.verify_evaluation_inventory(tmp_path, ["base_ppo"], [7], ["equal_50"])
    (tmp_path / "seed_7_base_ppo.csv").unlink()
    with pytest.raises(ValueError, match="inventory|missing"):
        module.verify_evaluation_inventory(tmp_path, ["base_ppo"], [7], ["equal_50"])


def test_fixed_rule_cannot_silently_use_a_different_volatility_budget():
    module = auditor()
    ledger, prices, dividends, splits = risk_fixture()
    module.audit_risk_targets(ledger, prices, dividends, splits, rule="vol_target10")
    exposure = ledger.loc[1, "requested_exposure"] * .8
    ledger.loc[1, "requested_exposure"] = exposure
    ledger.loc[1, "requested_target_vol"] = .08
    ledger.at[1, "target_weights"] = [1 - exposure, .5 * exposure, .5 * exposure]
    with pytest.raises(ValueError, match="rule"):
        module.audit_risk_targets(ledger, prices, dividends, splits, rule="vol_target10")


def complete_run_fixture(tmp_path):
    """Small real-environment run with opaque checkpoint bytes for hash tests."""
    from rlasset.corporate_actions import CorporateActions
    from rlasset.data import MarketPanel
    from rlasset.study_v3 import run_policy, save_ledger
    from rlasset.study_v4 import (BASELINES, CANDIDATES, RULE_CANDIDATES, baseline_policy,
                                 choose_candidates, make_env, signed_record, write_json)
    module = auditor()
    data_dir, run_dir = tmp_path / "data", tmp_path / "run"
    data_dir.mkdir()
    run_dir.mkdir()
    _, prices, dividends, splits = risk_fixture()
    for frame, name in ((prices, "ashare_broad_raw.csv"), (dividends, "cash_actions.csv"),
                        (splits, "splits.csv")):
        frame.to_csv(data_dir / name, index=False)
    panel = MarketPanel.from_frame(prices)
    actions = CorporateActions.empty(panel)
    fold = {"name": "fold_01", "train": [120, 121], "validation": [122, 125], "test": [126, 129]}
    fold["dates"] = {name: [str(panel.dates[a].date()), str(panel.dates[b].date())]
                     for name, (a, b) in ((key, fold[key]) for key in ("train", "validation", "test"))}
    manifest = {"identity": {"data": {path.name: module.digest(path) for path in data_dir.iterdir()},
                             "sources": {}, "candidates": list(CANDIDATES), "seeds": [7],
                             "baselines": list(BASELINES), "cost_bps": [5], "steps": 100},
                "folds": [fold], "sessions": len(panel.dates), "tickers": list(panel.tickers)}
    write_json(run_dir / "manifest.json", manifest)
    folder = run_dir / "fold_01" / "seed_7"
    folder.mkdir(parents=True)
    learned, models, validations, records = {}, {}, {}, {}
    for candidate in CANDIDATES:
        env = make_env(panel, actions, fold["validation"], candidate)
        ledger, metrics = run_policy(env, lambda env, obs: env.action_to_weights(np.zeros(3)))
        ledger_path, model_path = folder / f"validation_{candidate}.csv", folder / f"{candidate}.zip"
        save_ledger(ledger_path, ledger)
        model_path.write_bytes(b"opaque test checkpoint")
        record = signed_record({"candidate": candidate, "seed": 7, "fold": "fold_01",
                                "requested_steps": 100, "actual_steps": 100,
                                "model_sha256": module.digest(model_path),
                                "validation_sha256": module.digest(ledger_path), "validation_metrics": metrics})
        path = folder / f"{candidate}.json"
        write_json(path, record)
        key = f"7/{candidate}"
        learned[candidate] = [metrics]
        models[key], validations[key], records[key] = module.digest(model_path), module.digest(ledger_path), module.digest(path)
    rule_dir = run_dir / "fold_01" / "validation_baselines"
    rule_dir.mkdir()
    rules, rule_validations, rule_records = {}, {}, {}
    for name in RULE_CANDIDATES:
        ledger, metrics = run_policy(make_env(panel, actions, fold["validation"], "joint_ppo"), baseline_policy(name))
        ledger_path, path = rule_dir / f"{name}.csv", rule_dir / f"{name}.json"
        save_ledger(ledger_path, ledger)
        write_json(path, signed_record({"candidate": name, "fold": "fold_01", "kind": "rule",
                                       "validation_sha256": module.digest(ledger_path), "validation_metrics": metrics}))
        rules[name], rule_validations[name], rule_records[name] = metrics, module.digest(ledger_path), module.digest(path)
    # choose_candidates requires the registered three seeds; identical replicas
    # supply the same mathematical means, while this fixture stores just one.
    choice = choose_candidates({name: rows * 3 for name, rows in learned.items()}, rules)
    frozen_fold = {**choice, "model_hashes": models, "validation_hashes": validations,
                   "record_hashes": records, "baseline_validation_hashes": rule_validations,
                   "baseline_record_hashes": rule_records}
    write_json(run_dir / "frozen_selection.json", {"manifest_sha256": module.digest(run_dir / "manifest.json"),
                                                  "folds": {"fold_01": frozen_fold}})
    test_dir = run_dir / "evaluation_5bps" / "fold_01"
    test_dir.mkdir(parents=True)
    for candidate in (*CANDIDATES, "joint_mixture"):
        env = make_env(panel, actions, fold["test"], "joint_ppo" if candidate == "joint_mixture" else candidate)
        ledger, _ = run_policy(env, lambda env, obs: env.action_to_weights(np.zeros(3)))
        save_ledger(test_dir / f"seed_7_{candidate}.csv", ledger)
    for name in BASELINES:
        ledger, _ = run_policy(make_env(panel, actions, fold["test"], "joint_ppo"), baseline_policy(name))
        save_ledger(test_dir / f"{name}.csv", ledger)
    return data_dir, run_dir


def test_complete_run_checks_every_expected_ledger_and_frozen_rule_choice(tmp_path):
    module = auditor()
    data_dir, run_dir = complete_run_fixture(tmp_path)
    result = module.audit_run(data_dir, run_dir)
    assert result["ledgers"] == 21
    assert result["model_count"] == 6
    assert result["selection_verified"] is True
    assert result["risk_requests"] > 0
    assert result["max_absolute_error"] < 1e-10
    assert result["risk_max_absolute_error"] < 1e-10


@pytest.mark.parametrize("kind", ["rule_ledger", "checkpoint", "manifest", "deploy_choice", "missing_test", "rebalance_phase"])
def test_run_rejects_tampering_and_incomplete_evaluation(tmp_path, kind):
    module = auditor()
    data_dir, run_dir = complete_run_fixture(tmp_path)
    if kind == "rule_ledger":
        path = run_dir / "fold_01" / "validation_baselines" / "vol_target10.csv"
        path.write_bytes(path.read_bytes() + b"\n")
    elif kind == "checkpoint":
        (run_dir / "fold_01" / "seed_7" / "base_ppo.zip").write_bytes(b"changed")
    elif kind == "manifest":
        path = run_dir / "manifest.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif kind == "deploy_choice":
        path = run_dir / "frozen_selection.json"
        frozen = json.loads(path.read_text(encoding="utf-8"))
        frozen["folds"]["fold_01"]["deploy_candidate"] = "nonexistent"
        path.write_text(json.dumps(frozen), encoding="utf-8")
    elif kind == "missing_test":
        (run_dir / "evaluation_5bps" / "fold_01" / "seed_7_base_ppo.csv").unlink()
    else:
        path = run_dir / "evaluation_5bps" / "fold_01" / "seed_7_base_ppo.csv"
        ledger = pd.read_csv(path)
        ledger.loc[2, "rebalance_executed"] = True
        ledger.to_csv(path, index=False)
    with pytest.raises(ValueError):
        module.audit_run(data_dir, run_dir)
