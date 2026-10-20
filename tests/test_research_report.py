"""Report aggregation keeps market dates, training seeds and folds distinct."""
import json

import numpy as np
import pandas as pd
import pytest


def _json(path, content):
    path.write_text(json.dumps(content), encoding="utf-8")


def _ledger(path, dates, returns, turnover=.1):
    returns = np.r_[0., returns]
    pd.DataFrame({
        "date": dates,
        "decision_date": [dates[0], *dates[:-1]],
        "net_return": returns,
        "portfolio_value": np.cumprod(1 + returns),
        "turnover": [0., *np.full(len(returns) - 1, turnover)],
    }).to_csv(path, index=False)


def _run(root, start="2025-01-01"):
    root.mkdir(parents=True)
    all_dates = pd.bdate_range(start, periods=42)
    val_dates, dates = all_dates[:21], all_dates[21:]
    splits = {}
    for name, index in [("validation", val_dates), ("test", dates)]:
        splits[name] = {"first_decision": index[0].isoformat(),
                        "last_valuation": index[-1].isoformat(), "transitions": 20}
    _json(root / "manifest.json", {
        "status": "complete", "data_sha256": "test-data-hash",
        "config": {"seeds": [7, 17], "algorithms": ["td3", "sac"],
                   "risk_levels": [2.], "cost_bps": 5., "total_timesteps": 10000},
        "splits": splits,
    })
    for seed, td3_return in [(7, .01), (17, -.005)]:
        directory = root / f"seed_{seed}"
        directory.mkdir()
        paths = {
            "td3_risk_2": np.full(20, td3_return),
            "sac": np.full(20, .002),
            "equal_weight": np.zeros(20),
            "cash": np.zeros(20),
            "dqn_selector": np.tile([.02, -.02], 10),
            "equal_mixture": np.full(20, .001),
        }
        for name, returns in paths.items():
            _ledger(directory / f"test_{name}.csv", dates, returns)
        # The report must reconstruct aliases from the validation choice; this
        # deliberately stale alias looks far better in test than either model.
        _ledger(directory / "test_best_rl_on_validation.csv", dates, np.full(20, .2))
        _json(directory / "selection.json", {
            "selected_on": "validation", "strategy": "cash",
            "best_rl_on_validation": "sac",
            "criterion": "total_return - 0.5 * max_drawdown",
        })
        _json(directory / "models.json", {
            "td3_risk_2": {"algorithm": "td3", "training_wall_seconds": float(seed),
                           "actual_steps": 10000, "requested_steps": 10000},
            "sac": {"algorithm": "sac", "training_wall_seconds": float(seed * 2),
                    "actual_steps": 10000, "requested_steps": 10000},
        })
        _json(directory / "selector.json", {
            "algorithm": "dqn", "training_wall_seconds": 2.,
            "actual_steps": 10000, "requested_steps": 10000,
        })
        # TD3 wins the average validation score, although SAC is the deliberately
        # retained per-seed choice. Never choose the highest test seed instead.
        _ledger(directory / "validation_td3_risk_2.csv", val_dates, np.full(20, .003))
        _ledger(directory / "validation_sac.csv", val_dates, np.full(20, .001))
    return root


def test_mean_seed_metrics_initial_row_and_validation_selected_aliases(tmp_path):
    from rlasset.research_report import analyze_run

    result = analyze_run(_run(tmp_path / "run"), n_bootstrap=50)
    td3 = result["strategies"]["td3_risk_2"]
    seed_totals = np.array([1.01 ** 20 - 1, .995 ** 20 - 1])
    assert result["n_observations"] == 20
    assert td3["comparison"]["n_observations"] == 20  # not seeds * dates
    assert td3["mean_total_return"] == pytest.approx(seed_totals.mean())
    assert td3["std_total_return"] == pytest.approx(seed_totals.std(ddof=1))
    assert td3["mean_max_drawdown"] == pytest.approx((1 - .995 ** 20) / 2)
    assert td3["mean_turnover"] == pytest.approx(.1)  # exclude initial reset row
    assert td3["mean_equity"][-1] == pytest.approx(1 + seed_totals.mean())
    assert td3["total_return_of_mean_daily_returns"] == pytest.approx(1.0025 ** 20 - 1)
    assert result["strategies"]["best_rl_on_validation"]["mean_total_return"] == pytest.approx(1.002 ** 20 - 1)
    assert result["strategies"]["selected_on_validation"]["mean_total_return"] == 0
    assert result["strategies"]["best_rl_on_validation"]["source_by_seed"] == {"7": "sac", "17": "sac"}
    assert result["study_status"] == "retrospective_exploratory"
    assert result["training_costs"]["td3_risk_2"]["mean_wall_seconds"] == 12
    json.dumps(result, allow_nan=False)


def test_consensus_algorithm_uses_average_validation_score_and_all_test_seeds(tmp_path):
    from rlasset.research_report import analyze_run

    run = _run(tmp_path / "run")
    # Give SAC a much better test outcome without changing its validation.
    # The consensus must remain TD3 and must use both TD3 seeds.
    for seed in [7, 17]:
        path = run / f"seed_{seed}/test_sac.csv"
        dates = pd.to_datetime(pd.read_csv(path).date)
        _ledger(path, dates, np.full(20, .02))
    result = analyze_run(run, n_bootstrap=50)
    ranking = result["validation_algorithm_ranking"]
    assert [item["strategy"] for item in ranking] == ["td3_risk_2", "sac"]
    assert ranking[0]["mean_validation_score"] == pytest.approx(1.003 ** 20 - 1)
    assert result["consensus_validation_algorithm"] == "td3_risk_2"
    consensus = result["strategies"]["consensus_validation_algorithm"]
    assert consensus["source_by_seed"] == {"7": "td3_risk_2", "17": "td3_risk_2"}
    assert consensus["mean_total_return"] == pytest.approx((1.01 ** 20 + .995 ** 20) / 2 - 1)
    assert result["strategies"]["sac"]["mean_total_return"] > consensus["mean_total_return"]


@pytest.mark.parametrize("damage,match", [
    ("different_dates", "date"),
    ("duplicate_dates", "date"),
    ("missing_strategy", "strateg"),
    ("incomplete_run", "complete"),
    ("wrong_nav", "portfolio_value"),
    ("missing_initial_row", "initial|row|length"),
    ("different_benchmark", "benchmark|equal_weight"),
    ("selection_from_test", "validation"),
    ("validation_dates", "date"),
])
def test_report_rejects_incomparable_or_corrupt_inputs(tmp_path, damage, match):
    from rlasset.research_report import analyze_run

    run = _run(tmp_path / "run")
    target = run / "seed_17" / "test_td3_risk_2.csv"
    frame = pd.read_csv(target)
    if damage == "different_dates":
        frame.loc[5, "date"] = "2025-02-09"
        frame.to_csv(target, index=False)
    elif damage == "duplicate_dates":
        frame.loc[5, "date"] = frame.loc[4, "date"]
        frame.to_csv(target, index=False)
    elif damage == "missing_strategy":
        target.unlink()
    elif damage == "incomplete_run":
        manifest = json.loads((run / "manifest.json").read_text())
        manifest["status"] = "running"
        _json(run / "manifest.json", manifest)
    elif damage == "wrong_nav":
        frame.loc[8, "portfolio_value"] = 100
        frame.to_csv(target, index=False)
    elif damage == "missing_initial_row":
        frame.iloc[1:].to_csv(target, index=False)
    elif damage == "different_benchmark":
        _ledger(run / "seed_17/test_equal_weight.csv", pd.to_datetime(frame.date), np.full(20, .001))
    elif damage == "selection_from_test":
        path = run / "seed_17/selection.json"
        selection = json.loads(path.read_text())
        selection["selected_on"] = "test"
        _json(path, selection)
    elif damage == "validation_dates":
        path = run / "seed_17/validation_sac.csv"
        frame = pd.read_csv(path)
        frame.loc[5, "date"] = frame.loc[4, "date"]
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        analyze_run(run, n_bootstrap=20)


def test_report_writes_independent_fold_results_and_readable_artifacts(tmp_path):
    from rlasset.research_report import build_research_report

    run = _run(tmp_path / "run")
    walk = tmp_path / "wf"
    _run(walk / "fold_01", "2024-01-01")
    _run(walk / "fold_02", "2024-06-01")
    _json(walk / "walkforward_summary.json", {"fold_count": 2, "data_sha256": "test-data-hash"})
    output = tmp_path / "report"
    result = build_research_report(run, output, walk_forward=walk, n_bootstrap=30)
    saved = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    assert saved == result
    assert len(saved["walk_forward"]["folds"]) == 2
    assert saved["walk_forward"]["continuous_equity_constructed"] is False
    for fold in saved["walk_forward"]["folds"]:
        assert fold["strategies"]["sac"]["comparison"]["n_observations"] == 20
    for filename in ["comparison.png", "equity.png"]:
        assert (output / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "回顾性" in report and "包含零" in report
    assert "consensus_validation_algorithm" in report
    assert "不拼接" in report
    assert saved["main"]["strategies"]["sac"]["comparison"]["ci_contains_zero"] is False
    assert saved["main"]["strategies"]["equal_weight"]["comparison"]["ci_contains_zero"] is True


def test_walk_forward_rejects_overlapping_test_dates(tmp_path):
    from rlasset.research_report import build_research_report

    run = _run(tmp_path / "run")
    walk = tmp_path / "wf"
    _run(walk / "fold_01")
    _run(walk / "fold_02")
    _json(walk / "walkforward_summary.json", {"fold_count": 2, "data_sha256": "test-data-hash"})
    with pytest.raises(ValueError, match="overlap"):
        build_research_report(run, tmp_path / "report", walk_forward=walk, n_bootstrap=20)
