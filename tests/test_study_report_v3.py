"""Synthetic-only reporting tests: no market/test datasets are read here."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from rlasset.metrics import performance_metrics


def reporter():
    from rlasset import study_report_v3
    return study_report_v3


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_run(root, seeds=(7, 17, 29), same_seeds=False, fold_count=1):
    candidates = ["legacy_ppo", "robust_ppo", "robust_sac", "robust_td3"]
    manifest = {"identity": {"seeds": list(seeds), "candidates": candidates, "steps": 20},
                "folds": [], "tickers": ["510050", "510300", "510500"],
                "claim": "Synthetic fixture"}
    frozen = {"folds": {}}
    summary = {"cost_bps": 5, "folds": {}}
    for i in range(fold_count):
        fold = f"fold_{i + 1:02}"
        dates = pd.bdate_range(f"{2019 + 2 * i}-01-02", periods=21)
        manifest["folds"].append({"name": fold, "test": [100 + i * 21, 120 + i * 21],
                                  "dates": {"test": [str(dates[0].date()), str(dates[-1].date())]}})
        frozen["folds"][fold] = {"candidate": "robust_sac", "selected_on": "validation",
                                  "criterion": "mean_seed(total_return - 0.5 * max_drawdown)",
                                  "scores": {c: (1 if c == "robust_sac" else 0) for c in candidates}}
        folder = root / "evaluation_5bps" / fold
        folder.mkdir(parents=True)

        def ledger(name, returns, cash=.1):
            values = np.r_[1., np.cumprod(1 + returns)]
            frame = pd.DataFrame({"date": dates, "net_return": np.r_[0., returns],
                                  "portfolio_value": values, "turnover": np.r_[0., np.full(20, .2)],
                                  "weights": [json.dumps([1., 0., 0., 0.])] +
                                             [json.dumps([cash, (1-cash)/3, (1-cash)/3, (1-cash)/3])] * 20})
            frame.to_csv(folder / f"{name}.csv", index=False)
            metrics = performance_metrics(returns, np.full(20, .2))
            return {**metrics, "mean_cash_weight": cash, "return_days": 20}

        records = {}
        for j, seed in enumerate(seeds):
            returns = np.r_[.1, -.1, np.zeros(18)] * (1 if same_seeds else j + 1)
            records[str(seed)] = {c: ledger(f"seed_{seed}_{c}", returns) for c in candidates + ["robust_mixture"]}
            records[str(seed)]["selected_on_validation"] = records[str(seed)]["robust_sac"]
        baselines = {name: ledger(name, np.zeros(20), cash)
                     for name, cash in [("equal_80", .2), ("equal_90", .1), ("equal_100", 0.),
                                        ("buy_hold", 0.), ("inverse_vol_90", .1)]}
        summary["folds"][fold] = {"selected": "robust_sac", "seeds": records, "baselines": baselines}
    put_json(root / "manifest.json", manifest)
    put_json(root / "frozen_selection.json", frozen)
    put_json(root / "evaluation_5bps" / "summary.json", summary)
    return root


def test_manual_metrics_keep_seed_dispersion_separate_from_market_dates(tmp_path):
    result = reporter().analyze_run(fixture_run(tmp_path), n_bootstrap=30)
    fold = result["evaluations"]["5"]["folds"]["fold_01"]
    row = fold["strategies"]["robust_ppo"]
    # Each two-day product is .99, .96, .91; the following 18 days are flat.
    assert row["metrics"]["total_return"]["mean"] == pytest.approx(-.0466666666666667)
    assert row["metrics"]["total_return"]["std"] == pytest.approx(.0404145188432738)
    assert row["metrics"]["max_drawdown"]["mean"] == pytest.approx(.2)
    assert row["metrics"]["sharpe"]["mean"] == pytest.approx(0.)
    assert row["metrics"]["annual_return"]["mean"] == pytest.approx(
        ((.99 ** 12.6 - 1) + (.96 ** 12.6 - 1) + (.91 ** 12.6 - 1)) / 3)
    assert row["metrics"]["mean_turnover"]["mean"] == pytest.approx(.2)
    assert row["metrics"]["mean_cash_weight"]["mean"] == pytest.approx(.1)
    assert row["comparisons"]["equal_90"]["comparison"]["n_observations"] == 20
    assert row["mean_nav"][-1] == pytest.approx(.9533333333333333)
    assert row["comparisons"]["equal_90"]["total_return_of_mean_daily_returns"] == pytest.approx(-.04)
    assert fold["baselines"]["equal_90"]["n_paths"] == 1
    assert fold["baselines"]["equal_90"]["metrics"]["total_return"]["std"] is None
    assert fold["strategies"]["selected_on_validation"]["source_candidate"] == "robust_sac"


def test_repeating_identical_training_seed_outcomes_does_not_add_market_observations(tmp_path):
    one = reporter().analyze_run(fixture_run(tmp_path / "one", seeds=(7,), same_seeds=True), n_bootstrap=50)
    three = reporter().analyze_run(fixture_run(tmp_path / "three", same_seeds=True), n_bootstrap=50)
    def comparison(data):
        return data["evaluations"]["5"]["folds"]["fold_01"]["strategies"]["legacy_ppo"]["comparisons"]["equal_100"]["comparison"]
    a, b = comparison(one), comparison(three)
    assert a["n_observations"] == b["n_observations"] == 20
    assert a["annualized_mean_excess_ci"] == pytest.approx(b["annualized_mean_excess_ci"])


@pytest.mark.parametrize("damage", ["missing_date", "all_missing_date", "shifted_date", "nav", "summary", "selection", "weights"])
def test_corrupted_or_misaligned_research_is_rejected(tmp_path, damage):
    root = fixture_run(tmp_path)
    folder = root / "evaluation_5bps" / "fold_01"
    if damage in {"missing_date", "all_missing_date", "shifted_date", "nav", "weights"}:
        paths = list(folder.glob("*.csv")) if damage == "all_missing_date" else [folder / "seed_7_legacy_ppo.csv"]
        for path in paths:
            frame = pd.read_csv(path)
            if damage in {"missing_date", "all_missing_date"}:
                frame = frame.drop(index=5)
            elif damage == "shifted_date":
                frame.loc[5, "date"] = "2019-01-08 12:00:00"
            elif damage == "nav":
                frame.loc[5, "portfolio_value"] = 3.
            else:
                frame.loc[5, "weights"] = "[2, 0, 0, 0]"
            frame.to_csv(path, index=False)
    else:
        path = root / "evaluation_5bps" / "summary.json"
        summary = json.loads(path.read_text())
        if damage == "summary":
            summary["folds"]["fold_01"]["seeds"]["7"]["legacy_ppo"]["total_return"] = 7.
        else:
            summary["folds"]["fold_01"]["selected"] = "legacy_ppo"
        put_json(path, summary)
    with pytest.raises(ValueError):
        reporter().analyze_run(root, n_bootstrap=20)


def test_build_report_emits_separate_fold_analysis_and_readable_artifacts(tmp_path):
    root = fixture_run(tmp_path, fold_count=4)
    result = reporter().build_report(root, n_bootstrap=20)
    assert len(result["evaluations"]["5"]["folds"]) == 4
    assert (root / "analysis.json").is_file()
    assert (root / "report.md").is_file()
    assert (root / "equity.png").stat().st_size > 1000
    assert (root / "comparison.png").stat().st_size > 1000
    metrics = pd.read_csv(root / "metrics.csv")
    intervals = pd.read_csv(root / "excess_intervals.csv")
    assert len(metrics) == 44  # (6 seeded strategies + 5 unique baselines) * 4 folds.
    assert len(intervals) == 48  # 6 strategies * 2 baselines * 4 folds.
    strategy = metrics.loc[(metrics.fold == "fold_01") & (metrics.strategy == "robust_ppo")].iloc[0]
    assert strategy.total_return_mean == pytest.approx(-.0466666666666667)
    assert strategy.total_return_std == pytest.approx(.0404145188432738)
    assert strategy.kind == "strategy" and strategy.n_paths == 3 and strategy.return_days == 20
    assert strategy.start_date == "2019-01-03" and strategy.initial_date == "2019-01-02"
    baseline = metrics.loc[(metrics.fold == "fold_01") & (metrics.strategy == "equal_90")].iloc[0]
    assert baseline.kind == "baseline" and baseline.n_paths == 1 and pd.isna(baseline.total_return_std)
    assert len(metrics.drop_duplicates(["cost_bps", "fold", "strategy"])) == len(metrics)
    ci = intervals.loc[(intervals.fold == "fold_01") & (intervals.strategy == "robust_ppo")
                       & (intervals.benchmark == "equal_90")].iloc[0]
    assert ci.estimate == pytest.approx(0.)
    assert ci.n_days == 20 and ci.n_seeds == 3
    assert ci.mean_seed_total_return == pytest.approx(-.0466666666666667)
    assert ci.cumulative_return_difference == pytest.approx(-.0466666666666667)
    assert ci.ci_lower <= ci.ci_upper


def test_cost_stress_must_use_the_same_market_dates_as_primary(tmp_path):
    root = fixture_run(tmp_path)
    source = root / "evaluation_5bps"
    stress = root / "evaluation_10bps"
    shutil.copytree(source, stress)
    summary = json.loads((stress / "summary.json").read_text())
    summary["cost_bps"] = 10
    put_json(stress / "summary.json", summary)
    for path in (stress / "fold_01").glob("*.csv"):
        frame = pd.read_csv(path)
        frame.loc[7, "date"] = "2019-01-12"  # Same length, endpoints and per-cost alignment, wrong calendar.
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="calendar"):
        reporter().analyze_run(root, n_bootstrap=20)
