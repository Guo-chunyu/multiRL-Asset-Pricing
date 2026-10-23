"""Audit and summarize frozen v3 test ledgers, separately for each market fold.

This module never loads/trains a model or selects a candidate from test results.
Run only after the study's select/evaluate stages have completed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import performance_metrics
from .statistics import compare_seeded_returns


BASELINES = ("equal_80", "equal_90", "equal_100", "buy_hold", "inverse_vol_90")
METRICS = ("total_return", "annual_return", "sharpe", "max_drawdown", "mean_turnover", "mean_cash_weight")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ledger(path, fold, n_assets, expected_dates=None):
    frame = pd.read_csv(path)
    required = {"date", "net_return", "portfolio_value", "turnover", "weights"}
    if not required <= set(frame):
        raise ValueError(f"Missing ledger columns: {path}")
    dates = pd.DatetimeIndex(pd.to_datetime(frame.date, format="mixed", errors="raise"))
    if (not dates.is_unique or not dates.is_monotonic_increasing or dates.hasnans
            or not dates.equals(dates.normalize())):
        raise ValueError(f"Dates must be unique, increasing daily sessions: {path}")
    expected_count = fold["test"][1] - fold["test"][0] + 1
    start, end = map(pd.Timestamp, fold["dates"]["test"])
    if len(dates) != expected_count or dates[0] != start or dates[-1] != end:
        raise ValueError(f"Ledger date count/endpoints differ from registered fold: {path}")
    if expected_dates is not None and not dates.equals(expected_dates):
        raise ValueError(f"All strategy and baseline dates must align exactly: {path}")
    returns = frame.net_return.to_numpy(float)
    values = frame.portfolio_value.to_numpy(float)
    turnover = frame.turnover.to_numpy(float)
    if (not np.isfinite(values).all() or np.any(values <= 0)
            or not np.isclose(values[0], 1., rtol=0, atol=1e-10)
            or not np.isclose(returns[0], 0., rtol=0, atol=1e-12)
            or not np.isclose(turnover[0], 0., rtol=0, atol=1e-12)):
        raise ValueError(f"Invalid initial NAV/return/turnover: {path}")
    if not np.allclose(values, np.cumprod(1 + returns), rtol=1e-8, atol=1e-10):
        raise ValueError(f"NAV does not reconcile with daily returns: {path}")
    try:
        weights = np.asarray([json.loads(v) for v in frame.weights], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid portfolio weights: {path}") from exc
    if (weights.shape != (len(frame), n_assets + 1) or not np.isfinite(weights).all()
            or np.any(weights < -1e-9) or not np.allclose(weights.sum(axis=1), 1., rtol=0, atol=1e-7)):
        raise ValueError(f"Weights must be finite cash-first simplex vectors: {path}")
    metrics = performance_metrics(returns[1:], turnover[1:])
    metrics.update(mean_cash_weight=float(weights[1:, 0].mean()), return_days=len(frame) - 1)
    return {"dates": dates, "returns": returns[1:], "nav": values, "metrics": metrics}


def _check_summary(actual, reported, label):
    for key, expected in actual.items():
        if key not in reported:
            raise ValueError(f"Missing summary metric {key}: {label}")
        observed = reported[key]
        if expected is None:
            match = observed is None
        else:
            match = (isinstance(observed, (int, float)) and np.isfinite(observed)
                     and np.isclose(expected, observed, rtol=1e-7, atol=1e-10))
        if not match:
            raise ValueError(f"Summary metric {key} differs from ledger: {label}")


def _aggregate(paths):
    metrics = {}
    for key in METRICS:
        values = [p["metrics"][key] for p in paths]
        defined = [v for v in values if v is not None]
        # Do not silently average only a successful subset of seed ratios.
        complete = len(defined) == len(values)
        metrics[key] = {"mean": float(np.mean(defined)) if complete else None,
                        "std": float(np.std(defined, ddof=1)) if complete and len(values) > 1 else None,
                        "n_defined": len(defined)}
    return {"n_paths": len(paths), "metrics": metrics,
            "per_path_metrics": [p["metrics"] for p in paths],
            "mean_nav": np.mean([p["nav"] for p in paths], axis=0).tolist()}


def analyze_run(run, *, n_bootstrap=2000):
    """Read only completed outputs; all inference uses aligned market dates."""
    run = Path(run)
    manifest = _json(run / "manifest.json")
    frozen = _json(run / "frozen_selection.json")
    candidates = list(manifest["identity"]["candidates"])
    seeds = list(manifest["identity"]["seeds"])
    if (not seeds or len(set(seeds)) != len(seeds) or not candidates
            or len(set(candidates)) != len(candidates)):
        raise ValueError("Need unique registered candidates and training seeds")
    strategies = candidates + ["robust_mixture", "selected_on_validation"]
    fold_names = [f["name"] for f in manifest["folds"]]
    if len(set(fold_names)) != len(fold_names) or set(frozen["folds"]) != set(fold_names):
        raise ValueError("Frozen selection must cover exactly the registered folds")
    previous_end = None
    for fold in manifest["folds"]:
        start, end = map(pd.Timestamp, fold["dates"]["test"])
        if start >= end or (previous_end is not None and start <= previous_end):
            raise ValueError("Test folds must be ordered and nonoverlapping")
        previous_end = end
        selection = frozen["folds"][fold["name"]]
        if selection.get("selected_on") != "validation" or selection["candidate"] not in candidates:
            raise ValueError("Only a registered candidate frozen on validation is allowed")

    result = {"manifest": manifest, "frozen_selection": frozen,
              "source_hashes": {name: _digest(run / name) for name in ["manifest.json", "frozen_selection.json"]},
              "method": {"bootstrap_block_days": 10, "bootstrap_resamples": n_bootstrap,
                         "confidence": .95, "annualization": 252,
                         "seed_uncertainty": "Sample SD across training seeds; not independent market histories",
                         "bootstrap_unit": "one aligned market date, after averaging seed daily returns",
                         "equity_curve": "arithmetic mean of initial-1 NAV paths; not compounded mean daily returns",
                         "cash_weight": "Cash plus dividend receivables, divided by total NAV",
                         "folds": "Independent starting NAV=1; never concatenate fold returns",
                         "selection": "Read frozen validation selection; no ranking or selection by test results"},
              "evaluations": {}}
    for cost in (5, 10, 20):
        directory = run / f"evaluation_{cost}bps"
        path = directory / "summary.json"
        if not path.is_file():
            if cost == 5:
                raise ValueError("Primary 5bps evaluation summary is required")
            continue
        summary = _json(path)
        if summary["cost_bps"] != cost or set(summary["folds"]) != set(fold_names):
            raise ValueError("Evaluation cost/folds differ from registered study")
        evaluation = {"cost_bps": cost, "summary_sha256": _digest(path), "folds": {}}
        for fold in manifest["folds"]:
            name = fold["name"]
            report = summary["folds"][name]
            selected = frozen["folds"][name]["candidate"]
            if (report["selected"] != selected or set(report["seeds"]) != {str(s) for s in seeds}
                    or set(report["baselines"]) != set(BASELINES)):
                raise ValueError("Selection, seed coverage, or baseline coverage mismatch")
            expected_dates = None
            paths, baseline_paths, file_hashes = {}, {}, {}
            for baseline in BASELINES:
                file = directory / name / f"{baseline}.csv"
                ledger = _ledger(file, fold, len(manifest["tickers"]), expected_dates)
                expected_dates = ledger["dates"]
                _check_summary(ledger["metrics"], report["baselines"][baseline], str(file))
                baseline_paths[baseline] = ledger
                file_hashes[file.name] = _digest(file)
            if cost != 5:
                primary_dates = pd.DatetimeIndex(result["evaluations"]["5"]["folds"][name]["dates"])
                if not expected_dates.equals(primary_dates):
                    raise ValueError("Cost-stress calendar must match the primary evaluation exactly")
            for strategy in strategies:
                source = selected if strategy == "selected_on_validation" else strategy
                paths[strategy] = []
                for seed in seeds:
                    if set(report["seeds"][str(seed)]) != set(strategies):
                        raise ValueError("Each seed must report all registered strategies")
                    file = directory / name / f"seed_{seed}_{source}.csv"
                    ledger = _ledger(file, fold, len(manifest["tickers"]), expected_dates)
                    _check_summary(ledger["metrics"], report["seeds"][str(seed)][strategy], str(file))
                    paths[strategy].append(ledger)
                    file_hashes[file.name] = _digest(file)
            rows = {}
            for strategy, ledgers in paths.items():
                row = _aggregate(ledgers)
                row["seeds"] = seeds
                row["source_candidate"] = selected if strategy == "selected_on_validation" else strategy
                row["comparisons"] = {
                    benchmark: compare_seeded_returns(
                        np.asarray([p["returns"] for p in ledgers]), baseline_paths[benchmark]["returns"],
                        block_length=10, n_bootstrap=n_bootstrap, seed=7)
                    for benchmark in ("equal_90", "equal_100")}
                rows[strategy] = row
            evaluation["folds"][name] = {
                "selected_on_validation": selected, "dates": expected_dates.strftime("%Y-%m-%d").tolist(),
                "return_days": len(expected_dates) - 1, "strategies": rows,
                "baselines": {b: _aggregate([p]) for b, p in baseline_paths.items()},
                "ledger_sha256": file_hashes,
            }
        result["evaluations"][str(cost)] = evaluation
    return result


def _number(value, percent=False):
    if value is None:
        return "—"
    return f"{value * 100:.2f}%" if percent else f"{value:.2f}"


def _metric_cell(row, name):
    metric = row["metrics"][name]
    percent = name != "sharpe"
    value = _number(metric["mean"], percent)
    return value if metric["std"] is None else f"{value} ± {_number(metric['std'], percent)}"


def _markdown(analysis):
    lines = ["# A 股 ETF 组合优化：v3 历史研究", "",
             "本报告来自真实历史原始价格与现金分红、份额拆分账本。旧版腾讯 hfq 收益口径不适合直接计算投资收益，本研究重新计算。",
             "这是回溯探索性研究，不是实盘业绩，也不是从未接触历史后得到的严格样本外证明。", "",
             "每折独立从净值 1 开始，分别报告；不拼接为连续投资净值。所有候选均展示，选中策略只读取冻结的验证期结果。",
             "累计收益、年化收益、Sharpe、最大回撤等均先逐种子计算，再报告均值 ± 样本标准差。种子反映训练随机性，不能当作三段独立市场。",
             "主表、累计收益柱图和 metrics.csv 的累计收益均采用种子累计收益的算术平均；净值图为种子净值的算术平均，因此其期末净值减 1 与该累计收益一致。",
             "年化采用每年 252 个交易日；Sharpe 无风险利率取 0；换手率为日均买卖成交额双边合计/NAV；现金栏包含尚未到账的应收分红。", "",
             "下载实验数据：[指标与种子标准差](metrics.csv)、[成对超额收益区间](excess_intervals.csv)。CSV 的收益、回撤、换手和现金占比均用小数表示（0.10=10%），未定义值留空。", "",
             "## 主实验：单边交易费率 5 bps", ""]
    primary = analysis["evaluations"]["5"]["folds"]
    for name, fold in primary.items():
        lines.extend([f"### {name}：{fold['dates'][1]} 至 {fold['dates'][-1]}", "",
                      f"有效收益日 {fold['return_days']}；验证期冻结选择 `{fold['selected_on_validation']}`。", "",
                      "| 策略 | 累计收益 | 年化收益 | Sharpe | 最大回撤 | 日均换手 | 平均现金及应收 |",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for strategy, row in {**fold["strategies"], **fold["baselines"]}.items():
            lines.append("| " + strategy + " | " + " | ".join(_metric_cell(row, key) for key in METRICS) + " |")
        lines.extend(["", "基准每折仅有一条路径，因此没有种子标准差。`equal_90` 是与新策略风险资产暴露接近的控制，`equal_100` 表示满仓等权；不能把降低仓位带来的回撤下降全归因于 RL。", "",
                      "| 策略 | 对 equal_90 年化日均超额 95% 区间 | 对 equal_100 年化日均超额 95% 区间 |",
                      "|---|---:|---:|"])
        for strategy, row in fold["strategies"].items():
            cells = []
            for baseline in ("equal_90", "equal_100"):
                ci = row["comparisons"][baseline]["comparison"]["annualized_mean_excess_ci"]
                cells.append(f"[{_number(ci[0], True)}, {_number(ci[1], True)}]")
            lines.append(f"| {strategy} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend([f"区间使用逐日对齐的种子平均收益，与单条基准路径进行 10 日移动分块成对 bootstrap（{analysis['method']['bootstrap_resamples']} 次）。",
                  "区间估计的是 252×日均算术超额，不是复合年化收益区间；没有多重比较校正。区间包含 0 时，不能据此确认稳定超额收益。", "",
                  "## 交易成本压力测试", ""])
    available = [c for c in ("10", "20") if c in analysis["evaluations"]]
    if available:
        lines.extend(["同一冻结模型在更高成本下重新执行；模型不再训练，持仓反馈和动作可能随成本改变。", "",
                      "| 折 | 策略 | 5 bps 累计收益 | 10 bps 累计收益 | 20 bps 累计收益 |",
                      "|---|---|---:|---:|---:|"])
        for name, fold in primary.items():
            for strategy in [*fold["strategies"], *fold["baselines"]]:
                group = "strategies" if strategy in fold["strategies"] else "baselines"
                cells = [_metric_cell(analysis["evaluations"][c]["folds"][name][group][strategy], "total_return")
                         if c in analysis["evaluations"] else "未运行" for c in ("5", "10", "20")]
                lines.append(f"| {name} | {strategy} | " + " | ".join(cells) + " |")
    else:
        lines.append("10/20 bps 的完整评估尚未提供，未对其表现作推断。")
    lines.extend(["", "## 数据与实验边界", "",
                  "- 固定 510050、510300、510500 三只长期存续的 A 股宽基 ETF，不代表全部个股或所有股票组合。",
                  "- 原始价格与分红资料来自公开回溯快照，存在供应商误差与资产池选择偏差；并非完整历史时点数据库。",
                  "- 分红除息时确认应收、发放后变为可用现金，拆分按生效日调整份额；应收不能提前用于交易。",
                  "- 研究环境允许分数份额，不能将本报告视为已验证真实成交、盘口排队或券商实盘接入。",
                  "- 新旧策略同时改变输入、奖励、暴露和训练方式；没有完全逐项消融时，收益差异不能唯一归因某个改动。",
                  "- 是否优秀应结合多个时期、同暴露基准、交易成本、回撤和置信区间判断；本报告不按测试收益另选赢家。", "",
                  "![各折种子平均净值与基准](equity.png)", "",
                  "![各折累计收益与种子离散程度](comparison.png)", ""])
    return "\n".join(lines)


def _plots(analysis, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folds = analysis["evaluations"]["5"]["folds"]
    columns, rows = 2, (len(folds) + 1) // 2
    fig, axes = plt.subplots(rows, columns, figsize=(15, rows * 5), squeeze=False)
    comparison, bars = plt.subplots(rows, columns, figsize=(16, rows * 5), squeeze=False)
    colors = {"legacy_ppo": "#377eb8", "robust_ppo": "#e69520", "robust_sac": "#009775",
              "robust_td3": "#e15759", "robust_mixture": "#8c62aa", "selected_on_validation": "#141414",
              "equal_90": "#7a7a7a", "equal_100": "#aaa06b"}
    for ax, bar, (name, fold) in zip(axes.flat, bars.flat, folds.items()):
        dates = pd.to_datetime(fold["dates"])
        plot_rows = {**fold["strategies"], **{b: fold["baselines"][b] for b in ("equal_90", "equal_100")}}
        for strategy, row in plot_rows.items():
            ax.plot(dates, row["mean_nav"], label=strategy, color=colors.get(strategy),
                    linestyle="--" if strategy.startswith("equal_") else "-",
                    linewidth=2.1 if strategy == "selected_on_validation" else 1.25,
                    alpha=.85)
        ax.set_title(f"{name}: {fold['dates'][1]} to {fold['dates'][-1]}")
        ax.set_ylabel("Mean initial-1 NAV across training seeds")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, ncol=2)
        ax.tick_params(axis="x", labelrotation=20)
        names = list(plot_rows)
        means = [plot_rows[n]["metrics"]["total_return"]["mean"] * 100 for n in names]
        deviations = [(plot_rows[n]["metrics"]["total_return"]["std"] or 0) * 100 for n in names]
        y = np.arange(len(names))
        bar.barh(y, means, xerr=deviations, color=[colors.get(n, "#888888") for n in names], capsize=3)
        bar.set_yticks(y, names, fontsize=8)
        bar.invert_yaxis()
        bar.axvline(0, color="#333333", linewidth=.7)
        bar.set_xlabel("Cumulative return (%); error bars = seed SD")
        bar.set_title(name)
        bar.grid(axis="x", alpha=.2)
    for ax in list(axes.flat)[len(folds):] + list(bars.flat)[len(folds):]:
        ax.set_visible(False)
    fig.tight_layout()
    comparison.tight_layout()
    fig.savefig(output / "equity.png", dpi=160)
    comparison.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    plt.close(comparison)


def _export_csv(analysis, output):
    """Flatten already computed analysis; never refit, rank or resample."""
    metric_rows, excess_rows = [], []
    for evaluation in analysis["evaluations"].values():
        for name, fold in evaluation["folds"].items():
            context = {"cost_bps": evaluation["cost_bps"], "fold": name,
                       "initial_date": fold["dates"][0], "start_date": fold["dates"][1],
                       "end_date": fold["dates"][-1], "return_days": fold["return_days"]}
            for kind, group in (("strategy", "strategies"), ("baseline", "baselines")):
                for strategy, row in fold[group].items():
                    record = {**context, "strategy": strategy, "kind": kind, "n_paths": row["n_paths"],
                              "source_candidate": row.get("source_candidate", strategy)}
                    for metric, values in row["metrics"].items():
                        for suffix in ("mean", "std", "n_defined"):
                            record[f"{metric}_{suffix}"] = values[suffix]
                    metric_rows.append(record)
                    for benchmark, compared in row.get("comparisons", {}).items():
                        comparison = compared["comparison"]
                        seed_total = row["metrics"]["total_return"]["mean"]
                        benchmark_total = fold["baselines"][benchmark]["metrics"]["total_return"]["mean"]
                        excess_rows.append({**context, "strategy": strategy, "benchmark": benchmark,
                                            "estimate": comparison["annualized_mean_excess_return"],
                                            "ci_lower": comparison["annualized_mean_excess_ci"][0],
                                            "ci_upper": comparison["annualized_mean_excess_ci"][1],
                                            "n_days": comparison["n_observations"], "n_seeds": row["n_paths"],
                                            "confidence": comparison["confidence"],
                                            "block_days": comparison["block_length"],
                                            "bootstrap_resamples": comparison["n_bootstrap"],
                                            "bootstrap_seed": comparison["bootstrap_seed"],
                                            "ci_contains_zero": comparison["ci_contains_zero"],
                                            "mean_seed_total_return": seed_total,
                                            "benchmark_total_return": benchmark_total,
                                            "cumulative_return_difference": seed_total - benchmark_total,
                                            "estimate_definition": "252 * mean(aligned seed-mean daily return minus benchmark)",
                                            "cumulative_definition": "mean seed cumulative return minus benchmark cumulative return"})
    pd.DataFrame(metric_rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(excess_rows).to_csv(output / "excess_intervals.csv", index=False)


def build_report(run, *, n_bootstrap=2000):
    output = Path(run)
    analysis = analyze_run(output, n_bootstrap=n_bootstrap)
    (output / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "report.md").write_text(_markdown(analysis), encoding="utf-8")
    _export_csv(analysis, output)
    _plots(analysis, output)
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/research_v3")
    args = parser.parse_args()
    result = build_report(args.run)
    print(f"Verified and reported {len(result['evaluations']['5']['folds'])} separate folds in {args.run}")


if __name__ == "__main__":
    main()
