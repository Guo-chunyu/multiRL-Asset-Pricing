"""Strict, retrospective analysis of completed seed and walk-forward ledgers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import performance_metrics
from .statistics import compare_seeded_returns


ALIASES = {"best_rl_on_validation", "selected_on_validation", "consensus_validation_algorithm"}
LABELS = {
    "td3_risk_2": "TD3", "ddpg": "DDPG", "sac": "SAC", "ppo": "PPO", "a2c": "A2C",
    "best_rl_on_validation": "Best RL per seed (validation)",
    "selected_on_validation": "Best overall per seed (validation)",
    "consensus_validation_algorithm": "Consensus RL (validation)",
    "dqn_selector": "DQN selector", "equal_mixture": "Equal policy mixture",
    "equal_weight": "Equal weight", "buy_hold": "Buy and hold",
    "momentum": "Momentum", "reversal": "Reversal", "cash": "Cash",
}
PLOT_ORDER = list(LABELS)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read required result file {path}: {error}") from error


def _ledger(path, interval):
    """Read the complete ledger, keeping the reset row only for equity plots."""
    path = Path(path)
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"Cannot read strategy ledger {path}: {error}") from error
    required = {"date", "decision_date", "net_return", "portfolio_value", "turnover"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing ledger columns in {path}: {required - set(frame.columns)}")
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
    decisions = pd.DatetimeIndex(pd.to_datetime(frame["decision_date"], errors="raise"))
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError(f"Ledger dates must be unique and increasing: {path}")
    if len(frame) != int(interval["transitions"]) + 1 or len(frame) < 2:
        raise ValueError(f"Ledger row count differs from manifest interval: {path}")
    if dates[0] != pd.Timestamp(interval["first_decision"]) or dates[-1] != pd.Timestamp(interval["last_valuation"]):
        raise ValueError(f"Ledger date boundaries differ from manifest: {path}")
    if decisions[0] != dates[0] or not decisions[1:].equals(dates[:-1]):
        raise ValueError(f"Ledger decision dates must be the preceding valuation dates: {path}")
    numeric = frame[["net_return", "portfolio_value", "turnover"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"Nonfinite ledger values: {path}")
    returns, equity, turnover = numeric.T
    if np.any(returns <= -1) or np.any(equity <= 0) or np.any(turnover < 0):
        raise ValueError(f"Invalid simple returns, portfolio_value or turnover: {path}")
    if not np.allclose(numeric[0], [0, 1, 0], rtol=0, atol=1e-12):
        raise ValueError(f"Ledger must begin with the cash reset initial row: {path}")
    if not np.allclose(equity, np.cumprod(1 + returns), rtol=1e-9, atol=1e-11):
        raise ValueError(f"portfolio_value does not reconcile to net returns: {path}")
    return {"dates": dates, "returns": returns[1:], "equity": equity, "turnover": turnover[1:]}


def _std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def _training_costs(models, selectors, seeds):
    result = {}
    names = list(models[str(seeds[0])])
    if all(selectors[str(seed)] is not None for seed in seeds):
        names.append("dqn_selector")
    for name in names:
        rows = [selectors[str(seed)] if name == "dqn_selector" else models[str(seed)][name]
                for seed in seeds]
        times = [row.get("training_wall_seconds") for row in rows]
        valid = all(value is not None and np.isfinite(value) and value >= 0 for value in times)
        result[name] = {
            "mean_wall_seconds": float(np.mean(times)) if valid else None,
            "std_wall_seconds": _std(times) if valid else None,
            "wall_seconds_by_seed": dict(zip(map(str, seeds), times)),
            "requested_steps_by_seed": {str(seed): row.get("requested_steps") for seed, row in zip(seeds, rows)},
            "actual_steps_by_seed": {str(seed): row.get("actual_steps", row.get("num_timesteps"))
                                     for seed, row in zip(seeds, rows)},
        }
    return result


def analyze_run(run, block_length=10, n_bootstrap=2000, bootstrap_seed=7):
    """Analyze all seeds on exactly matching dates; never silently inner-join."""
    run = Path(run).resolve()
    manifest = _read_json(run / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"Run must be complete before statistical reporting: {run}")
    seeds = manifest["config"]["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Manifest seeds must be nonempty and unique")
    test_interval, val_interval = manifest["splits"]["test"], manifest["splits"]["validation"]
    if pd.Timestamp(val_interval["last_valuation"]) >= pd.Timestamp(test_interval["first_decision"]):
        raise ValueError("validation date interval must precede the test interval")
    paths, models, selectors, selections, validation = {}, {}, {}, {}, {}
    expected_names, expected_models, reference_dates, validation_dates = None, None, None, None
    for seed in seeds:
        key, directory = str(seed), run / f"seed_{seed}"
        files = {path.stem.removeprefix("test_"): path for path in directory.glob("test_*.csv")
                 if path.stem.removeprefix("test_") not in ALIASES}
        if "equal_weight" not in files:
            raise ValueError(f"Missing equal_weight benchmark strategy for seed {seed}")
        if expected_names is None:
            expected_names = set(files)
        if set(files) != expected_names:
            raise ValueError(f"Test strategy sets differ across seeds: {seed}")
        paths[key] = {}
        for name, path in files.items():
            ledger = _ledger(path, test_interval)
            if reference_dates is None:
                reference_dates = ledger["dates"]
            if not reference_dates.equals(ledger["dates"]):
                raise ValueError(f"Test date alignment differs for {name}, seed {seed}")
            paths[key][name] = ledger
        models[key] = _read_json(directory / "models.json")
        if not models[key] or not set(models[key]).issubset(files):
            raise ValueError(f"Learned model strategies are missing from test ledgers: {seed}")
        if expected_models is None:
            expected_models = set(models[key])
        if set(models[key]) != expected_models:
            raise ValueError(f"Learned strategy sets differ across seeds: {seed}")
        selectors[key] = _read_json(directory / "selector.json") if (directory / "selector.json").exists() else None
        selections[key] = _read_json(directory / "selection.json")
        choice = selections[key]
        if choice.get("selected_on") != "validation":
            raise ValueError(f"Strategy selection must explicitly use validation: seed {seed}")
        if choice.get("strategy") not in files or choice.get("best_rl_on_validation") not in models[key]:
            raise ValueError(f"Unknown validation-selected strategy for seed {seed}")
        validation[key] = {}
        for name in sorted(models[key]):
            ledger = _ledger(directory / f"validation_{name}.csv", val_interval)
            if validation_dates is None:
                validation_dates = ledger["dates"]
            if not validation_dates.equals(ledger["dates"]):
                raise ValueError(f"validation date alignment differs for {name}, seed {seed}")
            metrics = performance_metrics(ledger["returns"], ledger["turnover"])
            validation[key][name] = metrics

    ranking = []
    for name in sorted(expected_models):
        scores = [validation[str(seed)][name]["total_return"] - .5 * validation[str(seed)][name]["max_drawdown"]
                  for seed in seeds]
        ranking.append({
            "strategy": name, "algorithm": models[str(seeds[0])][name].get("algorithm", name),
            "mean_validation_score": float(np.mean(scores)), "std_validation_score": _std(scores),
            "score_by_seed": dict(zip(map(str, seeds), scores)),
            "validation_metrics_by_seed": {str(seed): validation[str(seed)][name] for seed in seeds},
        })
    # Alphabetic tie breaking is deterministic and never consults test returns.
    ranking.sort(key=lambda row: (-row["mean_validation_score"], row["strategy"]))
    consensus = ranking[0]["strategy"]
    for seed in seeds:
        key = str(seed)
        for alias, source in {
            "best_rl_on_validation": selections[key]["best_rl_on_validation"],
            "selected_on_validation": selections[key]["strategy"],
            "consensus_validation_algorithm": consensus,
        }.items():
            paths[key][alias] = paths[key][source]
    benchmark = paths[str(seeds[0])]["equal_weight"]["returns"]
    for seed in seeds[1:]:
        if not np.allclose(benchmark, paths[str(seed)]["equal_weight"]["returns"], rtol=0, atol=1e-12):
            raise ValueError("The deterministic equal_weight benchmark differs across training seeds")
    results = {}
    for name in sorted(expected_names | ALIASES):
        ledgers = [paths[str(seed)][name] for seed in seeds]
        returns = np.stack([ledger["returns"] for ledger in ledgers])
        aggregate = compare_seeded_returns(returns, benchmark, block_length=block_length,
                                          n_bootstrap=n_bootstrap, seed=bootstrap_seed)
        metrics = [performance_metrics(ledger["returns"], ledger["turnover"]) for ledger in ledgers]
        source_by_seed = {}
        for seed in seeds:
            source = (selections[str(seed)]["best_rl_on_validation"] if name == "best_rl_on_validation"
                      else selections[str(seed)]["strategy"] if name == "selected_on_validation"
                      else consensus if name == "consensus_validation_algorithm" else name)
            source_by_seed[str(seed)] = source
        results[name] = {
            **aggregate,
            "source_by_seed": source_by_seed,
            "metrics_by_seed": dict(zip(map(str, seeds), metrics)),
            "mean_total_return": float(np.mean([item["total_return"] for item in metrics])),
            "std_total_return": _std([item["total_return"] for item in metrics]),
            "mean_max_drawdown": float(np.mean([item["max_drawdown"] for item in metrics])),
            "std_max_drawdown": _std([item["max_drawdown"] for item in metrics]),
            "mean_turnover": float(np.mean([item["mean_turnover"] for item in metrics])),
            "std_turnover": _std([item["mean_turnover"] for item in metrics]),
            "mean_equity": np.mean([ledger["equity"] for ledger in ledgers], axis=0).tolist(),
            "equity_by_seed": {str(seed): ledger["equity"].tolist() for seed, ledger in zip(seeds, ledgers)},
        }
    return {
        "study_status": "retrospective_exploratory",
        "run_directory": str(run), "data_sha256": manifest.get("data_sha256"),
        "config": manifest["config"], "splits": manifest["splits"], "seeds": list(seeds),
        "n_observations": len(reference_dates) - 1,
        "initial_date": reference_dates[0].isoformat(),
        "return_dates": [date.isoformat() for date in reference_dates[1:]],
        "equity_dates": [date.isoformat() for date in reference_dates],
        "strategies": results, "selections_by_seed": selections,
        "validation_algorithm_ranking": ranking, "consensus_validation_algorithm": consensus,
        "consensus_selection_rule": "maximum across algorithms of mean across seeds(validation total_return - 0.5 * max_drawdown)",
        "training_costs": _training_costs(models, selectors, seeds),
    }


def _plot(main, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    names = [name for name in PLOT_ORDER if name in main["strategies"]]
    names += sorted(set(main["strategies"]) - set(names))
    y = np.arange(len(names))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150}):
        fig, axes = plt.subplots(1, 2, figsize=(14, max(6.5, len(names) * .40)), sharey=True,
                                 gridspec_kw={"wspace": .12})
        means = [main["strategies"][name]["mean_total_return"] for name in names]
        stds = [main["strategies"][name]["std_total_return"] or 0 for name in names]
        colors = ["#e18c25" if name in ALIASES else "#24728d" for name in names]
        axes[0].barh(y, means, xerr=stds, height=.62, color=colors, alpha=.9,
                     error_kw={"capsize": 3, "linewidth": 1})
        axes[0].set_title("Test cumulative return: seed mean +/- seed SD", loc="left", fontsize=11)
        axes[0].set_yticks(y, [LABELS.get(name, name) for name in names])
        for index, name in enumerate(names):
            comparison = main["strategies"][name]["comparison"]
            low, high = comparison["annualized_mean_excess_ci"]
            color = "#929aa2" if comparison["ci_contains_zero"] else "#24728d"
            axes[1].hlines(index, low, high, color=color, linewidth=2)
            axes[1].plot(comparison["annualized_mean_excess_return"], index, "o", color=color, ms=5)
        axes[1].set_title("Annualized mean excess vs equal weight: 95% CI", loc="left", fontsize=11)
        for ax in axes:
            ax.axvline(0, color="#45515b", linewidth=.8)
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis="x", color="#d9dfe3", alpha=.6)
            ax.set_axisbelow(True)
        axes[0].invert_yaxis()
        fig.suptitle("A-share ETF portfolio study | retrospective / exploratory", x=.04, ha="left", fontsize=15)
        settings = main["strategies"]["equal_weight"]["comparison"]
        fig.text(.04, .015, f"{main['n_observations']} common market dates; {len(main['seeds'])} training seeds. "
                 f"Paired moving blocks: L={settings['block_length']}, B={settings['n_bootstrap']}. "
                 "CI is arithmetic excess, not CAGR; no multiplicity correction.", fontsize=9)
        fig.subplots_adjust(left=.25, right=.98, bottom=.10, top=.89)
        fig.savefig(output / "comparison.png", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 6))
        dates = pd.to_datetime(main["equity_dates"])
        equity_names = ["equal_weight", "buy_hold", "best_rl_on_validation",
                        "consensus_validation_algorithm", "selected_on_validation", "dqn_selector", "equal_mixture"]
        for name in equity_names:
            if name in main["strategies"]:
                ax.plot(dates, main["strategies"][name]["mean_equity"],
                        label=LABELS.get(name, name), linewidth=1.8)
        ax.axhline(1, linewidth=.7, color="#838c94")
        ax.set_title("Test equity | mean of separately initialized seed portfolios", loc="left", pad=16)
        ax.set_ylabel("Normalized portfolio value (initial = 1)")
        ax.grid(alpha=.2)
        ax.legend(loc="upper left", fontsize=8)
        fig.autofmt_xdate()
        fig.text(.08, .015, "Retrospective / exploratory. Mean seed NAV is a descriptive summary, "
                 "not a new traded ensemble. Walk-forward folds are not spliced.", fontsize=9)
        fig.tight_layout(rect=(0, .05, 1, 1))
        fig.savefig(output / "equity.png", bbox_inches="tight")
        plt.close(fig)


def _pct(value):
    return "未定义" if value is None else f"{100 * value:.2f}%"


def _table(run):
    rows = [
        "| 策略 | 种子平均累计收益 | 种子收益标准差 | 平均最大回撤 | 日均双边换手 | 年化算术超额均值 | 95% 区间 | 包含零 |",
        "|---|---:|---:|---:|---:|---:|---|:---:|",
    ]
    for name, item in run["strategies"].items():
        comparison = item["comparison"]
        low, high = comparison["annualized_mean_excess_ci"]
        rows.append(f"| `{name}` | {_pct(item['mean_total_return'])} | {_pct(item['std_total_return'])} "
                    f"| {_pct(item['mean_max_drawdown'])} | {_pct(item['mean_turnover'])} "
                    f"| {_pct(comparison['annualized_mean_excess_return'])} | [{_pct(low)}, {_pct(high)}] "
                    f"| {'是' if comparison['ci_contains_zero'] else '否'} |")
    return "\n".join(rows)


def _write_report(analysis, output):
    main = analysis["main"]
    primary = main["strategies"]["best_rl_on_validation"]
    comparison = primary["comparison"]
    low, high = comparison["annualized_mean_excess_ci"]
    verdict = ("区间包含零，当前样本不足以支持相对等权存在稳定超额的结论。" if comparison["ci_contains_zero"]
               else "区间未包含零，但它仍是未经多重比较校正的探索性结果，不能证明未来或真正未见数据上的优势。")
    consensus_name = main["consensus_validation_algorithm"]
    consensus = main["strategies"]["consensus_validation_algorithm"]
    lines = [
        "# A 股 ETF 深度强化学习比较：回顾性研究",
        "",
        "本轮使用已经被研究过程查看过的历史测试区间，因此所有新增算法、消融及推断均属于**回顾性 / 探索性**研究。"
        "代码中的时间隔离不能恢复研究者层面的未见测试集；不得将本报告称为全新、未经查看的样本外验证。",
        "",
        f"主实验测试收益覆盖 {main['return_dates'][0][:10]} 至 {main['return_dates'][-1][:10]}，"
        f"共 {main['n_observations']} 个交易日；训练种子为 {main['seeds']}。"
        f"每个基础算法请求 {main['config'].get('total_timesteps')} 步，成本假设为 "
        f"{main['config'].get('cost_bps')} bps；训练风险参数为 {main['config'].get('risk_levels')}。",
        "",
        "## 主要比较",
        "",
        f"逐种子验证期选出的最佳 RL（`best_rl_on_validation`）平均测试累计收益为 {_pct(primary['mean_total_return'])}，"
        f"种子收益标准差为 {_pct(primary['std_total_return'])}。相对等权的年化算术平均日超额为 "
        f"{_pct(comparison['annualized_mean_excess_return'])}，95% 区间为 [{_pct(low)}, {_pct(high)}]。{verdict}",
        "",
        f"统一算法由 {len(main['validation_algorithm_ranking'])} 个候选的跨种子**验证期**平均 "
        "`累计收益 - 0.5 × 最大回撤` 排名决定：本次为 "
        f"`{consensus_name}`。其全部种子的平均测试累计收益为 {_pct(consensus['mean_total_return'])}，"
        f"种子标准差为 {_pct(consensus['std_total_return'])}。此选择未挑选最好 seed，也未用测试收益选择算法；"
        "它保留为 `consensus_validation_algorithm`，不覆盖逐种子选择结果。候选算法数量以实际配置为准。",
        "",
        "| 验证期排名 | 算法 / 策略 | 平均验证得分 | 得分种子标准差 |",
        "|---:|---|---:|---:|",
    ]
    for index, row in enumerate(main["validation_algorithm_ranking"], 1):
        lines.append(f"| {index} | `{row['strategy']}` | {_pct(row['mean_validation_score'])} "
                     f"| {_pct(row['std_validation_score'])} |")
    lines += [
        "", "## 主实验完整比较", "", _table(main), "",
        "累计收益、回撤和换手先在每个种子内计算，再取种子均值；标准差采用样本标准差，表示训练随机性。"
        "等权基准只有一条共同市场路径，三个 seed 不构成三个独立市场样本。图中净值是各 seed 净值的平均，"
        "它与“逐日平均收益后复利”不同，两者均在 JSON 中保留。",
        "",
        "置信区间使用各 seed 在同一天的平均日收益减等权日收益，做配对、非环绕的重叠移动区块 bootstrap："
        f"固定区块长度 {comparison['block_length']}、重抽样 {comparison['n_bootstrap']} 次、置信度 "
        f"{comparison['confidence']:.0%}、随机种子 {comparison['bootstrap_seed']}。"
        "区间量纲为 252 × 平均日超额，即年化算术超额；它不是累计收益差或 CAGR 的区间，不代表未来获胜概率。"
        "区间条件于这些已训练模型，未重训模型，也未校正多重比较。正收益本身不等于优于基准；包含零必须明确报告。",
        "", "## 验证期选择记录", "",
        "| 训练种子 | 最佳 RL | 全候选最佳（包括现金） |", "|---:|---|---|",
    ]
    for seed, choice in main["selections_by_seed"].items():
        lines.append(f"| {seed} | `{choice['best_rl_on_validation']}` | `{choice['strategy']}` |")
    lines += ["", "## 训练计算成本", "",
              "墙钟时间受并发负载和硬件影响，不是算法复杂度的严格比较。DQN 另有训练预算，其依赖的基础模型成本不能忽略。",
              "", "| 模型 | 平均训练秒数 | 秒数标准差 | 请求步数（按 seed） | 实际步数（按 seed） |",
              "|---|---:|---:|---|---|"]
    for name, cost in main["training_costs"].items():
        mean = "未记录" if cost["mean_wall_seconds"] is None else f"{cost['mean_wall_seconds']:.2f}"
        std = "未定义" if cost["std_wall_seconds"] is None else f"{cost['std_wall_seconds']:.2f}"
        lines.append(f"| `{name}` | {mean} | {std} | {list(cost['requested_steps_by_seed'].values())} "
                     f"| {list(cost['actual_steps_by_seed'].values())} |")
    lines += ["", "## 扩展窗口检验", "",
              "各折独立从现金重置，分别计算指标和 bootstrap；**不拼接**为连续净值，不让区块跨折，"
              "也不把不同折或训练种子当成独立市场来增加显著性。后续折合法使用此前已发生的数据作为训练历史。"]
    walk = analysis["walk_forward"]
    if walk is None:
        lines += ["", "此次未提供完整的扩展窗口结果。"]
    else:
        for index, fold in enumerate(walk["folds"], 1):
            best = fold["strategies"]["best_rl_on_validation"]
            interval = best["comparison"]["annualized_mean_excess_ci"]
            if best["comparison"]["ci_contains_zero"]:
                direction = "区间包含零，不能据此声称稳定超额。"
            elif interval[1] < 0:
                direction = "区间完全低于零，提示该历史样本内落后于等权；仍为未经多重比较校正的探索性比较。"
            else:
                direction = "区间完全高于零，但仍为未经多重比较校正的探索性比较，不能直接推断未来优势。"
            lines += [
                "", f"### 第 {index} 折", "",
                f"收益日期 {fold['return_dates'][0][:10]} 至 {fold['return_dates'][-1][:10]}，"
                f"{fold['n_observations']} 日；训练预算 {fold['config'].get('total_timesteps')} 步 / 算法 / seed。"
                f"本折验证期统一算法为 `{fold['consensus_validation_algorithm']}`。",
                "",
                f"本折逐种子最佳 RL 平均累计收益 {_pct(best['mean_total_return'])}，等权为 "
                f"{_pct(fold['strategies']['equal_weight']['mean_total_return'])}；"
                f"年化算术超额的 95% 区间为 [{_pct(interval[0])}, {_pct(interval[1])}]。{direction}",
                "", _table(fold),
            ]
    lines += [
        "", "## 解释边界", "",
        "数据是固定 ETF 集合和事后复权快照，存在标的选择、时间段选择与数据版本限制。"
        "模拟考虑下一开盘交易、隔夜持仓收益和比例成本，但不等同于实际可成交结果；"
        "逐笔成交、整手、涨跌停、停牌、容量、冲击和其他真实执行限制需要另行验证。"
        "本报告提供研究比较，不构成收益承诺。",
        "", "![策略收益及区间](comparison.png)", "", "![主实验平均净值](equity.png)", "",
        f"数据 SHA-256：`{main['data_sha256']}`。完整指标、逐 seed 结果和逐折区间见 `analysis.json`。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_research_report(run, output, walk_forward=None, block_length=10,
                          n_bootstrap=2000, bootstrap_seed=7):
    """Write JSON, two English figures, and a Chinese exploratory report."""
    main = analyze_run(run, block_length, n_bootstrap, bootstrap_seed)
    walk = None
    if walk_forward is not None:
        directory = Path(walk_forward).resolve()
        summary = _read_json(directory / "walkforward_summary.json")
        fold_dirs = sorted(path for path in directory.glob("fold_*") if path.is_dir())
        if not fold_dirs or len(fold_dirs) != summary.get("fold_count"):
            raise ValueError("Walk-forward fold directory count differs from completed summary")
        folds, seen_dates = [], set()
        for folder in fold_dirs:
            fold = analyze_run(folder, block_length, n_bootstrap, bootstrap_seed)
            if fold["data_sha256"] != main["data_sha256"] or fold["seeds"] != main["seeds"]:
                raise ValueError("Walk-forward data hash and seeds must match the main study")
            if seen_dates.intersection(fold["return_dates"]):
                raise ValueError("Walk-forward test return dates overlap")
            seen_dates.update(fold["return_dates"])
            folds.append(fold)
        if summary.get("data_sha256") != main["data_sha256"]:
            raise ValueError("Walk-forward summary data hash differs from the main study")
        walk = {"directory": str(directory), "folds": folds, "test_return_dates_overlap": False,
                "continuous_equity_constructed": False,
                "interpretation": "Each reset fold has separate metrics and separate bootstrap; no pooled return path."}
    analysis = {
        "study_status": "retrospective_exploratory", "main": main, "walk_forward": walk,
        "statistics": {"benchmark": "equal_weight", "block_length": block_length,
                       "n_bootstrap": n_bootstrap, "bootstrap_seed": bootstrap_seed,
                       "confidence": .95, "seed_aggregation": "mean per common date; no seed flattening"},
    }
    output = Path(output).resolve()
    if output == Path(run).resolve() or (walk_forward is not None and output == Path(walk_forward).resolve()):
        raise ValueError("Report output must differ from source experiment directories")
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _plot(main, output)
    _write_report(analysis, output)
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Completed main experiment directory")
    parser.add_argument("--walk-forward", help="Completed expanding walk-forward directory")
    parser.add_argument("--output", required=True, help="Separate output directory")
    args = parser.parse_args()
    analysis = build_research_report(args.run, args.output, walk_forward=args.walk_forward)
    result = analysis["main"]
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "consensus_validation_algorithm": result["consensus_validation_algorithm"],
        "mean_test_return": result["strategies"]["consensus_validation_algorithm"]["mean_total_return"],
        "study_status": analysis["study_status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
