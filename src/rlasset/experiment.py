"""Chronological base training, frozen-policy selection and held-out evaluation."""
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN

from .data import chronological_splits, load_panel, validate_splits
from .env import PortfolioEnv
from .extractors import SharedTemporalExtractor
from .finrl_adapter import ALGORITHM_CLASSES, make_agent, training_budget
from .meta import SelectorEnv, baseline_candidates, mixture_candidate, model_candidate
from .metrics import performance_metrics


@dataclass(frozen=True)
class ExperimentConfig:
    total_timesteps: int = 5000
    seeds: tuple[int, ...] = (7, 17, 29)
    lookback: int = 20
    cost_bps: float = 5.
    max_weight: float = .5
    risk_levels: tuple[float, ...] = (0., 2., 8.)
    backend: str = "finrl"
    learning_rate: float = 3e-4
    shared_encoder: bool = True
    algorithms: tuple[str, ...] = ("td3", "ddpg")
    selector_all_algorithms: bool = False


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _factory(panel, interval, config, risk=2.):
    return lambda: PortfolioEnv(panel, start=interval[0], end=interval[1], lookback=config.lookback, cost_bps=config.cost_bps, risk_aversion=risk, max_weight=config.max_weight)


def _pool(models, selector_all_algorithms=False, pool_order=None):
    """Preserve legacy TD3-only selectors unless all-algorithm mode is explicit."""
    learned = [model_candidate(name, model) for name, model in models.items()
               if selector_all_algorithms or name.startswith("td3_")]
    if not learned:
        raise ValueError("Legacy selector pool requires TD3; enable selector_all_algorithms for other algorithms")
    baselines = baseline_candidates()
    selection_pool = learned + [baselines[x] for x in ("momentum", "reversal", "cash")]
    if pool_order is not None:
        by_name = {candidate.name: candidate for candidate in selection_pool}
        if len(pool_order) != len(by_name) or set(pool_order) != set(by_name):
            raise ValueError("Saved selector pool does not match the configured candidates")
        selection_pool = [by_name[name] for name in pool_order]
    all_policies = {**baselines, **{name: model_candidate(name, model) for name, model in models.items()}, "equal_mixture": mixture_candidate(learned)}
    return selection_pool, all_policies


def _specifications(config):
    specifications = []
    for algorithm in config.algorithms:
        if algorithm == "td3":
            specifications.extend((f"td3_risk_{risk:g}", "td3", risk) for risk in config.risk_levels)
        else:
            specifications.append((algorithm, algorithm, 2.))
    return specifications


def _restore_selector_order(selector, models, config, saved_order=None):
    """Bind discrete actions to saved policy names, refusing inconsistent artifacts."""
    checkpoint_order = getattr(selector, "rlasset_pool_order", None)
    if checkpoint_order is not None and saved_order is not None and list(checkpoint_order) != list(saved_order):
        raise ValueError("Selector checkpoint and saved pool order disagree")
    pool, _ = _pool(models, config.selector_all_algorithms, checkpoint_order or saved_order)
    if selector.action_space.n != len(pool) or selector.observation_space["performance"].shape != (3 * len(pool),):
        raise ValueError("Selector dimensions do not match the saved policy pool")
    selector.rlasset_pool_order = [candidate.name for candidate in pool]
    return pool


def _rollout(factory, candidate=None, selector=None, pool=None):
    env = SelectorEnv(factory, pool) if selector is not None else factory()
    observation, _ = env.reset()
    done = False
    while not done:
        if selector is not None:
            action, _ = selector.predict(observation, deterministic=True)
            observation, _, done, _, _ = env.step(action)
        else:
            observation, _, done, _, _ = env.step_weights(candidate.weights(env, observation))
    ledger = pd.DataFrame((env.base if selector is not None else env).history)
    metrics = performance_metrics(ledger.net_return.iloc[1:].to_numpy(), turnover=ledger.turnover.iloc[1:].to_numpy())
    return ledger, metrics


def _save_ledger(ledger, path, tickers):
    df = ledger.copy()
    for field in ("weights", "target_weights", "pretrade_weights"):
        matrix = np.asarray(df.pop(field).tolist())
        for i, ticker in enumerate(("cash", *tickers)):
            df[f"{field}_{ticker}"] = matrix[:, i]
    df.to_csv(path, index=False)


def _evaluate(panel, interval, config, models, selector, directory=None, stage="test"):
    pool, candidates = _pool(models, config.selector_all_algorithms, getattr(selector, "rlasset_pool_order", None))
    factory = _factory(panel, interval, config)
    ledgers, metrics = {}, {}
    for name, candidate in candidates.items():
        ledgers[name], metrics[name] = _rollout(factory, candidate=candidate)
    ledgers["dqn_selector"], metrics["dqn_selector"] = _rollout(factory, selector=selector, pool=pool)
    if directory is not None:
        for name, ledger in ledgers.items():
            _save_ledger(ledger, directory / f"{stage}_{name}.csv", panel.tickers)
        _write_json(directory / f"{stage}_metrics.json", metrics)
    return ledgers, metrics


def _plot(ledgers, selected, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys([selected, "dqn_selector", "td3_risk_2", "ddpg", "equal_weight", "buy_hold"]))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for name in names:
        if name not in ledgers:
            continue
        frame = ledgers[name]
        values = frame.portfolio_value.to_numpy()
        dates = pd.to_datetime(frame.date)
        axes[0].plot(dates, values, label=name, linewidth=1.5)
        axes[1].plot(dates, values / np.maximum.accumulate(values) - 1, linewidth=1.1)
    axes[0].set_title("Held-out test | net of costs | next-open execution")
    axes[0].set_ylabel("Portfolio value (start = 1)")
    axes[0].legend(loc="best", ncol=3, fontsize=8)
    axes[1].set_ylabel("Drawdown")
    for axis in axes:
        axis.grid(alpha=.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _versions():
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ("numpy", "pandas", "torch", "gymnasium", "stable-baselines3", "sb3-contrib")}}


def run_experiment(data_path, output, config=None, data_kind="market", split_indices=None, resume=False):
    config = config or ExperimentConfig()
    if config.total_timesteps < 1 or not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("Positive training budget and distinct nonempty seeds required")
    if not config.risk_levels or len(set(config.risk_levels)) != len(config.risk_levels):
        raise ValueError("Distinct nonempty risk levels required")
    if not config.algorithms or len(set(config.algorithms)) != len(config.algorithms) or any(a not in ALGORITHM_CLASSES for a in config.algorithms):
        raise ValueError("Distinct nonempty algorithms from td3, ddpg, sac, ppo, a2c required")
    if not config.selector_all_algorithms and "td3" not in config.algorithms:
        raise ValueError("Algorithms without TD3 require selector_all_algorithms=True")
    for algorithm in config.algorithms:
        training_budget(algorithm, config.total_timesteps)
    torch.set_num_threads(1)
    data_path = Path(data_path).resolve()
    panel = load_panel(data_path)
    if any(t.startswith("SYN") for t in panel.tickers):
        data_kind = "synthetic"
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Output is not empty: {output}. Use a new run folder.")
    output.mkdir(parents=True, exist_ok=True)
    splits = validate_splits(split_indices or chronological_splits(len(panel.dates), config.lookback), len(panel.dates), config.lookback)
    metadata_file = data_path.with_suffix(".metadata.json")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "data_path": str(data_path),
        "data_kind": data_kind, "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "data_metadata": json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else None,
        "tickers": list(panel.tickers), "config": asdict(config), "versions": _versions(),
        "splits": {name: {"start_index": start, "end_index": end, "first_decision": panel.dates[start].isoformat(), "last_valuation": panel.dates[end].isoformat(), "transitions": end - start} for name, (start, end) in splits.items()},
        "selection_rule": "Max validation (total_return - 0.5 * max_drawdown), including cash; never use test metrics to choose.",
        "protocol": "custom chronological splits" if split_indices is not None else "50/20/15/15 chronological holdout",
        "status": "running",
    }
    if resume and (output / "manifest.json").exists():
        previous = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        # Runs created before algorithm configuration keep the legacy defaults.
        previous["config"] = json.loads(json.dumps(asdict(ExperimentConfig(**previous["config"]))))
        normalized = json.loads(json.dumps(manifest))
        for key in ("config", "data_sha256", "splits", "tickers", "versions"):
            if previous[key] != normalized[key]:
                raise ValueError(f"Resume must match the original {key}")
        manifest["created_at_utc"] = previous["created_at_utc"]
    elif resume and any(output.iterdir()):
        raise ValueError("Resume must match a run with an existing manifest")
    _write_json(output / "manifest.json", manifest)
    seed_results = {}
    for seed in config.seeds:
        directory = output / f"seed_{seed}"
        (directory / "models").mkdir(parents=True, exist_ok=True)
        models = {}
        model_details = {}
        specifications = _specifications(config)
        for i, (name, algorithm, risk) in enumerate(specifications):
            print(f"Seed {seed}: {name}, {config.total_timesteps} steps on base-train only", flush=True)
            checkpoint = directory / "models" / f"{name}.zip"
            budget = training_budget(algorithm, config.total_timesteps)
            if resume and checkpoint.exists():
                model = ALGORITHM_CLASSES[algorithm].load(checkpoint, device="cpu")
                if model.num_timesteps != budget["expected_actual_steps"] or model.seed != seed + i * 100 or model.rlasset_backend != config.backend:
                    raise ValueError(f"Checkpoint does not match run configuration: {name}")
                if getattr(model, "rlasset_training_budget", budget) != budget:
                    raise ValueError(f"Checkpoint training budget does not match: {name}")
                print(f"  Reused completed checkpoint: {name}", flush=True)
            else:
                env = _factory(panel, splits["train"], config, risk)()
                policy_kwargs = {"features_extractor_class": SharedTemporalExtractor} if config.shared_encoder else None
                model = make_agent(env, algorithm=algorithm, seed=seed + i * 100, total_timesteps=config.total_timesteps, learning_rate=config.learning_rate, policy_kwargs=policy_kwargs, backend=config.backend)
                started = perf_counter()
                model.learn(total_timesteps=config.total_timesteps, progress_bar=False)
                model.rlasset_training_wall_seconds = perf_counter() - started
                if model.num_timesteps != budget["expected_actual_steps"]:
                    raise RuntimeError(f"Unexpected training step count for {name}: {model.num_timesteps}")
                model.save(checkpoint)
            models[name] = model
            model_details[name] = {"algorithm": algorithm, "risk_aversion": risk, "seed": seed + i * 100, "num_timesteps": model.num_timesteps,
                                   "requested_steps": config.total_timesteps, "actual_steps": model.num_timesteps,
                                   "training_wall_seconds": getattr(model, "rlasset_training_wall_seconds", None),
                                   "training_budget": budget, "backend": model.rlasset_backend, "provenance": model.rlasset_provenance}
            _write_json(directory / "models.json", model_details)
        pool, _ = _pool(models, config.selector_all_algorithms)
        print(f"Seed {seed}: frozen pool -> DQN selector on meta-train only", flush=True)
        selector_path = directory / "models" / "dqn_selector.zip"
        if resume and selector_path.exists():
            selector = DQN.load(selector_path, device="cpu")
            if selector.num_timesteps != config.total_timesteps or selector.seed != seed:
                raise ValueError("Selector checkpoint does not match run configuration")
            selection_path = directory / "selection.json"
            saved_order = json.loads(selection_path.read_text(encoding="utf-8")).get("pool_order") if selection_path.exists() else None
            pool = _restore_selector_order(selector, models, config, saved_order)
        else:
            meta_env = SelectorEnv(_factory(panel, splits["meta"], config), pool)
            selector = DQN("MultiInputPolicy", meta_env, learning_rate=config.learning_rate, learning_starts=min(100, config.total_timesteps // 5), batch_size=64, buffer_size=50000, train_freq=1, gradient_steps=1, target_update_interval=100, exploration_fraction=.4, policy_kwargs={"net_arch": [64, 64]}, seed=seed, device="cpu", verbose=0)
            started = perf_counter()
            selector.learn(total_timesteps=config.total_timesteps, progress_bar=False)
            selector.rlasset_training_wall_seconds = perf_counter() - started
            selector.rlasset_pool_order = [candidate.name for candidate in pool]
            selector.save(selector_path)
        _write_json(directory / "selector.json", {"algorithm": "dqn", "seed": seed,
                    "requested_steps": config.total_timesteps, "actual_steps": selector.num_timesteps,
                    "training_wall_seconds": getattr(selector, "rlasset_training_wall_seconds", None),
                    "pool_order": [candidate.name for candidate in pool]})
        train_counts = {name: model.num_timesteps for name, model in models.items()}
        selector_train_count = selector.num_timesteps
        validation_ledgers, validation = _evaluate(panel, splits["validation"], config, models, selector, directory, "validation")
        chosen = max(validation, key=lambda name: validation[name]["total_return"] - .5 * validation[name]["max_drawdown"])
        best_rl = max(models, key=lambda name: validation[name]["total_return"] - .5 * validation[name]["max_drawdown"])
        selection = {"selected_on": "validation", "strategy": chosen, "criterion": "total_return - 0.5 * max_drawdown", "validation_metrics": validation[chosen], "pool_order": [x.name for x in pool],
                     "best_rl_on_validation": best_rl, "best_rl_validation_metrics": validation[best_rl]}
        _write_json(directory / "selection.json", selection)
        ledgers, test = _evaluate(panel, splits["test"], config, models, selector, directory, "test")
        _save_ledger(validation_ledgers[best_rl], directory / "validation_best_rl_on_validation.csv", panel.tickers)
        _save_ledger(ledgers[best_rl], directory / "test_best_rl_on_validation.csv", panel.tickers)
        assert train_counts == {name: model.num_timesteps for name, model in models.items()}
        assert selector_train_count == selector.num_timesteps, "Frozen DQN changed during validation/test"
        learned_names = [name for name in models if name.startswith("td3")]
        diversity = {}
        for i, first in enumerate(learned_names):
            for second in learned_names[i + 1:]:
                a = np.asarray(validation_ledgers[first].target_weights.tolist())[1:]
                b = np.asarray(validation_ledgers[second].target_weights.tolist())[1:]
                diversity[f"{first} vs {second}"] = float(np.abs(a - b).sum(axis=1).mean())
        _write_json(directory / "validation_diversity.json", {"mean_L1_distance": diversity, "interpretation": "Observed validation allocations; different risk coefficients do not guarantee diverse policies."})
        _plot(ledgers, chosen, directory / "test_equity.png")
        seed_results[str(seed)] = {"selected_strategy": chosen, "selected_test_metrics": test[chosen], "test_metrics": test, "selector_choices": ledgers["dqn_selector"].selected_policy.dropna().value_counts().to_dict(),
                                   "best_rl_on_validation": best_rl, "best_rl_test_metrics": test[best_rl]}
        print(f"Seed {seed}: validation selected {chosen}; held-out return {test[chosen]['total_return']:.2%}", flush=True)
    names = list(next(iter(seed_results.values()))["test_metrics"])
    aggregate = {}
    for name in names:
        values = [x["test_metrics"][name]["total_return"] for x in seed_results.values()]
        aggregate[name] = {"test_return_mean": float(np.mean(values)), "test_return_std_across_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    summary = {"data_kind": data_kind, "seeds": seed_results, "aggregate": aggregate, "claim": "Executable research experiment. No claim of profitable or statistically superior performance."}
    _write_json(output / "summary.json", summary)
    manifest["status"] = "complete"
    _write_json(output / "manifest.json", manifest)
    _report(output, manifest, summary)
    return summary


def _report(output, manifest, summary):
    algorithms = manifest["config"].get("algorithms", ["td3", "ddpg"])
    pool_description = "全部已训练基础RL模型" if manifest["config"].get("selector_all_algorithms", False) else "仅TD3风险偏好模型（旧版顺序）"
    lines = ["# 实验结果", "", f"数据：{manifest['data_kind']}；资产：{', '.join(manifest['tickers'])}；每个基础模型及选择器请求预算：{manifest['config']['total_timesteps']} steps。", "", f"基础算法：{', '.join(algorithms)}；TD3（若启用）风险参数：{manifest['config']['risk_levels']}；另训练DQN选择器。所有测试模型均冻结。各模型实际步数与训练秒数见 seed 目录 models.json / selector.json，on-policy 算法仅在完整 rollout 边界结束。协议：{manifest['protocol']}。各阶段从现金开始，边界跨期收益舍弃。", "", f"选择器池：{pool_description}，随后为 momentum、reversal、cash；确切顺序见 selection.json 的 pool_order。equal_mixture 只混合该模式的基础RL候选。基础模型初始化沿用 master seed + 模型序号 × 100，各真实seed保存在 models.json；并非所有算法使用同一个初始化seed。", "", "## 时间划分", "", "| 阶段 | 首次决策 | 最后估值 | 收益期数 |", "|---|---|---|---:|"]
    for name, split in manifest["splits"].items():
        lines.append(f"| {name} | {split['first_decision'][:10]} | {split['last_valuation'][:10]} | {split['transitions']} |")
    lines += ["", "## 测试收益（扣费）", "", "| 策略 | 多种子平均累计收益 | 种子间标准差 |", "|---|---:|---:|"]
    for name, result in summary["aggregate"].items():
        std = result["test_return_std_across_seeds"]
        lines.append(f"| {name} | {result['test_return_mean']:.2%} | {'不适用' if std is None else f'{std:.2%}'} |")
    lines += ["", "## 验证集选出的策略", ""]
    for seed, result in summary["seeds"].items():
        metric = result["selected_test_metrics"]
        lines += [f"- seed {seed}：{result['selected_strategy']}，测试累计收益 {metric['total_return']:.2%}，最大回撤 {metric['max_drawdown']:.2%}。", f"  图：[净值与回撤](seed_{seed}/test_equity.png)。"]
        lines.append(f"  仅在基础RL模型中按同一验证规则选择：{result['best_rl_on_validation']}；固定后测试收益 {result['best_rl_test_metrics']['total_return']:.2%}。")
    lines += ["", "## 解释范围", "", "本次是有限训练预算下的研究实验；能跑通、单段收益较高均不能证明样本外优势。不同种子共用市场历史，不是独立市场样本；测试表仅比较，不用于再挑参数。固定存续ETF与当前复权快照存在选择/时点限制。连续仓位没有整数手、最低佣金、涨跌停、停牌与订单成交限制，不能作为实盘结果。", "", "详见项目 docs/research、docs/design.md 与 docs/limitations.md。manifest.json 保存数据哈希、软件版本、配置和划分；seed目录保存模型、验证选择、每日权重与净收益。"]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate_saved(run, seed=7, cost_bps=None):
    """Reload frozen checkpoints; verify data fingerprint and reproduce test returns."""
    torch.set_num_threads(1)
    run = Path(run).resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    path = Path(manifest["data_path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["data_sha256"]:
        raise ValueError("Market snapshot has changed; evaluation refused")
    config = ExperimentConfig(**manifest["config"])
    if cost_bps is not None:
        config = replace(config, cost_bps=float(cost_bps))
    panel = load_panel(path)
    directory = run / f"seed_{seed}"
    details = json.loads((directory / "models.json").read_text(encoding="utf-8"))
    models = {name: ALGORITHM_CLASSES[entry["algorithm"]].load(directory / "models" / name, device="cpu") for name, entry in details.items()}
    selector = DQN.load(directory / "models" / "dqn_selector", device="cpu")
    selection_path = directory / "selection.json"
    saved_order = json.loads(selection_path.read_text(encoding="utf-8")).get("pool_order") if selection_path.exists() else None
    _restore_selector_order(selector, models, config, saved_order)
    split = manifest["splits"]["test"]
    train_counts = {name: model.num_timesteps for name, model in models.items()}
    selector_train_count = selector.num_timesteps
    _, metrics = _evaluate(panel, (split["start_index"], split["end_index"]), config, models, selector)
    assert train_counts == {name: model.num_timesteps for name, model in models.items()}
    assert selector_train_count == selector.num_timesteps, "Frozen DQN changed during evaluation"
    filename = "reloaded_test_metrics.json" if cost_bps is None else f"stress_{float(cost_bps):g}bps_metrics.json"
    _write_json(directory / filename, metrics)
    return metrics
