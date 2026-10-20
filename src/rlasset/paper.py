"""Frozen-policy daily replay through an integer-share paper broker."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_panel
from .env import PortfolioEnv
from .execution import ExecutionBar, ExecutionConfig, PaperBroker
from .meta import SelectorEnv
from .metrics import performance_metrics


class PaperPolicyDriver:
    """Signal data and actual broker holdings at decision close; causal shadows."""
    def __init__(self, panel, start, end, lookback, max_weight, cost_bps,
                 candidate=None, selector=None, pool=None):
        self.panel, self.start, self.end = panel, start, end
        self.candidate, self.selector, self.pool = candidate, selector, pool
        factory = lambda: PortfolioEnv(panel, start, end, lookback=lookback,
                                      max_weight=max_weight, cost_bps=cost_bps)
        self.env = factory()
        self.context = SelectorEnv(factory, pool) if selector is not None else None
        if self.context:
            self.context.reset()

    def decide(self, t, actual_weights):
        self.env.t = t
        self.env.weights = np.asarray(actual_weights, dtype=float)
        observation = self.env._observation()
        if self.selector is None:
            return self.candidate.weights(self.env, observation), self.candidate.name
        action, _ = self.selector.predict(self.context._augment(observation), deterministic=True)
        selected = self.pool[int(np.asarray(action).item())]
        return selected.weights(self.env, observation), selected.name

    def advance_shadows(self):
        if self.context is None:
            return
        for i, (candidate, shadow) in enumerate(zip(self.pool, self.context.shadows)):
            target = candidate.weights(shadow, self.context.shadow_obs[i])
            obs, _, _, _, info = shadow.step_weights(target)
            self.context.shadow_obs[i] = obs
            self.context.performance_history[i].append(info["net_return"])


def _bars(panel, index, scenario="normal", offset=0):
    bars = {}
    for j, ticker in enumerate(panel.tickers):
        # Approximate bands; exchange ex-distribution reference prices absent.
        band = .20 if ticker == "159915" else .10
        previous = panel.close[max(0, index - 1), j]
        up, down = round(previous * (1 + band), 3), round(previous * (1 - band), 3)
        halt = False
        volume = float(panel.volume[index, j])
        if scenario == "disruption" and offset > 0:
            # Synthetic disturbances on declared dates, never inferred from profit.
            halt = offset % 17 == 0
            if offset % 11 == 0:
                volume = min(volume, 1000)
            if offset % 23 == 0:
                up = float(panel.open[index, j])
            if offset % 29 == 0:
                down = float(panel.open[index, j])
        bars[ticker] = ExecutionBar(str(panel.dates[index].date()), ticker,
                                   float(panel.open[index, j]), float(panel.close[index, j]),
                                   volume, halted=halt, limit_up=up, limit_down=down)
    return bars


def simulate_policy(raw_panel, driver, initial_cash=1_000_000,
                    config=None, scenario="normal"):
    if scenario not in {"normal", "disruption"}:
        raise ValueError("scenario must be normal or disruption")
    if (raw_panel.tickers != driver.panel.tickers or
            not raw_panel.dates.equals(driver.panel.dates)):
        raise ValueError("Signal and execution calendars/tickers must match exactly")
    broker = PaperBroker(initial_cash, config=config or ExecutionConfig(slippage_bps=2))
    first = broker.close_snapshot(_bars(raw_panel, driver.start))
    first.update(net_return=0., portfolio_value=1., turnover=0., selected_policy="initial_cash")
    ledger = [first]
    for t in range(driver.start, driver.end):
        actual = [ledger[-1]["weights"].get(key, 0.) for key in ("cash", *raw_panel.tickers)]
        target, selected = driver.decide(t, actual)
        bars = _bars(raw_panel, t + 1, scenario, t + 1 - driver.start)
        opening_nav = broker.cash + sum(q * bars[tic].open for tic, q in broker.positions.items())
        orders = []
        if target is not None:
            target = np.asarray(target, dtype=float)
            if (target.shape != (len(raw_panel.tickers) + 1,) or
                    not np.isfinite(target).all() or (target < 0).any() or
                    not np.isclose(target.sum(), 1, atol=1e-7, rtol=0)):
                raise ValueError("Policy produced invalid target weights")
            if (target[1:] > driver.env.max_weight + 1e-7).any():
                raise ValueError("Policy target weights exceed max_weight")
            # Match the research environment's simplex-roundoff treatment.
            # The broker requires risky weights <= 1 without float32 drift.
            target = target / target.sum()
            targets = {tic: float(target[i + 1]) for i, tic in enumerate(raw_panel.tickers)}
            orders = broker.rebalance(targets, bars,
                                      decision_date=str(raw_panel.dates[t].date()),
                                      client_order_prefix=f"{scenario}:{t}")
        snapshot = broker.close_snapshot(bars)
        snapshot.update(
            decision_date=str(raw_panel.dates[t].date()), selected_policy=selected,
            net_return=snapshot["nav"] / ledger[-1]["nav"] - 1,
            portfolio_value=snapshot["nav"] / initial_cash,
            turnover=sum(order.notional for order in orders) / opening_nav,
            target_weights=None if target is None else dict(zip(("cash", *raw_panel.tickers), target.tolist())),
        )
        if snapshot["cash"] < -1e-8 or any(q < 0 for q in snapshot["positions"].values()):
            raise AssertionError("Broker violated long-only cash constraints")
        ledger.append(snapshot)
        driver.advance_shadows()
    records = [order.to_dict() for order in broker.orders]
    status_counts = pd.Series([o["status"] for o in records], dtype=str).value_counts().to_dict()
    rejection_counts = pd.Series([o["reject_reason"] for o in records if o["reject_reason"]], dtype=str).value_counts().to_dict()
    requested = sum(o["requested_quantity"] for o in records)
    summary = {
        "scenario": scenario, "initial_cash": initial_cash, "final_nav": ledger[-1]["nav"],
        "metrics": performance_metrics([r["net_return"] for r in ledger[1:]],
                                       turnover=[r["turnover"] for r in ledger[1:]]),
        "orders": len(records), "status_counts": status_counts, "reasons": rejection_counts,
        "quantity_fill_ratio": sum(o["filled_quantity"] for o in records) / requested if requested else None,
        "commission": ledger[-1]["cumulative_commission"],
        "slippage_cost": ledger[-1]["cumulative_slippage"],
        "minimum_cash": min(r["cash"] for r in ledger), "negative_cash_or_position": False,
        "config": asdict(broker.config),
        "claim": "Offline unadjusted price-only paper replay. No cash distributions, tick queue, broker connectivity or actual live fills.",
        "assumptions": [
            "Target weights decided at prior close; share sizes use observed next open.",
            "IOC fills at open plus slippage, capped by a fraction of full-day volume (ex post liquidity proxy, not executable auction proof).",
            "Price limits approximate previous raw close; ex-distribution reference prices are not available.",
            "Shadow policy statistics remain fractional adjusted-price research ledgers; actual policy receives paper-account holdings.",
            "Disruption scenario injects synthetic halts, thin volume and directional price-limit blocks; these are not historical exchange events.",
        ],
    }
    return {"summary": summary, "ledger": ledger, "orders": records}


def validate_execution_snapshot(path):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if metadata.get("adjust") != "none":
        raise ValueError("Paper execution requires explicit unadjusted price provenance")
    if metadata.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Execution snapshot hash mismatch")
    if metadata.get("volume_unit") != "shares":
        raise ValueError("Execution volume_unit must explicitly be shares")
    return metadata


def choose_consensus_rl(run):
    """Choose one algorithm by mean validation score; no seed or test selection."""
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    scores = {}
    expected_names = None
    for seed in manifest["config"]["seeds"]:
        directory = run / f"seed_{seed}"
        names = list(json.loads((directory / "models.json").read_text(encoding="utf-8")))
        if expected_names is not None and names != expected_names:
            raise ValueError("All seeds must share the same ordered algorithm candidates")
        expected_names = names
        validation = json.loads((directory / "validation_metrics.json").read_text(encoding="utf-8"))
        for name in names:
            score = validation[name]["total_return"] - .5 * validation[name]["max_drawdown"]
            if not np.isfinite(score):
                raise ValueError("Nonfinite validation score")
            scores.setdefault(name, []).append(score)
    if not scores:
        raise ValueError("No validation candidates")
    return max(scores, key=lambda name: float(np.mean(scores[name])))


def replay_run(run, raw_data, output, seed=7, scenario="normal", initial_cash=1_000_000):
    import torch
    from stable_baselines3 import DQN
    from .experiment import ExperimentConfig, _pool
    from .finrl_adapter import ALGORITHM_CLASSES
    torch.set_num_threads(1)
    run, output = Path(run), Path(output)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    signal_path = Path(manifest["data_path"])
    if hashlib.sha256(signal_path.read_bytes()).hexdigest() != manifest["data_sha256"]:
        raise ValueError("Signal snapshot hash mismatch")
    raw_metadata = validate_execution_snapshot(raw_data)
    signal, raw = load_panel(signal_path), load_panel(raw_data)
    config = ExperimentConfig(**manifest["config"])
    directory = run / f"seed_{seed}"
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    details = json.loads((directory / "models.json").read_text(encoding="utf-8"))
    checkpoints = list((directory / "models").glob("*.zip"))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints}
    models = {name: ALGORITHM_CLASSES[item["algorithm"]].load(directory / "models" / name, device="cpu")
              for name, item in details.items()}
    selector = DQN.load(directory / "models" / "dqn_selector", device="cpu")
    pool, candidates = _pool(models, config.selector_all_algorithms, selection["pool_order"])
    split = manifest["splits"]["test"]
    consensus = choose_consensus_rl(run)
    results = {}
    for name in ["best_rl_on_validation", "consensus_on_validation", "selected_on_validation", "dqn_selector", "equal_weight"]:
        policy_name = (selection["best_rl_on_validation"] if name == "best_rl_on_validation"
                       else consensus if name == "consensus_on_validation"
                       else selection["strategy"] if name == "selected_on_validation" else name)
        driver = PaperPolicyDriver(signal, split["start_index"], split["end_index"],
                                   config.lookback, config.max_weight, config.cost_bps,
                                   candidate=candidates.get(policy_name),
                                   selector=selector if policy_name == "dqn_selector" else None, pool=pool)
        result = simulate_policy(raw, driver, initial_cash=initial_cash, scenario=scenario)
        target = output / f"seed_{seed}" / scenario / name
        target.mkdir(parents=True, exist_ok=True)
        result["summary"].update(policy=policy_name, selected_on="validation", seed=seed,
                                 signal_data_sha256=manifest["data_sha256"],
                                 execution_data_sha256=raw_metadata["sha256"], model_hashes=hashes)
        (target / "ledger.json").write_text(json.dumps(result["ledger"], indent=2, allow_nan=False), encoding="utf-8")
        pd.DataFrame(result["orders"]).to_csv(target / "orders.csv", index=False)
        (target / "summary.json").write_text(json.dumps(result["summary"], indent=2, allow_nan=False), encoding="utf-8")
        results[name] = result["summary"]
    if hashes != {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints}:
        raise AssertionError("Frozen model checkpoints changed during execution")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Offline historical paper replay, no broker orders")
    parser.add_argument("--run", required=True)
    parser.add_argument("--raw-data", default="data/processed/ashare_etf_unadjusted.csv")
    parser.add_argument("--output", default="runs/paper_v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    args = parser.parse_args()
    all_results = {}
    for seed in args.seeds:
        all_results[str(seed)] = {}
        for scenario in ("normal", "disruption"):
            print(f"Paper replay seed {seed}: {scenario}", flush=True)
            all_results[str(seed)][scenario] = replay_run(args.run, args.raw_data, args.output, seed, scenario)
    (Path(args.output) / "summary.json").write_text(json.dumps(all_results, indent=2, allow_nan=False), encoding="utf-8")
