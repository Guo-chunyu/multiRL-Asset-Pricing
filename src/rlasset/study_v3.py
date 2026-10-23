"""Frozen, validation-selected research using raw prices and corporate actions.

Stages are deliberately separate: prepare -> fit (all seeds) -> select -> evaluate.
No test ledger is read by the fitting or selection stages.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .data import load_panel
from .metrics import performance_metrics

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = ('legacy_ppo', 'robust_ppo', 'robust_sac', 'robust_td3')
SEEDS = (7, 17, 29)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def calendar_folds(dates):
    dates = pd.DatetimeIndex(dates)
    if not dates.is_unique or not dates.is_monotonic_increasing or dates[0].year > 2015 or dates[-1].year < 2026:
        raise ValueError('Need unique ordered calendar covering 2015 through 2026')
    result = []
    for number, (first, last) in enumerate(((2019, 2020), (2021, 2022), (2023, 2024), (2025, 2026)), 1):
        def interval(mask):
            indices = np.flatnonzero(mask)
            if len(indices) < 2:
                raise ValueError('Empty or too short calendar interval')
            return [int(indices[0]), int(indices[-1])]
        train = interval(dates.year <= first - 2)
        train[0] = 120
        validation = interval(dates.year == first - 1)
        test = interval((dates.year >= first) & (dates.year <= last))
        if train[1] - train[0] < 126:
            raise ValueError('Insufficient training history')
        result.append({'name': f'fold_{number:02}', 'train': train, 'validation': validation, 'test': test})
    return result


def select_candidate(validation):
    if not validation or len({len(v) for v in validation.values()}) != 1 or not all(validation.values()):
        raise ValueError('All candidates require the same nonempty seed count')
    scores = {}
    for name, rows in validation.items():
        values = np.asarray([[x['total_return'], x['max_drawdown']] for x in rows], float)
        if not np.isfinite(values).all() or np.any(values[:, 1] < 0):
            raise ValueError('Invalid validation metrics')
        scores[name] = float(np.mean(values[:, 0] - .5 * values[:, 1]))
    return {'candidate': max(scores, key=scores.get), 'scores': scores,
            'criterion': 'mean_seed(total_return - 0.5 * max_drawdown)', 'selected_on': 'validation'}


def constant_weights(n_assets, exposure):
    if n_assets < 2 or not 0 <= exposure <= 1:
        raise ValueError('Require >=2 assets and exposure in [0,1]')
    return np.r_[1 - exposure, np.full(n_assets, exposure / n_assets)]


def load_inputs(data_dir):
    from .corporate_actions import CorporateActions
    data_dir = Path(data_dir)
    panel = load_panel(data_dir / 'ashare_broad_raw.csv')
    actions = CorporateActions.from_csv(panel, data_dir / 'cash_actions.csv', data_dir / 'splits.csv')
    return panel, actions


def identity(data_dir, steps):
    files = [Path(data_dir) / name for name in ('ashare_broad_raw.csv', 'cash_actions.csv', 'splits.csv')]
    source = [Path(__file__), *[Path(__file__).with_name(name) for name in
              ('corporate_actions.py', 'robust_env.py', 'env.py', 'data.py', 'finrl_adapter.py', 'extractors.py', 'metrics.py')],
              ROOT / 'docs/research-protocol-v3.md']
    return {'data': {p.name: digest(p) for p in files}, 'sources': {p.name: digest(p) for p in source},
            'steps': steps, 'seeds': list(SEEDS), 'candidates': list(CANDIDATES)}


def prepare(data_dir, output, steps):
    panel, _ = load_inputs(data_dir)
    output = Path(output)
    expected = identity(data_dir, steps)
    path = output / 'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest['identity'] != expected:
            raise ValueError('Existing experiment identity differs; use a fresh output directory')
        return manifest
    folds = calendar_folds(panel.dates)
    for fold in folds:
        fold['dates'] = {name: [str(panel.dates[a].date()), str(panel.dates[b].date())]
                         for name, (a, b) in ((k, fold[k]) for k in ('train', 'validation', 'test'))}
    manifest = {'identity': expected, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'tickers': list(panel.tickers), 'sessions': len(panel.dates), 'folds': folds,
                'claim': 'Retrospective exploratory study, not pristine out-of-sample or live performance'}
    write_json(path, manifest)
    return manifest


def checked_manifest(data_dir, output, steps):
    manifest = json.loads((Path(output) / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['identity'] != identity(data_dir, steps):
        raise ValueError('Data/code/protocol changed since experiment preparation')
    return manifest


def make_env(panel, actions, interval, candidate, *, training=False, cost_bps=5):
    from .corporate_actions import CorporateActionPortfolioEnv
    from .robust_env import RobustPortfolioEnv
    kwargs = dict(panel=panel, actions=actions, start=interval[0], end=interval[1], cost_bps=cost_bps, risk_aversion=2.)
    if candidate == 'legacy_ppo':
        return CorporateActionPortfolioEnv(**kwargs, lookback=20)
    return RobustPortfolioEnv(**kwargs, training=training)


def run_policy(env, policy):
    observation, _ = env.reset()
    done = False
    while not done:
        target = policy(env, observation)
        observation, _, terminated, truncated, _ = env.step_weights(target)
        done = terminated or truncated
    ledger = pd.DataFrame(env.history)
    metrics = performance_metrics(ledger.net_return.to_numpy()[1:], ledger.turnover.to_numpy()[1:])
    metrics['mean_cash_weight'] = float(np.mean([row[0] for row in ledger.weights.iloc[1:]]))
    metrics['return_days'] = len(ledger) - 1
    return ledger, metrics


def save_ledger(path, ledger):
    serial = ledger.copy()
    for column in serial:
        if serial[column].map(lambda x: isinstance(x, (list, tuple, dict, np.ndarray))).any():
            serial[column] = serial[column].map(lambda x: json.dumps(x.tolist() if isinstance(x, np.ndarray) else x))
    serial.to_csv(path, index=False)


def model_policy(model):
    return lambda env, obs: env.action_to_weights(model.predict(obs, deterministic=True)[0])


def fit_seed(data_dir, output, steps, seed):
    import torch
    from .finrl_adapter import make_agent
    from .extractors import SharedTemporalExtractor
    if seed not in SEEDS:
        raise ValueError('Seed not in registered plan')
    torch.set_num_threads(1)
    manifest = checked_manifest(data_dir, output, steps)
    panel, actions = load_inputs(data_dir)
    for fold in manifest['folds']:
        directory = Path(output) / fold['name'] / f'seed_{seed}'
        directory.mkdir(parents=True, exist_ok=True)
        for candidate in CANDIDATES:
            record_path = directory / f'{candidate}.json'
            model_path = directory / f'{candidate}.zip'
            if record_path.exists():
                record = json.loads(record_path.read_text(encoding='utf-8'))
                if digest(model_path) != record['model_sha256']:
                    raise ValueError('Model checksum mismatch')
                print(f'Reuse {fold["name"]} seed={seed} {candidate}', flush=True)
                continue
            legacy = candidate == 'legacy_ppo'
            env = make_env(panel, actions, fold['train'], candidate, training=True)
            model = make_agent(env, algorithm=candidate.split('_')[1], seed=seed, total_timesteps=steps,
                               learning_rate=3e-4 if legacy else 1e-4,
                               policy='MultiInputPolicy' if legacy else 'MlpPolicy',
                               policy_kwargs={'features_extractor_class': SharedTemporalExtractor} if legacy else None)
            start = perf_counter()
            model.learn(total_timesteps=steps, progress_bar=False)
            elapsed = perf_counter() - start
            model.save(model_path)
            ledger, metrics = run_policy(make_env(panel, actions, fold['validation'], candidate), model_policy(model))
            save_ledger(directory / f'validation_{candidate}.csv', ledger)
            record = {'candidate': candidate, 'seed': seed, 'actual_steps': int(model.num_timesteps),
                      'requested_steps': steps, 'training_seconds': elapsed,
                      'model_sha256': digest(model_path), 'validation_metrics': metrics,
                      'backend': model.rlasset_backend, 'provenance': model.rlasset_provenance}
            write_json(record_path, record)
            print(f'{fold["name"]} seed={seed} {candidate}: {model.num_timesteps} steps {elapsed:.1f}s; validation={metrics["total_return"]:.4f}', flush=True)


def freeze_selection(data_dir, output, steps):
    manifest = checked_manifest(data_dir, output, steps)
    selections = {}
    for fold in manifest['folds']:
        validation = {}
        hashes = {}
        for candidate in CANDIDATES:
            validation[candidate] = []
            for seed in SEEDS:
                directory = Path(output) / fold['name'] / f'seed_{seed}'
                record = json.loads((directory / f'{candidate}.json').read_text(encoding='utf-8'))
                if record['model_sha256'] != digest(directory / f'{candidate}.zip'):
                    raise ValueError('Model checksum mismatch')
                validation[candidate].append(record['validation_metrics'])
                hashes[f'{seed}/{candidate}'] = record['model_sha256']
        selections[fold['name']] = {**select_candidate(validation), 'model_hashes': hashes}
    path = Path(output) / 'frozen_selection.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))['folds'] != selections:
        raise ValueError('Frozen selection cannot change')
    if not path.exists():
        write_json(path, {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'folds': selections})
    return selections


def evaluate(data_dir, output, steps, cost_bps):
    import torch
    from .finrl_adapter import ALGORITHM_CLASSES
    torch.set_num_threads(1)
    manifest = checked_manifest(data_dir, output, steps)
    frozen_path = Path(output) / 'frozen_selection.json'
    if not frozen_path.exists():
        raise ValueError('Freeze all validation selections before opening any test')
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))['folds']
    panel, actions = load_inputs(data_dir)
    result = {'cost_bps': cost_bps, 'folds': {}}
    for fold in manifest['folds']:
        folder = Path(output) / f'evaluation_{cost_bps:g}bps' / fold['name']
        folder.mkdir(parents=True, exist_ok=True)
        records = {}
        for seed in SEEDS:
            models = {}
            for candidate in CANDIDATES:
                path = Path(output) / fold['name'] / f'seed_{seed}' / f'{candidate}.zip'
                if digest(path) != frozen[fold['name']]['model_hashes'][f'{seed}/{candidate}']:
                    raise ValueError('Frozen model hash changed')
                models[candidate] = ALGORITHM_CLASSES[candidate.split('_')[1]].load(path, device='cpu')
            seed_metrics = {}
            for name, model in models.items():
                ledger, metrics = run_policy(make_env(panel, actions, fold['test'], name, cost_bps=cost_bps), model_policy(model))
                save_ledger(folder / f'seed_{seed}_{name}.csv', ledger)
                seed_metrics[name] = metrics
            def mixture(env, obs):
                return np.mean([env.action_to_weights(models[name].predict(obs, deterministic=True)[0])
                                for name in CANDIDATES if name.startswith('robust_')], axis=0)
            ledger, metrics = run_policy(make_env(panel, actions, fold['test'], 'robust_ppo', cost_bps=cost_bps), mixture)
            save_ledger(folder / f'seed_{seed}_robust_mixture.csv', ledger)
            seed_metrics['robust_mixture'] = metrics
            selected = frozen[fold['name']]['candidate']
            seed_metrics['selected_on_validation'] = seed_metrics[selected]
            records[str(seed)] = seed_metrics
        baselines = {}
        for name, exposure in (('equal_80', .8), ('equal_90', .9), ('equal_100', 1.), ('buy_hold', 1.), ('inverse_vol_90', .9)):
            env = make_env(panel, actions, fold['test'], 'robust_ppo', cost_bps=cost_bps)
            def policy(env, obs, name=name, exposure=exposure):
                if name == 'buy_hold' and len(env.history) > 1:
                    return None
                if name == 'inverse_vol_90':
                    from .robust_env import capped_risky_weights
                    returns = np.diff(np.log(env.signal_close[env.t - 60:env.t + 1]), axis=0)
                    inv = 1 / np.maximum(returns.std(axis=0, ddof=1), 1e-6)
                    weights = capped_risky_weights(inv, exposure, .5)
                    return np.r_[1 - exposure, weights]
                return constant_weights(env.n_assets, exposure)
            ledger, metrics = run_policy(env, policy)
            save_ledger(folder / f'{name}.csv', ledger)
            baselines[name] = metrics
        result['folds'][fold['name']] = {'selected': frozen[fold['name']]['candidate'], 'seeds': records, 'baselines': baselines}
        print(f'Evaluated {fold["name"]} at {cost_bps:g}bps', flush=True)
    write_json(Path(output) / f'evaluation_{cost_bps:g}bps' / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'fit', 'select', 'evaluate'], required=True)
    parser.add_argument('--data-dir', default='data/v3')
    parser.add_argument('--output', default='runs/research_v3')
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--cost-bps', type=float, default=5)
    args = parser.parse_args()
    if args.stage == 'prepare': prepare(args.data_dir, args.output, args.steps)
    elif args.stage == 'fit': fit_seed(args.data_dir, args.output, args.steps, args.seed)
    elif args.stage == 'select': freeze_selection(args.data_dir, args.output, args.steps)
    else: evaluate(args.data_dir, args.output, args.steps, args.cost_bps)


if __name__ == '__main__':
    main()
