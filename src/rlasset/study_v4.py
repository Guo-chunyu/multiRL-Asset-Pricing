"""Factorial information/risk study: prepare, fit, freeze, then evaluate.

The historical test periods were already inspected in v3. These experiments
are retrospective development evidence, not a fresh out-of-sample discovery.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .metrics import performance_metrics
from .study_v3 import (calendar_folds, constant_weights, digest, load_inputs,
                       model_policy, run_policy, save_ledger, select_candidate,
                       write_json)

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = ('base_ppo', 'signal_ppo', 'risk_ppo', 'joint_ppo', 'joint_sac', 'joint_td3')
SEEDS = (7, 17, 29)
BASELINES = ('equal_50', 'equal_90', 'equal_100', 'buy_hold', 'vol_target10', 'invvol_target10')
RULE_CANDIDATES = ('vol_target10', 'invvol_target10')
COSTS = (5, 10, 20)


def candidate_config(candidate):
    if candidate not in CANDIDATES:
        raise ValueError(f'Unregistered candidate: {candidate}')
    family, algorithm = candidate.split('_')
    return family in ('signal', 'joint'), family in ('risk', 'joint'), algorithm


def choose_candidates(validation, rules):
    if not validation or any(len(rows) != len(SEEDS) for rows in validation.values()):
        raise ValueError('Each RL candidate requires all registered seeds')
    learned = select_candidate(validation)
    combined = dict(validation)
    combined.update({name: [metrics] * len(SEEDS) for name, metrics in rules.items()})
    deployed = select_candidate(combined)
    return {**learned, 'deploy_candidate': deployed['candidate'],
            'deploy_scores': deployed['scores'],
            'deploy_kind': 'rule' if deployed['candidate'] in rules else 'rl'}


def _record_signature(record):
    value = {key: item for key, item in record.items() if key != 'record_sha256'}
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def signed_record(record):
    result = deepcopy(record)
    result['record_sha256'] = _record_signature(result)
    return result


def check_record_signature(record):
    if record.get('record_sha256') != _record_signature(record):
        raise ValueError('Training/validation record content changed')


def selected_folds(manifest, fold_name=None):
    if fold_name is None:
        return manifest['folds']
    result = [fold for fold in manifest['folds'] if fold['name'] == fold_name]
    if len(result) != 1:
        raise ValueError(f'Unregistered fold: {fold_name}')
    return result


def identity(data_dir, steps):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError('steps must be an integer >= 2')
    data_files = [Path(data_dir) / name for name in ('ashare_broad_raw.csv', 'cash_actions.csv', 'splits.csv')]
    sources = [Path(__file__), *[Path(__file__).with_name(name) for name in
               ('study_v3.py', 'risk_env.py', 'signal_diagnostics.py', 'corporate_actions.py', 'robust_env.py',
                'env.py', 'data.py', 'finrl_adapter.py', 'metrics.py')],
               ROOT / 'docs/research-protocol-v4.md']
    return {'data': {path.name: digest(path) for path in data_files},
            'sources': {path.name: digest(path) for path in sources},
            'steps': steps, 'seeds': list(SEEDS), 'candidates': list(CANDIDATES),
            'baselines': list(BASELINES), 'cost_bps': list(COSTS)}


def _folds(panel):
    result = calendar_folds(panel.dates)
    for fold in result:
        fold['dates'] = {name: [str(panel.dates[a].date()), str(panel.dates[b].date())]
                         for name, (a, b) in ((key, fold[key]) for key in ('train', 'validation', 'test'))}
    return result


def prepare(data_dir, output, steps):
    output = Path(output)
    path = output / 'manifest.json'
    if path.exists():
        return checked_manifest(data_dir, output, steps)
    panel, _ = load_inputs(data_dir)
    manifest = {'identity': identity(data_dir, steps),
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'tickers': list(panel.tickers), 'sessions': len(panel.dates),
                'folds': _folds(panel),
                'claim': 'Retrospective exploratory development on previously inspected historical periods; not pristine out-of-sample or live performance'}
    write_json(path, manifest)
    return manifest


def checked_manifest(data_dir, output, steps):
    manifest = json.loads((Path(output) / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['identity'] != identity(data_dir, steps):
        raise ValueError('Data/code/protocol changed since experiment preparation')
    panel, _ = load_inputs(data_dir)
    if (manifest['folds'] != _folds(panel) or manifest['sessions'] != len(panel.dates)
            or manifest['tickers'] != list(panel.tickers)):
        raise ValueError('Registered manifest calendar changed')
    return manifest


def make_env(panel, actions, interval, candidate, *, training=False, cost_bps=5):
    from .risk_env import InformationRiskEnv
    enhanced, risk, _ = candidate_config(candidate)
    return InformationRiskEnv(panel=panel, actions=actions, start=interval[0], end=interval[1],
                              training=training, cost_bps=cost_bps,
                              enhanced_features=enhanced, risk_control=risk)


def baseline_policy(name):
    if name not in BASELINES:
        raise ValueError(f'Unregistered baseline: {name}')
    def policy(env, observation):
        if name == 'buy_hold':
            return constant_weights(env.n_assets, 1.) if len(env.history) == 1 else None
        if name.startswith('equal_'):
            return constant_weights(env.n_assets, int(name.split('_')[1]) / 100)
        scores = np.ones(env.n_assets)
        if name == 'invvol_target10':
            returns = np.diff(np.log(env.signal_close[env.t - 60:env.t + 1]), axis=0)
            scores = 1 / np.maximum(returns.std(axis=0, ddof=1), 1e-6)
        return env.risk_target(scores, budget=.10)
    return policy


def validation_metrics(path, expected_dates):
    frame = pd.read_csv(path)
    required = ('date', 'net_return', 'portfolio_value', 'turnover', 'weights')
    if not set(required) <= set(frame):
        raise ValueError('Validation ledger lacks required fields')
    dates = pd.DatetimeIndex(pd.to_datetime(frame.date, errors='raise', format='mixed'))
    if not dates.equals(pd.DatetimeIndex(expected_dates)) or not dates.is_unique:
        raise ValueError('Validation ledger calendar changed')
    values = frame[['net_return', 'portfolio_value', 'turnover']].to_numpy(float)
    if not np.isfinite(values).all() or len(frame) < 2 or np.any(values[:, 1] <= 0) or np.any(values[:, 2] < 0):
        raise ValueError('Invalid validation account values')
    expected_nav = np.cumprod(1 + values[:, 0])
    if values[0, 0] != 0 or not np.allclose(values[:, 1], expected_nav, atol=1e-12, rtol=1e-12):
        raise ValueError('Validation NAV does not reconcile with returns')
    weights = np.asarray([json.loads(item) for item in frame.weights], float)
    if (weights.ndim != 2 or not np.isfinite(weights).all() or np.any(weights < -1e-12)
            or not np.allclose(weights.sum(axis=1), 1, atol=1e-10, rtol=0)):
        raise ValueError('Invalid validation holdings')
    metrics = performance_metrics(values[1:, 0], values[1:, 2])
    metrics['mean_cash_weight'] = float(weights[1:, 0].mean())
    metrics['return_days'] = len(frame) - 1
    return metrics


def _check_metrics(stored, actual):
    def equal(first, second):
        if first is None or second is None:
            return first is None and second is None
        return bool(np.isclose(first, second, atol=1e-11, rtol=1e-11))
    if set(stored) != set(actual) or any(not equal(stored[key], actual[key]) for key in actual):
        raise ValueError('Validation metrics disagree with ledger')


def checked_record(directory, candidate, seed, fold, panel, steps):
    from .finrl_adapter import training_budget
    directory = Path(directory)
    record = json.loads((directory / f'{candidate}.json').read_text(encoding='utf-8'))
    check_record_signature(record)
    expected_steps = training_budget(candidate_config(candidate)[2], steps)['expected_actual_steps']
    if (record['candidate'] != candidate or record['seed'] != seed or record['fold'] != fold['name']
            or record['requested_steps'] != steps or record['actual_steps'] != expected_steps):
        raise ValueError('Training record identity or budget mismatch')
    if digest(directory / f'{candidate}.zip') != record['model_sha256']:
        raise ValueError('Model checksum mismatch')
    path = directory / f'validation_{candidate}.csv'
    if digest(path) != record['validation_sha256']:
        raise ValueError('Validation ledger checksum mismatch')
    start, end = fold['validation']
    _check_metrics(record['validation_metrics'], validation_metrics(path, panel.dates[start:end + 1]))
    return record


def fit_seed(data_dir, output, steps, seed, fold_name=None):
    import torch
    from .finrl_adapter import make_agent
    if seed not in SEEDS:
        raise ValueError('Seed not in registered plan')
    torch.set_num_threads(1)
    manifest = checked_manifest(data_dir, output, steps)
    panel, actions = load_inputs(data_dir)
    for fold in selected_folds(manifest, fold_name):
        directory = Path(output) / fold['name'] / f'seed_{seed}'
        directory.mkdir(parents=True, exist_ok=True)
        for candidate in CANDIDATES:
            record_path = directory / f'{candidate}.json'
            if record_path.exists():
                checked_record(directory, candidate, seed, fold, panel, steps)
                print(f'Reuse {fold["name"]} seed={seed} {candidate}', flush=True)
                continue
            model_path = directory / f'{candidate}.zip'
            env = make_env(panel, actions, fold['train'], candidate, training=True)
            model = make_agent(env, algorithm=candidate_config(candidate)[2], seed=seed,
                               total_timesteps=steps, learning_rate=1e-4, policy='MlpPolicy')
            started = perf_counter()
            model.learn(total_timesteps=steps, progress_bar=False)
            elapsed = perf_counter() - started
            model.save(model_path)
            ledger, metrics = run_policy(make_env(panel, actions, fold['validation'], candidate), model_policy(model))
            validation_path = directory / f'validation_{candidate}.csv'
            save_ledger(validation_path, ledger)
            record = signed_record({'candidate': candidate, 'seed': seed, 'fold': fold['name'],
                                    'actual_steps': int(model.num_timesteps), 'requested_steps': steps,
                                    'training_seconds': elapsed, 'model_sha256': digest(model_path),
                                    'validation_sha256': digest(validation_path), 'validation_metrics': metrics,
                                    'backend': model.rlasset_backend, 'provenance': model.rlasset_provenance})
            write_json(record_path, record)
            checked_record(directory, candidate, seed, fold, panel, steps)
            print(f'{fold["name"]} seed={seed} {candidate}: {model.num_timesteps} steps {elapsed:.1f}s; validation={metrics["total_return"]:.4f}', flush=True)


def _validation_rules(panel, actions, fold, output):
    directory = Path(output) / fold['name'] / 'validation_baselines'
    directory.mkdir(parents=True, exist_ok=True)
    metrics, ledger_hashes, record_hashes = {}, {}, {}
    start, end = fold['validation']
    for name in RULE_CANDIDATES:
        ledger_path, record_path = directory / f'{name}.csv', directory / f'{name}.json'
        if not record_path.exists():
            env = make_env(panel, actions, fold['validation'], 'joint_ppo')
            ledger, values = run_policy(env, baseline_policy(name))
            save_ledger(ledger_path, ledger)
            write_json(record_path, signed_record({'candidate': name, 'fold': fold['name'],
                       'kind': 'rule', 'validation_sha256': digest(ledger_path), 'validation_metrics': values}))
        record = json.loads(record_path.read_text(encoding='utf-8'))
        check_record_signature(record)
        if (record['candidate'] != name or record['fold'] != fold['name'] or record['kind'] != 'rule'
                or record['validation_sha256'] != digest(ledger_path)):
            raise ValueError('Rule validation identity or checksum mismatch')
        _check_metrics(record['validation_metrics'], validation_metrics(ledger_path, panel.dates[start:end + 1]))
        metrics[name] = record['validation_metrics']
        ledger_hashes[name], record_hashes[name] = digest(ledger_path), digest(record_path)
    return metrics, ledger_hashes, record_hashes


def freeze_selection(data_dir, output, steps):
    manifest = checked_manifest(data_dir, output, steps)
    panel, actions = load_inputs(data_dir)
    selections = {}
    # All candidates and folds are mandatory, even if fit was dispatched in shards.
    for fold in manifest['folds']:
        validation, model_hashes, validation_hashes, record_hashes = {}, {}, {}, {}
        for candidate in CANDIDATES:
            validation[candidate] = []
            for seed in SEEDS:
                directory = Path(output) / fold['name'] / f'seed_{seed}'
                record = checked_record(directory, candidate, seed, fold, panel, steps)
                validation[candidate].append(record['validation_metrics'])
                key = f'{seed}/{candidate}'
                model_hashes[key], validation_hashes[key] = record['model_sha256'], record['validation_sha256']
                record_hashes[key] = digest(directory / f'{candidate}.json')
        rules, baseline_validation_hashes, baseline_record_hashes = _validation_rules(panel, actions, fold, output)
        selections[fold['name']] = {**choose_candidates(validation, rules),
            'model_hashes': model_hashes, 'validation_hashes': validation_hashes, 'record_hashes': record_hashes,
            'baseline_validation_hashes': baseline_validation_hashes, 'baseline_record_hashes': baseline_record_hashes}
    path = Path(output) / 'frozen_selection.json'
    manifest_hash = digest(Path(output) / 'manifest.json')
    if path.exists():
        stored = json.loads(path.read_text(encoding='utf-8'))
        if stored['folds'] != selections or stored['manifest_sha256'] != manifest_hash:
            raise ValueError('Frozen selection cannot change')
    else:
        write_json(path, {'created_at_utc': datetime.now(timezone.utc).isoformat(),
                         'manifest_sha256': manifest_hash, 'folds': selections})
    return selections


def evaluate(data_dir, output, steps, cost_bps):
    if cost_bps not in COSTS:
        raise ValueError('Evaluation cost must be one of the registered 5, 10, 20 bps')
    path = Path(output) / 'frozen_selection.json'
    if not path.exists():
        raise ValueError('Freeze all validation selections before opening any test')
    # Recompute selection from signed training/validation records. This also
    # verifies all frozen model, record, and ledger hashes before any test read.
    frozen = freeze_selection(data_dir, output, steps)
    import torch
    from .finrl_adapter import ALGORITHM_CLASSES
    torch.set_num_threads(1)
    manifest = checked_manifest(data_dir, output, steps)
    panel, actions = load_inputs(data_dir)
    result = {'cost_bps': cost_bps, 'folds': {}}
    for fold in manifest['folds']:
        folder = Path(output) / f'evaluation_{cost_bps:g}bps' / fold['name']
        folder.mkdir(parents=True, exist_ok=True)
        baselines = {}
        for name in BASELINES:
            env = make_env(panel, actions, fold['test'], 'joint_ppo', cost_bps=cost_bps)
            ledger, metrics = run_policy(env, baseline_policy(name))
            save_ledger(folder / f'{name}.csv', ledger)
            baselines[name] = metrics
        records = {}
        for seed in SEEDS:
            models = {}
            for candidate in CANDIDATES:
                model_path = Path(output) / fold['name'] / f'seed_{seed}' / f'{candidate}.zip'
                models[candidate] = ALGORITHM_CLASSES[candidate_config(candidate)[2]].load(model_path, device='cpu')
            seed_metrics = {}
            for name, model in models.items():
                ledger, metrics = run_policy(make_env(panel, actions, fold['test'], name, cost_bps=cost_bps), model_policy(model))
                save_ledger(folder / f'seed_{seed}_{name}.csv', ledger)
                seed_metrics[name] = metrics
            def mixture(env, observation):
                return np.mean([env.action_to_weights(models[name].predict(observation, deterministic=True)[0])
                                for name in ('joint_ppo', 'joint_sac', 'joint_td3')], axis=0)
            ledger, metrics = run_policy(make_env(panel, actions, fold['test'], 'joint_ppo', cost_bps=cost_bps), mixture)
            save_ledger(folder / f'seed_{seed}_joint_mixture.csv', ledger)
            seed_metrics['joint_mixture'] = metrics
            selected, deployed = frozen[fold['name']]['candidate'], frozen[fold['name']]['deploy_candidate']
            seed_metrics['selected_on_validation'] = seed_metrics[selected]
            seed_metrics['selected_with_rules'] = baselines[deployed] if deployed in baselines else seed_metrics[deployed]
            records[str(seed)] = seed_metrics
        result['folds'][fold['name']] = {'selected': frozen[fold['name']]['candidate'],
            'deploy_selected': frozen[fold['name']]['deploy_candidate'],
            'deploy_kind': frozen[fold['name']]['deploy_kind'], 'seeds': records, 'baselines': baselines}
        print(f'Evaluated {fold["name"]} at {cost_bps:g}bps', flush=True)
    write_json(Path(output) / f'evaluation_{cost_bps:g}bps' / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'fit', 'select', 'evaluate'], required=True)
    parser.add_argument('--data-dir', default='data/v3')
    parser.add_argument('--output', default='runs/research_v4')
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--seed', type=int, choices=SEEDS, default=7)
    parser.add_argument('--fold', choices=[f'fold_{number:02}' for number in range(1, 5)])
    parser.add_argument('--cost-bps', type=float, choices=COSTS, default=5)
    args = parser.parse_args()
    if args.fold is not None and args.stage != 'fit':
        parser.error('--fold is only a training shard; selection/evaluation always require every fold')
    if args.stage == 'prepare': prepare(args.data_dir, args.output, args.steps)
    elif args.stage == 'fit': fit_seed(args.data_dir, args.output, args.steps, args.seed, args.fold)
    elif args.stage == 'select': freeze_selection(args.data_dir, args.output, args.steps)
    else: evaluate(args.data_dir, args.output, args.steps, args.cost_bps)


if __name__ == '__main__':
    main()
