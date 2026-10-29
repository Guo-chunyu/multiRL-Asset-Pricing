"""Controlled bounded residual-risk study: prepare, fit, freeze, then evaluate.

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
CANDIDATES = ('v4risk_ppo', 'residual_vol_ppo', 'residual_trend_ppo')
SEEDS = (7, 17, 29)
BASELINES = ('equal_50', 'equal_100', 'cash', 'vol_target10', 'trend_target10')
RULE_CANDIDATES = ('vol_target10', 'trend_target10')
COSTS = (5, 10, 20)


def candidate_config(candidate):
    if candidate not in CANDIDATES:
        raise ValueError(f'Unregistered candidate: {candidate}')
    return False, candidate != 'v4risk_ppo', 'ppo'


def choose_candidates(validation, rules):
    if set(validation) != set(CANDIDATES) or set(rules) != set(RULE_CANDIDATES):
        raise ValueError('All registered candidates and rules are mandatory')
    if any(len(rows) != len(SEEDS) for rows in validation.values()):
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


def require_certified_data(data_dir):
    path = Path(data_dir) / 'restoration_verification.json'
    if not path.is_file():
        raise ValueError('Data not certified: restoration_verification.json missing; certified restoration evidence required')
    record = json.loads(path.read_text())
    if record.get('certified_ready') is not True:
        raise ValueError('Data not certified: approved raw-price/action evidence incomplete')
    for name in ('ashare_broad_raw.csv', 'cash_actions.csv', 'splits.csv'):
        if record.get('input_sha256', {}).get(name) != digest(Path(data_dir) / name):
            raise ValueError(f'Certified data checksum mismatch: {name}')
    return record


def identity(data_dir, steps):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps != 20000:
        raise ValueError('Formal v5 requires exactly 20000 steps; synthetic smoke tests are separate')
    require_certified_data(data_dir)
    data_files = [Path(data_dir) / name for name in ('ashare_broad_raw.csv', 'cash_actions.csv', 'splits.csv', 'restoration_verification.json', 'restoration_audit.json')]
    sources = [Path(__file__), *[Path(__file__).with_name(name) for name in
               ('study_v3.py', 'study_v4.py', 'statistics.py', 'residual_env.py', 'risk_env.py', 'signal_diagnostics.py', 'corporate_actions.py', 'robust_env.py',
                'env.py', 'data.py', 'finrl_adapter.py', 'metrics.py')],
               ROOT / 'docs/research-protocol-v5.md', ROOT / 'scripts/audit_research_v3.py']
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
    certified_identity = identity(data_dir, steps)
    panel, _ = load_inputs(data_dir)
    manifest = {'identity': certified_identity,
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'tickers': list(panel.tickers), 'sessions': len(panel.dates),
                'folds': _folds(panel),
                'data_snapshot': 'Restored historical inputs; certification states exact hash equivalence modulo line endings and any fresh verification limitations',
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
    from .residual_env import ResidualRiskEnv
    candidate_config(candidate)
    kwargs = dict(panel=panel, actions=actions, start=interval[0], end=interval[1],
                  training=training, cost_bps=cost_bps)
    if candidate == 'v4risk_ppo':
        return InformationRiskEnv(**kwargs, enhanced_features=False, risk_control=True)
    return ResidualRiskEnv(**kwargs, trend_gate=candidate == 'residual_trend_ppo')


def baseline_policy(name):
    if name not in BASELINES:
        raise ValueError(f'Unregistered baseline: {name}')
    def policy(env, observation):
        if name == 'cash':
            return constant_weights(env.n_assets, 0.)
        if name.startswith('equal_'):
            return constant_weights(env.n_assets, int(name.split('_')[1]) / 100)
        from .residual_env import trend_volatility_target
        return trend_volatility_target(env.signal_close, env.t,
                                       trend_gate=name == 'trend_target10')
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
    baseline_file = Path(output) / 'baseline_preview' / 'summary.json'
    if not baseline_file.exists():
        raise ValueError('Run the preregistered no-training baseline stage before fitting')
    preview = json.loads(baseline_file.read_text())
    check_record_signature(preview)
    if preview['manifest_sha256'] != digest(Path(output) / 'manifest.json'):
        raise ValueError('Baseline preview manifest changed')
    for relative, expected in preview['ledger_hashes'].items():
        if digest(Path(output) / relative) != expected:
            raise ValueError('Baseline preview ledger changed')
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
            audit_saved_ledger(validation_path, data_dir)
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
            env = make_env(panel, actions, fold['validation'], 'residual_vol_ppo')
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
            env = make_env(panel, actions, fold['test'], 'residual_vol_ppo', cost_bps=cost_bps)
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


def audit_saved_ledger(path, data_dir, cost_bps=5):
    """Independent raw-open accounting, not the environment's own bookkeeping."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('v5_independent_accountant', ROOT / 'scripts/audit_research_v3.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data_dir = Path(data_dir)
    outcome = module.audit_ledger(pd.read_csv(path),
        pd.read_csv(data_dir / 'ashare_broad_raw.csv', dtype={'tic': str}),
        pd.read_csv(data_dir / 'cash_actions.csv', dtype={'ticker': str}),
        pd.read_csv(data_dir / 'splits.csv', dtype={'ticker': str}), cost_bps=cost_bps)
    write_json(Path(path).with_suffix('.audit.json'), {'ledger_sha256': digest(path), 'result': outcome})
    return outcome


def baseline_stage(data_dir, output, steps):
    """Validation-only preview fixed before fitting; cannot change registry."""
    manifest = checked_manifest(data_dir, output, steps)
    panel, actions = load_inputs(data_dir)
    result = {'manifest_sha256': digest(Path(output) / 'manifest.json'),
              'claim': 'Validation-only pretraining baselines; no test read or efficacy selection',
              'folds': {}, 'ledger_hashes': {}}
    for fold in manifest['folds']:
        result['folds'][fold['name']] = {}
        for split in ('validation',):
            folder = Path(output) / 'baseline_preview' / fold['name'] / split
            folder.mkdir(parents=True, exist_ok=True)
            metrics = {}
            for name in BASELINES:
                ledger, values = run_policy(make_env(panel, actions, fold[split], 'residual_vol_ppo'), baseline_policy(name))
                path = folder / f'{name}.csv'
                save_ledger(path, ledger)
                audit_saved_ledger(path, data_dir)
                metrics[name] = values
                result['ledger_hashes'][str(path.relative_to(output))] = digest(path)
            result['folds'][fold['name']][split] = metrics
        print(f"Baseline preview {fold['name']}: {result['folds'][fold['name']]['validation']}", flush=True)
    write_json(Path(output) / 'baseline_preview/summary.json', signed_record(result))
    return result


def status(data_dir, output, steps):
    """Resume status fails closed on altered artifacts; no performance ranking."""
    output = Path(output)
    if not (output / 'manifest.json').exists():
        missing = [name for name in ('ashare_broad_raw.csv', 'cash_actions.csv', 'splits.csv')
                   if not (Path(data_dir) / name).is_file()]
        certification = None
        if not missing:
            try:
                require_certified_data(data_dir)
            except (ValueError, FileNotFoundError) as exc:
                certification = str(exc)
        return {'state': 'blocked_data' if missing or certification else 'unprepared',
                'certification_blocker': certification, 'missing_inputs': missing,
                'complete_models': 0, 'expected_models': 36}
    manifest = checked_manifest(data_dir, output, steps)
    panel, _ = load_inputs(data_dir)
    done = []
    for fold in manifest['folds']:
        for seed in SEEDS:
            directory = output / fold['name'] / f'seed_{seed}'
            for candidate in CANDIDATES:
                if (directory / f'{candidate}.json').exists():
                    checked_record(directory, candidate, seed, fold, panel, steps)
                    done.append(f"{fold['name']}/{seed}/{candidate}")
    frozen = (output / 'frozen_selection.json').exists()
    if frozen:
        freeze_selection(data_dir, output, steps)
    costs = [cost for cost in COSTS if (output / f'evaluation_{cost}bps/summary.json').exists()]
    return {'state': 'evaluated' if len(costs) == len(COSTS) else 'frozen' if frozen else 'fitting' if done else 'prepared',
            'complete_models': len(done), 'expected_models': len(manifest['folds']) * len(SEEDS) * len(CANDIDATES),
            'completed': done, 'evaluated_costs': costs}


def report(data_dir, output, steps):
    """Report all seed paths and paired increment, never select a test winner."""
    from .statistics import compare_seeded_returns
    output = Path(output)
    manifest = checked_manifest(data_dir, output, steps)
    freeze_selection(data_dir, output, steps)
    records, contrasts = [], []
    for cost in COSTS:
        folder = output / f'evaluation_{cost}bps'
        summary = json.loads((folder / 'summary.json').read_text())
        for fold in manifest['folds']:
            name = fold['name']
            start, end = fold['test']
            panel, _ = load_inputs(data_dir)
            expected = panel.dates[start:end + 1]
            baseline_returns = {}
            for baseline in BASELINES:
                path = folder / name / f'{baseline}.csv'
                metrics = validation_metrics(path, expected)
                _check_metrics(summary['folds'][name]['baselines'][baseline], metrics)
                audit_saved_ledger(path, data_dir, cost)
                baseline_returns[baseline] = pd.read_csv(path).net_return.to_numpy()[1:]
                records.append(dict(cost_bps=cost, fold=name, strategy=baseline, seed=None, **metrics))
            for candidate in CANDIDATES:
                paths = []
                for seed in SEEDS:
                    path = folder / name / f'seed_{seed}_{candidate}.csv'
                    metrics = validation_metrics(path, expected)
                    _check_metrics(summary['folds'][name]['seeds'][str(seed)][candidate], metrics)
                    audit_saved_ledger(path, data_dir, cost)
                    records.append(dict(cost_bps=cost, fold=name, strategy=candidate, seed=seed, **metrics))
                    paths.append(pd.read_csv(path).net_return.to_numpy()[1:])
                anchor = 'trend_target10' if candidate == 'residual_trend_ppo' else 'vol_target10'
                comparison = compare_seeded_returns(paths, baseline_returns[anchor])
                contrasts.append(dict(cost_bps=cost, fold=name, candidate=candidate, anchor=anchor,
                                      comparison=comparison['comparison']))
    pd.DataFrame(records).to_csv(output / 'all_metrics.csv', index=False)
    write_json(output / 'anchor_contrasts.json', contrasts)
    lines = ['# v5 bounded residual-risk experiment', '',
             'Retrospective exploratory historical development, not untouched OOS or live efficacy.',
             'All 36 checkpoints and validation selections frozen before learned test evaluation.',
             'Validation-only baseline previews were observed before training; the registered family was not changed.',
             'Costs are one-way bps. Seeds describe training randomness, not independent market histories.', '',
             '| Cost | Fold | Strategy | Return mean (seed range) | Max drawdown mean | Mean equity exposure | Daily turnover |',
             '|---:|---|---|---:|---:|---:|---:|']
    frame = pd.DataFrame(records)
    for (cost, fold, strategy), part in frame.groupby(['cost_bps', 'fold', 'strategy'], sort=False):
        r = part.total_return
        lines.append(f'| {cost} | {fold} | {strategy} | {r.mean():.2%} ({r.min():.2%}–{r.max():.2%}) | {part.max_drawdown.mean():.2%} | {1-part.mean_cash_weight.mean():.2%} | {part.mean_turnover.mean():.4f} |')
    lines += ['', 'Paired anchor increments and 10-day moving-block bootstrap intervals (2000 draws) are in `anchor_contrasts.json`.',
              'Intervals concern annualized arithmetic mean excess, not CAGR; no multiple-comparison correction.',
              'No historical candidate is a promise of future profit. Compare residuals with their own zero-action anchors, not only fixed exposure.',
              'Raw next-open trades, costs, dividends/receivables/payments, splits and NAV were independently reconstructed for every saved ledger.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return records, contrasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'baselines', 'fit', 'select', 'evaluate', 'status', 'report'], required=True)
    parser.add_argument('--data-dir', default='data/v3')
    parser.add_argument('--output', default='runs/research_v5')
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--seed', type=int, choices=SEEDS, default=7)
    parser.add_argument('--fold', choices=[f'fold_{number:02}' for number in range(1, 5)])
    parser.add_argument('--cost-bps', type=float, choices=COSTS, default=5)
    args = parser.parse_args()
    if args.fold is not None and args.stage != 'fit':
        parser.error('--fold is only a training shard; selection/evaluation always require every fold')
    if args.stage == 'prepare': prepare(args.data_dir, args.output, args.steps)
    elif args.stage == 'baselines': baseline_stage(args.data_dir, args.output, args.steps)
    elif args.stage == 'status': print(json.dumps(status(args.data_dir, args.output, args.steps), indent=2))
    elif args.stage == 'report': report(args.data_dir, args.output, args.steps)
    elif args.stage == 'fit': fit_seed(args.data_dir, args.output, args.steps, args.seed, args.fold)
    elif args.stage == 'select': freeze_selection(args.data_dir, args.output, args.steps)
    else: evaluate(args.data_dir, args.output, args.steps, args.cost_bps)


if __name__ == '__main__':
    main()
