"""Controlled fixed-expert selector study: prepare, fit, freeze, then evaluate.

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
                       save_ledger,
                       write_json)

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = ('lognet_alltrue', 'lognet_phase', 'drawdown_alltrue', 'drawdown_phase')
PPO_CONFIG = dict(learning_rate=1e-4, n_steps=100, batch_size=50, n_epochs=10,
                  gamma=.99, gae_lambda=.95, clip_range=.2, clip_range_vf=None,
                  normalize_advantage=True, ent_coef=0., vf_coef=.5,
                  max_grad_norm=.5, target_kl=None, policy_kwargs={'net_arch': [64, 64]})
SEEDS = (7, 17, 29)
EXPERTS = ('cash', 'equal50', 'vol10', 'trend10')
BASELINES = (*EXPERTS, 'equal_target_mix', 'trailing20')
RULE_CANDIDATES = BASELINES
COSTS = (5, 10, 20)


def candidate_config(candidate):
    if candidate not in CANDIDATES:
        raise ValueError(f'Unregistered candidate: {candidate}')
    return False, False, 'ppo'


def choose_candidates(validation, rules):
    if set(validation) != set(CANDIDATES) or set(rules) != set(RULE_CANDIDATES):
        raise ValueError('All registered candidates and rules are mandatory')
    if any(len(rows) != len(SEEDS) for rows in validation.values()):
        raise ValueError('Each RL candidate requires all registered seeds')
    def score(row):
        value = float(row['total_return'] - .5 * row['max_drawdown'])
        if not np.isfinite(value):
            raise ValueError('Nonfinite validation score')
        return value
    scores = {name: float(np.mean([score(row) for row in validation[name]])) for name in CANDIDATES}
    rule_scores = {name: score(rules[name]) for name in BASELINES}
    combined = {**scores, **rule_scores}
    candidate = max(CANDIDATES, key=scores.get)
    deployed = max(combined, key=combined.get)
    best_expert = max(EXPERTS, key=rule_scores.get)
    return {'candidate': candidate, 'scores': scores, 'selected_on': 'validation',
            'deploy_candidate': deployed, 'deploy_scores': combined,
            'deploy_kind': 'rule' if deployed in rules else 'rl',
            'best_validation_fixed_expert': best_expert,
            'rule_scores': rule_scores, 'rule_replicates': 1}


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
        raise ValueError('Formal v7 requires exactly 20000 steps; synthetic smoke tests are separate')
    require_certified_data(data_dir)
    data_files = [Path(data_dir) / name for name in ('ashare_broad_raw.csv', 'cash_actions.csv', 'splits.csv', 'restoration_verification.json', 'restoration_audit.json')]
    sources = [Path(__file__), *[Path(__file__).with_name(name) for name in
               ('study_v3.py', 'study_v4.py', 'study_v5.py', 'statistics.py', 'selector_env_v6.py', 'masked_selector_v7.py', 'study_v6.py', 'residual_env.py', 'risk_env.py', 'signal_diagnostics.py', 'corporate_actions.py', 'robust_env.py',
                'env.py', 'data.py', 'finrl_adapter.py', 'metrics.py')],
               ROOT / 'docs/research-protocol-v7.md', ROOT / 'scripts/audit_research_v3.py', ROOT / 'tests/test_study_v7.py', ROOT / 'tests/test_masked_selector_v7.py']
    return {'data': {path.name: digest(path) for path in data_files},
            'sources': {path.name: digest(path) for path in sources},
            'steps': steps, 'seeds': list(SEEDS), 'candidates': list(CANDIDATES),
            'baselines': list(BASELINES), 'cost_bps': list(COSTS),
            'backend': 'sb3_contrib.MaskablePPO==2.7.1', 'ppo_config': PPO_CONFIG,
            'daily_market_transitions': 960000, 'training_fits': 48}


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
    from .masked_selector_v7 import MaskedSelectorEnvV7
    candidate_config(candidate)
    return MaskedSelectorEnvV7(panel=panel, actions=actions, start=interval[0], end=interval[1],
        training=training, cost_bps=cost_bps,
        drawdown_penalty=candidate.startswith('drawdown'),
        mask_mode='phase' if candidate.endswith('_phase') else 'control')


def make_maskable_agent(env, seed):
    import sb3_contrib
    from sb3_contrib import MaskablePPO
    if sb3_contrib.__version__ != '2.7.1':
        raise ValueError('Official sb3-contrib==2.7.1 required')
    model = MaskablePPO('MlpPolicy', env, seed=seed, device='cpu', verbose=0, **deepcopy(PPO_CONFIG))
    model.rlasset_backend = 'sb3_contrib.MaskablePPO'
    model.rlasset_provenance = {'version': sb3_contrib.__version__, 'ppo_config': deepcopy(PPO_CONFIG),
        'training_entry': 'official sb3_contrib.MaskablePPO directly, not FinRL',
        'effective_v6_entropy': 0.0, 'nominal_finrl_entropy_not_merged': 0.01}
    return model


def baseline_policy(name):
    if name not in BASELINES:
        raise ValueError(f'Unregistered baseline: {name}')
    def policy(env, observation):
        if name in EXPERTS:
            return env.fixed_expert_action(name)
        if name == 'equal_target_mix':
            return env.equal_target_mix()
        return env.trailing_argmax()
    return policy


def model_policy(model):
    return lambda env, obs: int(np.asarray(model.predict(obs, deterministic=True, action_masks=env.action_masks())[0]).item())


def run_policy(env, policy):
    observation, _ = env.reset()
    done = False
    while not done:
        action = policy(env, observation)
        if np.asarray(action).ndim == 0:
            observation, _, terminated, truncated, _ = env.step(int(action))
        else:
            observation, _, terminated, truncated, _ = env.step_weights(action)
        done = terminated or truncated
    ledger = pd.DataFrame(env.history)
    metrics = performance_metrics(ledger.net_return.to_numpy()[1:], ledger.turnover.to_numpy()[1:])
    metrics['mean_cash_weight'] = float(np.mean([row[0] for row in ledger.weights.iloc[1:]]))
    metrics['return_days'] = len(ledger) - 1
    metrics.update(execution_metrics(ledger))
    return ledger, metrics


def execution_metrics(frame):
    executed = frame.loc[frame['rebalance_executed'].fillna(False).astype(str).str.lower().eq('true')]
    names = executed['executed_expert'].dropna().tolist() if 'executed_expert' in executed else []
    result = {'executed_rebalances': len(executed),
              'executed_expert_switches': sum(a != b for a, b in zip(names, names[1:])),
              'sum_daily_cost_fraction': float(frame['cost_fraction'].fillna(0).sum())}
    for name in EXPERTS:
        result['executed_fraction_' + name] = names.count(name) / len(names) if names else 0.
    return result


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
    metrics.update(execution_metrics(frame))
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
    expected_schedule = episode_schedule(fold['train'], seed, steps)
    if record.get('episode_schedule') != expected_schedule:
        raise ValueError('Actual sampled training episode schedule differs from paired plan')
    if record['backend'] != 'sb3_contrib.MaskablePPO' or record['provenance']['ppo_config'] != PPO_CONFIG:
        raise ValueError('Wrong backend or PPO configuration')
    return record


def episode_schedule(interval, seed, steps):
    rng = np.random.default_rng(seed)
    return [[int(start), int(start) + 126] for start in
            rng.integers(interval[0], interval[1] - 126 + 1, size=steps // 126 + 1)]


def fit_seed(data_dir, output, steps, seed, fold_name=None):
    import torch
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
            model = make_maskable_agent(env, seed)
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
                                    'episode_schedule': env.training_episode_schedule,
                                    'training_seconds': elapsed, 'model_sha256': digest(model_path),
                                    'validation_sha256': digest(validation_path), 'validation_metrics': metrics,
                                    'backend': model.rlasset_backend, 'provenance': model.rlasset_provenance})
            write_json(record_path, record)
            checked_record(directory, candidate, seed, fold, panel, steps)
            print(f'{fold["name"]} seed={seed} {candidate}: {model.num_timesteps} steps {elapsed:.1f}s; validation={metrics["total_return"]:.4f}', flush=True)


def _validation_rules(panel, actions, fold, output, data_dir):
    directory = Path(output) / fold['name'] / 'validation_baselines'
    directory.mkdir(parents=True, exist_ok=True)
    metrics, ledger_hashes, record_hashes = {}, {}, {}
    start, end = fold['validation']
    for name in RULE_CANDIDATES:
        ledger_path, record_path = directory / f'{name}.csv', directory / f'{name}.json'
        if not record_path.exists():
            env = make_env(panel, actions, fold['validation'], 'lognet_alltrue')
            ledger, values = run_policy(env, baseline_policy(name))
            save_ledger(ledger_path, ledger)
            audit_saved_ledger(ledger_path, data_dir)
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
        rules, baseline_validation_hashes, baseline_record_hashes = _validation_rules(panel, actions, fold, output, data_dir)
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
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(1)
    manifest = checked_manifest(data_dir, output, steps)
    panel, actions = load_inputs(data_dir)
    result = {'cost_bps': cost_bps, 'folds': {}}
    for fold in manifest['folds']:
        folder = Path(output) / f'evaluation_{cost_bps:g}bps' / fold['name']
        folder.mkdir(parents=True, exist_ok=True)
        baselines = {}
        for name in BASELINES:
            env = make_env(panel, actions, fold['test'], 'lognet_alltrue', cost_bps=cost_bps)
            ledger, metrics = run_policy(env, baseline_policy(name))
            save_ledger(folder / f'{name}.csv', ledger)
            audit_saved_ledger(folder / f'{name}.csv', data_dir, cost_bps)
            baselines[name] = metrics
        records = {}
        for seed in SEEDS:
            models = {}
            for candidate in CANDIDATES:
                model_path = Path(output) / fold['name'] / f'seed_{seed}' / f'{candidate}.zip'
                models[candidate] = MaskablePPO.load(model_path, device='cpu')
            seed_metrics = {}
            for name, model in models.items():
                ledger, metrics = run_policy(make_env(panel, actions, fold['test'], name, cost_bps=cost_bps), model_policy(model))
                save_ledger(folder / f'seed_{seed}_{name}.csv', ledger)
                audit_saved_ledger(folder / f'seed_{seed}_{name}.csv', data_dir, cost_bps)
                seed_metrics[name] = metrics
            selected, deployed = frozen[fold['name']]['candidate'], frozen[fold['name']]['deploy_candidate']
            seed_metrics['selected_on_validation'] = seed_metrics[selected]
            if deployed not in baselines:
                seed_metrics['selected_with_rules'] = seed_metrics[deployed]
            records[str(seed)] = seed_metrics
        result['folds'][fold['name']] = {'selected': frozen[fold['name']]['candidate'],
            'deploy_selected': frozen[fold['name']]['deploy_candidate'],
            'deploy_kind': frozen[fold['name']]['deploy_kind'],
            'deployment_rule_metrics': baselines.get(frozen[fold['name']]['deploy_candidate']),
            'seeds': records, 'baselines': baselines}
        print(f'Evaluated {fold["name"]} at {cost_bps:g}bps', flush=True)
    write_json(Path(output) / f'evaluation_{cost_bps:g}bps' / 'summary.json', result)
    return result


def audit_saved_ledger(path, data_dir, cost_bps=5):
    """Independent raw-open accounting, not the environment's own bookkeeping."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('v7_independent_accountant', ROOT / 'scripts/audit_research_v3.py')
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
                ledger, values = run_policy(make_env(panel, actions, fold[split], 'lognet_alltrue'), baseline_policy(name))
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
                'complete_models': 0, 'expected_models': 48}
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
    """All paths and prespecified contrasts; deterministic rules are single paths."""
    from .statistics import compare_seeded_returns
    output = Path(output)
    manifest = checked_manifest(data_dir, output, steps)
    frozen = freeze_selection(data_dir, output, steps)
    panel, _ = load_inputs(data_dir)
    records, contrasts, selection_rows = [], [], []
    for cost in COSTS:
        folder = output / f'evaluation_{cost}bps'
        summary = json.loads((folder / 'summary.json').read_text())
        for fold in manifest['folds']:
            name = fold['name']
            start, end = fold['test']
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
                    ledger = pd.read_csv(path)
                    paths.append(ledger.net_return.to_numpy()[1:])
                    for col in ('selected_expert', 'requested_expert', 'executed_expert', 'action_ignored'):
                        if col in ledger:
                            for value, count in ledger[col].value_counts(dropna=False).items():
                                selection_rows.append(dict(cost_bps=cost, fold=name, candidate=candidate, seed=seed,
                                                           field=col, value=value, count=int(count)))
                anchors = ('equal_target_mix', 'trailing20', frozen[name]['best_validation_fixed_expert'])
                for anchor in anchors:
                    comparison = compare_seeded_returns(paths, baseline_returns[anchor])
                    contrasts.append(dict(cost_bps=cost, fold=name, candidate=candidate, anchor=anchor,
                                          selected_on_validation=candidate == frozen[name]['candidate'],
                                          comparison=comparison['comparison']))
    paired = paired_mask_contrasts(output, manifest, panel)
    write_json(output / 'paired_mask_contrasts.json', paired)
    pd.DataFrame(records).to_csv(output / 'all_metrics.csv', index=False)
    pd.DataFrame(selection_rows).to_csv(output / 'selection_counts.csv', index=False)
    write_json(output / 'primary_contrasts.json', contrasts)
    lines = ['# v7 causal action-mask controlled experiment', '',
             'Retrospective exploratory historical development, not untouched OOS or live efficacy.',
             'Official sb3-contrib 2.7.1 MaskablePPO for both all-true and phase-mask arms; not a FinRL training entry.',
             '48 fits × 20,000 DAILY market transitions = 960,000; no macrostepping.',
             'Effective v6 ent_coef=0.0: nominal FinRL .01 is not merged when adapter supplies kwargs.',
             'All 48 checkpoints and all validation selections were frozen before any v7 test evaluation.',
             'Validation-only baseline previews were observed before training; the registered family was not changed.',
             'Costs are one-way bps. Seeds describe training randomness, not independent market histories.',
             'Deterministic rules have one path per fold/cost, not three independent seed replicates.', '',
             '| Cost | Fold | Strategy | Return mean (seed range) | Max drawdown mean | Mean equity exposure | Daily turnover |',
             '|---:|---|---|---:|---:|---:|---:|']
    frame = pd.DataFrame(records)
    for (cost, fold, strategy), part in frame.groupby(['cost_bps', 'fold', 'strategy'], sort=False):
        r = part.total_return
        lines.append(f'| {cost} | {fold} | {strategy} | {r.mean():.2%} ({r.min():.2%}–{r.max():.2%}) | {part.max_drawdown.mean():.2%} | {1-part.mean_cash_weight.mean():.2%} | {part.mean_turnover.mean():.4f} |')
    selection_metrics = []
    lines += ['', '## Frozen validation selections', '',
              '| Fold | Learned-only choice | With-rules choice |', '|---|---|---|']
    for fold_name, selection in frozen.items():
        lines.append(f"| {fold_name} | {selection['candidate']} | {selection['deploy_candidate']} |")
        for cost in COSTS:
            for label, key in [('selected_on_validation', 'candidate'), ('selected_with_rules', 'deploy_candidate')]:
                chosen = frame[(frame.fold == fold_name) & (frame.cost_bps == cost) & (frame.strategy == selection[key])]
                for row in chosen.to_dict('records'):
                    selection_metrics.append({**row, 'selection': label, 'chosen_strategy': selection[key]})
    pd.DataFrame(selection_metrics).to_csv(output / 'selected_metrics.csv', index=False)
    lines += ['', '## Paired masking effects (phase minus alltrue)', '',
              '| Cost | Fold | Objective | Mean paired total-return difference | Annualized arithmetic excess (95% block interval) |',
              '|---:|---|---|---:|---:|']
    for item in paired:
        c = item['comparison']
        low, high = c['annualized_mean_excess_ci']
        delta = np.mean([row['phase_minus_alltrue_total_return'] for row in item['paired_seeds']])
        lines.append(f"| {item['cost_bps']} | {item['fold']} | {item['objective']} | {delta:.2%} | {c['annualized_mean_excess_return']:.2%} ({low:.2%}, {high:.2%}) |")
    lines += ['', '## Executed choices and costs', '',
              'Frequencies below use executed selections only. Ignored dummy requests are never allocations.', '',
              '| Cost | Fold | Candidate | Cash | Equal50 | Vol10 | Trend10 | Mean switches | Sum daily cost fractions |',
              '|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for (cost, fold_name, strategy), part in frame[frame.strategy.isin(CANDIDATES)].groupby(['cost_bps','fold','strategy'], sort=False):
        frequencies = ' | '.join(f"{part['executed_fraction_'+name].mean():.1%}" for name in EXPERTS)
        lines.append(f"| {cost} | {fold_name} | {strategy} | {frequencies} | {part.executed_expert_switches.mean():.1f} | {part.sum_daily_cost_fraction.mean():.3%} |")
    lines += ['', 'Primary controlled contrasts: phase minus alltrue separately for lognet and drawdown; every seed/fold/cost is retained.',
              'Additional prespecified contrasts: equal-target mix, trailing-20 shadow selector, and the best validation-fixed expert.',
              'Moving-block intervals use 10-day blocks / 2000 draws, concern annualized arithmetic excess (not CAGR), and are exploratory without multiple-testing correction.',
              'All models and all six comparators are reported; no test winner is selected.',
              'Every saved actual-account test ledger was independently reconstructed from raw corporate-action data.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return records, contrasts


def paired_mask_contrasts(output, manifest, panel):
    from .statistics import paired_block_bootstrap
    result = []
    for cost in COSTS:
        for fold in manifest['folds']:
            folder = Path(output) / f'evaluation_{cost}bps' / fold['name']
            for objective in ('lognet', 'drawdown'):
                paths, differences = {}, []
                for arm in ('alltrue', 'phase'):
                    paths[arm] = []
                    for seed in SEEDS:
                        path = folder / f'seed_{seed}_{objective}_{arm}.csv'
                        start, end = fold['test']
                        validation_metrics(path, panel.dates[start:end + 1])
                        paths[arm].append(pd.read_csv(path).net_return.to_numpy()[1:])
                for index, seed in enumerate(SEEDS):
                    differences.append({'seed': seed, 'phase_minus_alltrue_total_return': float(
                        np.prod(1 + paths['phase'][index]) - np.prod(1 + paths['alltrue'][index]))})
                result.append({'cost_bps': cost, 'fold': fold['name'], 'objective': objective,
                    'direction': 'phase minus alltrue', 'paired_seeds': differences,
                    'comparison': paired_block_bootstrap(np.mean(paths['phase'], axis=0),
                                                        np.mean(paths['alltrue'], axis=0))})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'baselines', 'fit', 'select', 'evaluate', 'status', 'report'], required=True)
    parser.add_argument('--data-dir', default='data/v3')
    parser.add_argument('--output', default='runs/research_v7')
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
