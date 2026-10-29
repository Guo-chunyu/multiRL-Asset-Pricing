"""Count-only observation scaling: immutable v7 controls and 24 new paired fits.

Historical periods are repeatedly inspected exploratory development data.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd

from . import study_v7 as old
from .study_v3 import digest, load_inputs, save_ledger, write_json

ROOT = Path(__file__).resolve().parents[2]
SEEDS = old.SEEDS
EXPERTS = old.EXPERTS
BASELINES = old.BASELINES
COSTS = old.COSTS
PPO_CONFIG = deepcopy(old.PPO_CONFIG)
CANDIDATES = ('lognet_raw', 'lognet_scaled', 'drawdown_raw', 'drawdown_scaled')
NEW_CANDIDATES = ('lognet_scaled', 'drawdown_scaled')
RAW_CANDIDATES = ('lognet_raw', 'drawdown_raw')
STEPS = 20000
signed_record = old.signed_record
check_record_signature = old.check_record_signature
validation_metrics = old.validation_metrics
_check_metrics = old._check_metrics
baseline_policy = old.baseline_policy
model_policy = old.model_policy
run_policy = old.run_policy
audit_saved_ledger = old.audit_saved_ledger


def candidate_config(candidate):
    if candidate not in CANDIDATES:
        raise ValueError('Unregistered v8 candidate')
    return {'drawdown_penalty': candidate.startswith('drawdown'),
            'count_scale': 20 if candidate.endswith('_scaled') else 1}


def choose_candidates(validation, rules):
    if set(validation) != set(CANDIDATES) or set(rules) != set(BASELINES):
        raise ValueError('All registered candidates and rules are mandatory')
    if any(len(rows) != len(SEEDS) for rows in validation.values()):
        raise ValueError('Each candidate requires all registered seeds')
    def score(row):
        value = float(row['total_return'] - .5 * row['max_drawdown'])
        if not np.isfinite(value):
            raise ValueError('Nonfinite validation score')
        return value
    scores = {name: float(np.mean([score(row) for row in validation[name]])) for name in CANDIDATES}
    rule_scores = {name: score(rules[name]) for name in BASELINES}
    combined = {**scores, **rule_scores}
    deployed = max(combined, key=combined.get)
    return {'candidate': max(CANDIDATES, key=scores.get), 'scores': scores,
            'selected_on': 'validation', 'deploy_candidate': deployed,
            'deploy_scores': combined, 'deploy_kind': 'rule' if deployed in rules else 'rl',
            'best_validation_fixed_expert': max(EXPERTS, key=rule_scores.get),
            'rule_scores': rule_scores, 'rule_replicates': 1}


def runtime_info():
    from stable_baselines3.common.utils import get_system_info
    import sb3_contrib
    info = get_system_info(print_info=False)[0]
    info['sb3-contrib'] = sb3_contrib.__version__
    # Kernel/host identity is retained elsewhere; numerical library parity is
    # the reuse gate, not an unsupported promise of cross-machine bitwise fits.
    return {key: str(value) for key, value in info.items() if key != 'OS'}


def check_checkpoint_runtime(path):
    with ZipFile(path) as archive:
        saved = archive.read('system_info.txt').decode()
        version = archive.read('_stable_baselines3_version').decode().strip()
    current = runtime_info()
    for key in ('Python', 'Stable-Baselines3', 'PyTorch', 'GPU Enabled', 'Numpy', 'Cloudpickle', 'Gymnasium'):
        if f'- {key}: {current[key]}\n' not in saved:
            raise ValueError(f'Reused checkpoint runtime mismatch: {key}')
    if version != '2.7.1' or current['sb3-contrib'] != '2.7.1':
        raise ValueError('Official sb3/sb3-contrib 2.7.1 required')
    return {'model_system_info': saved, 'verified_runtime': current}


def source_control_identity(data_dir, controls, steps):
    controls = Path(controls)
    manifest = old.checked_manifest(data_dir, controls, steps)
    panel, _ = load_inputs(data_dir)
    files = {'manifest.json': digest(controls / 'manifest.json')}
    for fold in manifest['folds']:
        for seed in SEEDS:
            directory = controls / fold['name'] / f'seed_{seed}'
            for raw in RAW_CANDIDATES:
                name = raw.replace('_raw', '_phase')
                record = old.checked_record(directory, name, seed, fold, panel, steps)
                check_checkpoint_runtime(directory / f'{name}.zip')
                if record['provenance']['version'] != '2.7.1':
                    raise ValueError('Reused control backend version changed')
                for file in (f'{name}.zip', f'{name}.json', f'validation_{name}.csv'):
                    path = directory / file
                    files[str(path.relative_to(controls))] = digest(path)
    return files


def identity(data_dir, controls, steps):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps != STEPS:
        raise ValueError('Formal v8 requires exactly 20000 daily steps')
    inherited = old.identity(data_dir, steps)
    paths = [Path(__file__), Path(__file__).with_name('scaled_selector_v8.py'),
             ROOT / 'docs/research-protocol-v8.md', ROOT / 'tests/test_study_v8.py',
             ROOT / 'tests/test_scaled_selector_v8.py']
    return {'inherited_v7_identity': inherited,
            'sources': {str(path.relative_to(ROOT)): digest(path) for path in paths},
            'controls': source_control_identity(data_dir, controls, steps),
            'runtime': runtime_info(), 'steps': steps, 'seeds': list(SEEDS),
            'candidates': list(CANDIDATES), 'new_candidates': list(NEW_CANDIDATES),
            'ppo_config': PPO_CONFIG, 'new_training_fits': 24,
            'new_daily_market_transitions': 480000, 'reused_control_fits': 24}


def prepare(data_dir, output, controls, steps):
    output = Path(output)
    if (output / 'manifest.json').exists():
        return checked_manifest(data_dir, output, controls, steps)
    panel, _ = load_inputs(data_dir)
    result = {'identity': identity(data_dir, controls, steps), 'folds': old._folds(panel),
              'tickers': list(panel.tickers), 'sessions': len(panel.dates),
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'claim': 'Previously inspected historical development; not pristine OOS or live efficacy',
              'data_snapshot': 'Exact restored original three ETFs; current independent source verification limitations inherited',
              'controls_source': str(Path(controls).resolve())}
    write_json(output / 'manifest.json', result)
    return result


def checked_manifest(data_dir, output, controls, steps):
    manifest = json.loads((Path(output) / 'manifest.json').read_text())
    if manifest['identity'] != identity(data_dir, controls, steps):
        raise ValueError('Data/code/protocol/runtime/control changed since preparation')
    panel, _ = load_inputs(data_dir)
    if manifest['folds'] != old._folds(panel) or manifest['sessions'] != len(panel.dates) or manifest['tickers'] != list(panel.tickers):
        raise ValueError('Registered calendar changed')
    return manifest


def make_env(panel, actions, interval, candidate, *, training=False, cost_bps=5):
    from .scaled_selector_v8 import ScaledSelectorEnvV8
    return ScaledSelectorEnvV8(panel=panel, actions=actions, start=interval[0], end=interval[1],
                              training=training, cost_bps=cost_bps, **candidate_config(candidate))


def make_agent(env, seed):
    return old.make_maskable_agent(env, seed)


def run_model_diagnostics(env, model, *, fixed_equal50=False):
    """Own-policy path, phase-zero only; never change inputs or decisions.

    Tanh saturation is |activation| > .95. Probabilities retain all four
    available phase-zero actions; ignored singleton days are excluded.
    """
    import torch
    observations, probabilities, mature = [], [], []
    observation, _ = env.reset()
    done = False
    while not done:
        if env.rebalance_phase == 0:
            tensor = model.policy.obs_to_tensor(observation)[0]
            with torch.no_grad():
                distribution = model.policy.get_distribution(tensor, action_masks=env.action_masks())
                probabilities.append(distribution.distribution.probs.cpu().numpy()[0])
            observations.append(observation.copy())
            mature.append(len(env.performance_history[0]) >= 20)
        action = 1 if fixed_equal50 else model_policy(model)(env, observation)
        observation, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    obs = np.asarray(observations)
    probs = np.asarray(probabilities)
    with torch.no_grad():
        features = model.policy.extract_features(model.policy.obs_to_tensor(obs)[0])
        if isinstance(features, tuple):
            features = features[0]
        first = model.policy.mlp_extractor.policy_net[0](features)
        saturation = (torch.abs(torch.tanh(first)) > .95).float().mean(dim=1).cpu().numpy()
    def summary(mask):
        p = probs[mask]
        if not len(p):
            return {'states': 0}
        return {'states': len(p), 'mean_entropy_nats': float(np.mean(-np.sum(p * np.log(np.maximum(p, 1e-30)), axis=1))),
                'probability_std_by_action': p.std(axis=0).tolist(),
                'max_probability_std': float(p.std(axis=0).max()),
                'mean_first_tanh_saturation': float(saturation[mask].mean()),
                'distinct_argmax_actions': int(len(np.unique(p.argmax(axis=1))))}
    ledger = pd.DataFrame(env.history)
    return ledger, validation_diagnostics_metrics(ledger), {
        'state_path': 'fixed_equal50' if fixed_equal50 else 'own_policy',
        'phase_zero_only': True, 'saturation_definition': 'abs(tanh(first_linear)) > 0.95',
        'all_decisions': summary(np.ones(len(probs), dtype=bool)),
        'mature_decisions': summary(np.asarray(mature, dtype=bool))}


def validation_diagnostics_metrics(ledger):
    from .metrics import performance_metrics
    metrics = performance_metrics(ledger.net_return.to_numpy()[1:], ledger.turnover.to_numpy()[1:])
    metrics['mean_cash_weight'] = float(np.mean([row[0] for row in ledger.weights.iloc[1:]]))
    metrics['return_days'] = len(ledger) - 1
    metrics.update(old.execution_metrics(ledger))
    return metrics


def checked_record(directory, candidate, seed, fold, panel, steps):
    directory = Path(directory)
    record = json.loads((directory / f'{candidate}.json').read_text())
    check_record_signature(record)
    if (record['candidate'], record['seed'], record['fold'], record['actual_steps']) != (candidate, seed, fold['name'], steps):
        raise ValueError('Training identity/budget mismatch')
    if record['requested_steps'] != steps or record['config'] != candidate_config(candidate) or record['ppo_config'] != PPO_CONFIG:
        raise ValueError('Training configuration changed')
    for key, filename in (('model_sha256', f'{candidate}.zip'), ('validation_sha256', f'validation_{candidate}.csv')):
        if record[key] != digest(directory / filename):
            raise ValueError('Model or validation checksum changed')
    start, end = fold['validation']
    _check_metrics(record['validation_metrics'], validation_metrics(directory / f'validation_{candidate}.csv', panel.dates[start:end+1]))
    if record['episode_schedule'] != old.episode_schedule(fold['train'], seed, steps):
        raise ValueError('Actual training episode schedule differs from paired plan')
    if record['runtime'] != runtime_info() or record['backend'] != 'sb3_contrib.MaskablePPO':
        raise ValueError('Training runtime changed')
    if record['kind'] != ('reused_v7_control' if candidate in RAW_CANDIDATES else 'new_scaled_fit'):
        raise ValueError('Control/new-fit provenance changed')
    if record['kind'] == 'reused_v7_control':
        source = record['source_record']
        if (source['model_sha256'] != record['model_sha256'] or source['candidate'] != candidate.replace('_raw', '_phase')
                or source['episode_schedule'] != record['episode_schedule']):
            raise ValueError('Reused source record mismatch')
        check_record_signature(source)
        _check_metrics(source['validation_metrics'], record['validation_metrics'])
    return record


def write_model_record(directory, candidate, seed, fold, model, env, data_dir, steps, kind, elapsed, source=None, source_validation=None):
    ledger, metrics, diagnostics = run_model_diagnostics(env, model)
    _, _, common_diagnostics = run_model_diagnostics(env, model, fixed_equal50=True)
    if source is not None:
        _check_metrics(source['validation_metrics'], metrics)
        original = pd.read_csv(source_validation)
        if not np.allclose(original.net_return, ledger.net_return, atol=1e-12, rtol=1e-12):
            raise ValueError('Raw control validation replay changed')
    path = Path(directory) / f'validation_{candidate}.csv'
    save_ledger(path, ledger)
    audit_saved_ledger(path, data_dir)
    record = {'candidate': candidate, 'seed': seed, 'fold': fold['name'], 'kind': kind,
              'actual_steps': int(model.num_timesteps), 'requested_steps': steps,
              'episode_schedule': source['episode_schedule'] if source else model._v8_episode_schedule,
              'training_seconds': elapsed, 'source_training_seconds': source['training_seconds'] if source else None,
              'config': candidate_config(candidate), 'ppo_config': PPO_CONFIG, 'runtime': runtime_info(),
              'backend': 'sb3_contrib.MaskablePPO',
              'model_sha256': digest(Path(directory) / f'{candidate}.zip'),
              'validation_sha256': digest(path), 'validation_metrics': metrics,
              'validation_diagnostics': diagnostics, 'common_state_diagnostics': common_diagnostics}
    if source is not None:
        record['source_record'] = source
    write_json(Path(directory) / f'{candidate}.json', signed_record(record))
    return ledger, metrics


def import_controls(data_dir, output, controls, steps):
    import torch
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(1)
    manifest = checked_manifest(data_dir, output, controls, steps)
    panel, actions = load_inputs(data_dir)
    for fold in manifest['folds']:
        for seed in SEEDS:
            directory = Path(output) / fold['name'] / f'seed_{seed}'
            directory.mkdir(parents=True, exist_ok=True)
            for candidate in RAW_CANDIDATES:
                old_name = candidate.replace('_raw', '_phase')
                old_dir = Path(controls) / fold['name'] / f'seed_{seed}'
                source = old.checked_record(old_dir, old_name, seed, fold, panel, steps)
                if (directory / f'{candidate}.json').exists():
                    record = checked_record(directory, candidate, seed, fold, panel, steps)
                    if record['source_record'] != source:
                        raise ValueError('Reused original record changed')
                    continue
                model = MaskablePPO.load(old_dir / f'{old_name}.zip', device='cpu')
                # New filename, byte-identical checkpoint, no retraining/resaving.
                shutil.copyfile(old_dir / f'{old_name}.zip', directory / f'{candidate}.zip')
                ledger, metrics = write_model_record(directory, candidate, seed, fold, model,
                    make_env(panel, actions, fold['validation'], candidate), data_dir, steps,
                    'reused_v7_control', 0., source, old_dir / f'validation_{old_name}.csv')
                _check_metrics(source['validation_metrics'], metrics)
                original = pd.read_csv(old_dir / f'validation_{old_name}.csv')
                if not np.allclose(original.net_return, ledger.net_return, atol=1e-12, rtol=1e-12):
                    raise ValueError('Raw control validation replay changed')
                checked_record(directory, candidate, seed, fold, panel, steps)
        print(f'Imported and replay-verified controls {fold["name"]}', flush=True)


def baseline_stage(data_dir, output, controls, steps):
    manifest = checked_manifest(data_dir, output, controls, steps)
    panel, actions = load_inputs(data_dir)
    result = {'manifest_sha256': digest(Path(output) / 'manifest.json'), 'folds': {}, 'ledger_hashes': {}}
    for fold in manifest['folds']:
        values = {}
        directory = Path(output) / 'baseline_preview' / fold['name']
        directory.mkdir(parents=True, exist_ok=True)
        for name in BASELINES:
            ledger, metrics = run_policy(make_env(panel, actions, fold['validation'], 'lognet_raw'), baseline_policy(name))
            path = directory / f'{name}.csv'
            save_ledger(path, ledger)
            audit_saved_ledger(path, data_dir)
            result['ledger_hashes'][str(path.relative_to(output))] = digest(path)
            values[name] = metrics
        result['folds'][fold['name']] = values
    write_json(Path(output) / 'baseline_preview/summary.json', signed_record(result))
    return result


def checked_baselines(output, manifest, panel):
    path = Path(output) / 'baseline_preview/summary.json'
    preview = json.loads(path.read_text())
    check_record_signature(preview)
    if preview['manifest_sha256'] != digest(Path(output) / 'manifest.json'):
        raise ValueError('Baseline manifest changed')
    for relative, expected in preview['ledger_hashes'].items():
        if digest(Path(output) / relative) != expected:
            raise ValueError('Baseline ledger changed')
    if set(preview['folds']) != {fold['name'] for fold in manifest['folds']}:
        raise ValueError('Baseline folds incomplete')
    for fold in manifest['folds']:
        if set(preview['folds'][fold['name']]) != set(BASELINES):
            raise ValueError('Baseline registry incomplete')
        a, b = fold['validation']
        for name in BASELINES:
            _check_metrics(preview['folds'][fold['name']][name], validation_metrics(
                Path(output) / 'baseline_preview' / fold['name'] / f'{name}.csv', panel.dates[a:b+1]))
    return preview


def fit_seed(data_dir, output, controls, steps, seed, fold_name=None):
    import torch
    torch.set_num_threads(1)
    if seed not in SEEDS:
        raise ValueError('Seed not in registered plan')
    manifest = checked_manifest(data_dir, output, controls, steps)
    panel, actions = load_inputs(data_dir)
    checked_baselines(output, manifest, panel)
    for fold in manifest['folds']:
        for old_seed in SEEDS:
            for name in RAW_CANDIDATES:
                checked_record(Path(output) / fold['name'] / f'seed_{old_seed}', name, old_seed, fold, panel, steps)
    for fold in old.selected_folds(manifest, fold_name):
        directory = Path(output) / fold['name'] / f'seed_{seed}'
        for candidate in NEW_CANDIDATES:
            if (directory / f'{candidate}.json').exists():
                checked_record(directory, candidate, seed, fold, panel, steps)
                print(f'Reuse {fold["name"]} {seed} {candidate}', flush=True)
                continue
            env = make_env(panel, actions, fold['train'], candidate, training=True)
            model = make_agent(env, seed)
            started = perf_counter()
            model.learn(total_timesteps=steps, progress_bar=False)
            elapsed = perf_counter() - started
            model._v8_episode_schedule = env.training_episode_schedule
            model.save(directory / f'{candidate}.zip')
            write_model_record(directory, candidate, seed, fold, model,
                make_env(panel, actions, fold['validation'], candidate), data_dir, steps,
                'new_scaled_fit', elapsed)
            checked_record(directory, candidate, seed, fold, panel, steps)
            print(f'{fold["name"]} seed={seed} {candidate}: {model.num_timesteps} daily steps {elapsed:.1f}s', flush=True)


def freeze_selection(data_dir, output, controls, steps):
    manifest = checked_manifest(data_dir, output, controls, steps)
    panel, _ = load_inputs(data_dir)
    preview = checked_baselines(output, manifest, panel)
    frozen = {}
    for fold in manifest['folds']:
        validation, hashes = {}, {}
        for candidate in CANDIDATES:
            validation[candidate] = []
            for seed in SEEDS:
                directory = Path(output) / fold['name'] / f'seed_{seed}'
                record = checked_record(directory, candidate, seed, fold, panel, steps)
                validation[candidate].append(record['validation_metrics'])
                for filename in (f'{candidate}.json', f'{candidate}.zip', f'validation_{candidate}.csv'):
                    path = directory / filename
                    hashes[str(path.relative_to(output))] = digest(path)
        frozen[fold['name']] = {**choose_candidates(validation, preview['folds'][fold['name']]), 'artifact_hashes': hashes}
    value = {'manifest_sha256': digest(Path(output) / 'manifest.json'),
             'baseline_summary_sha256': digest(Path(output) / 'baseline_preview/summary.json'), 'folds': frozen}
    path = Path(output) / 'frozen_selection.json'
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Frozen selection cannot change')
    else:
        write_json(path, value)
    return frozen


def evaluate(data_dir, output, controls, steps, cost_bps):
    if cost_bps not in COSTS:
        raise ValueError('Unregistered cost')
    if not (Path(output) / 'frozen_selection.json').exists():
        raise ValueError('Freeze all selections before any v8 test evaluation')
    import torch
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(1)
    frozen = freeze_selection(data_dir, output, controls, steps)
    manifest = checked_manifest(data_dir, output, controls, steps)
    panel, actions = load_inputs(data_dir)
    summary = {'cost_bps': cost_bps, 'folds': {}}
    for fold in manifest['folds']:
        folder = Path(output) / f'evaluation_{cost_bps:g}bps' / fold['name']
        folder.mkdir(parents=True, exist_ok=True)
        baselines, seeds = {}, {}
        for name in BASELINES:
            ledger, metrics = run_policy(make_env(panel, actions, fold['test'], 'lognet_raw', cost_bps=cost_bps), baseline_policy(name))
            save_ledger(folder / f'{name}.csv', ledger)
            audit_saved_ledger(folder / f'{name}.csv', data_dir, cost_bps)
            baselines[name] = metrics
        for seed in SEEDS:
            metrics = {}
            for candidate in CANDIDATES:
                model = MaskablePPO.load(Path(output) / fold['name'] / f'seed_{seed}' / f'{candidate}.zip', device='cpu')
                ledger, values = run_policy(make_env(panel, actions, fold['test'], candidate, cost_bps=cost_bps), model_policy(model))
                path = folder / f'seed_{seed}_{candidate}.csv'
                save_ledger(path, ledger)
                audit_saved_ledger(path, data_dir, cost_bps)
                metrics[candidate] = values
            seeds[str(seed)] = metrics
        summary['folds'][fold['name']] = {'baselines': baselines, 'seeds': seeds, 'selection': frozen[fold['name']]}
        print(f'Evaluated {fold["name"]} {cost_bps:g}bps', flush=True)
    write_json(Path(output) / f'evaluation_{cost_bps:g}bps/summary.json', summary)
    return summary


def status(data_dir, output, controls, steps):
    if not (Path(output) / 'manifest.json').exists():
        return {'state': 'unprepared', 'new_complete': 0, 'new_expected': 24, 'reused_complete': 0}
    manifest = checked_manifest(data_dir, output, controls, steps)
    panel, _ = load_inputs(data_dir)
    completed = []
    for fold in manifest['folds']:
        for seed in SEEDS:
            directory = Path(output) / fold['name'] / f'seed_{seed}'
            for candidate in CANDIDATES:
                if (directory / f'{candidate}.json').exists():
                    checked_record(directory, candidate, seed, fold, panel, steps)
                    completed.append(candidate)
    return {'state': 'frozen' if (Path(output) / 'frozen_selection.json').exists() else 'prepared_or_fitting',
            'new_complete': sum(name in NEW_CANDIDATES for name in completed), 'new_expected': 24,
            'reused_complete': sum(name in RAW_CANDIDATES for name in completed), 'reused_expected': 24}


def report(data_dir, output, controls, steps):
    from .statistics import paired_block_bootstrap, compare_seeded_returns
    output = Path(output)
    manifest = checked_manifest(data_dir, output, controls, steps)
    frozen = freeze_selection(data_dir, output, controls, steps)
    panel, _ = load_inputs(data_dir)
    rows, paired, comparisons, diagnostics = [], [], [], []
    for fold in manifest['folds']:
        for seed in SEEDS:
            for candidate in CANDIDATES:
                record = checked_record(output / fold['name'] / f'seed_{seed}', candidate, seed, fold, panel, steps)
                diagnostics.append({'fold': fold['name'], 'seed': seed, 'candidate': candidate, **record['validation_diagnostics'], 'common_state': record['common_state_diagnostics']})
    for cost in COSTS:
        folder = output / f'evaluation_{cost}bps'
        summary = json.loads((folder / 'summary.json').read_text())
        for fold in manifest['folds']:
            name = fold['name']
            a, b = fold['test']
            paths = {}
            for candidate in (*BASELINES, *CANDIDATES):
                paths[candidate] = []
                for seed in (SEEDS if candidate in CANDIDATES else (None,)):
                    filename = f'seed_{seed}_{candidate}.csv' if seed is not None else f'{candidate}.csv'
                    path = folder / name / filename
                    metrics = validation_metrics(path, panel.dates[a:b+1])
                    stored = summary['folds'][name]['seeds'][str(seed)][candidate] if seed is not None else summary['folds'][name]['baselines'][candidate]
                    _check_metrics(stored, metrics)
                    audit_saved_ledger(path, data_dir, cost)
                    rows.append(dict(cost_bps=cost, fold=name, strategy=candidate, seed=seed, **metrics))
                    paths[candidate].append(pd.read_csv(path).net_return.to_numpy()[1:])
            for objective in ('lognet', 'drawdown'):
                raw, scaled = paths[objective+'_raw'], paths[objective+'_scaled']
                paired.append({'cost_bps': cost, 'fold': name, 'objective': objective,
                    'direction': 'scaled minus raw',
                    'paired_seeds': [{'seed': seed, 'scaled_minus_raw_total_return': float(np.prod(1+scaled[i])-np.prod(1+raw[i]))} for i, seed in enumerate(SEEDS)],
                    'comparison': paired_block_bootstrap(np.mean(scaled, axis=0), np.mean(raw, axis=0))})
            for candidate in CANDIDATES:
                for anchor in ('equal_target_mix', 'trailing20', frozen[name]['best_validation_fixed_expert']):
                    comparisons.append({'cost_bps': cost, 'fold': name, 'candidate': candidate, 'anchor': anchor,
                        'comparison': compare_seeded_returns(paths[candidate], paths[anchor][0])['comparison']})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'all_metrics.csv', index=False)
    write_json(output / 'paired_scaling_contrasts.json', paired)
    write_json(output / 'primary_contrasts.json', comparisons)
    write_json(output / 'validation_diagnostics.json', diagnostics)
    selected = []
    for fold, selection in frozen.items():
        for label, key in (('selected_on_validation', 'candidate'), ('selected_with_rules', 'deploy_candidate')):
            for row in frame[(frame.fold == fold) & (frame.strategy == selection[key])].to_dict('records'):
                selected.append({**row, 'selection': label, 'chosen_strategy': selection[key]})
    pd.DataFrame(selected).to_csv(output / 'selected_metrics.csv', index=False)
    lines = ['# v8 count-only scaling controlled experiment', '',
             'Exploratory repeatedly inspected historical data, not untouched OOS or live performance.',
             '24 new fits × 20,000 daily transitions = 480,000, paired to 24 byte-identical reused v7 phase controls.',
             'Only the history-count feature is divided by 20. No new features, entropy, rewards, schedules or assets.',
             'All source/data/control identities and 48 candidate checkpoints/validation records freeze before v8 test replays.',
             'Rules count once; seeds represent optimizer randomness, not independent market histories.',
             'Independent raw-ledger audit is an engineering check, not evidence of alpha.', '',
             '| Cost | Fold | Strategy | Net total return mean | Max drawdown mean | Exposure mean | Executed switches mean |',
             '|---:|---|---|---:|---:|---:|---:|']
    for (cost, fold, strategy), part in frame.groupby(['cost_bps', 'fold', 'strategy'], sort=False):
        lines.append(f'| {cost} | {fold} | {strategy} | {part.total_return.mean():.2%} | {part.max_drawdown.mean():.2%} | {1-part.mean_cash_weight.mean():.2%} | {part.executed_expert_switches.mean():.2f} |')
    lines += ['', '## Frozen validation choices', '', '| Fold | Learned-only | With rules |', '|---|---|---|']
    for fold, selection in frozen.items():
        lines.append(f"| {fold} | {selection['candidate']} | {selection['deploy_candidate']} |")
    lines += ['', '## Paired scaling effects', '', '| Cost | Fold | Objective | Total-return difference | Annualized arithmetic excess 95% block interval |', '|---:|---|---|---:|---:|']
    for item in paired:
        delta = np.mean([x['scaled_minus_raw_total_return'] for x in item['paired_seeds']])
        low, high = item['comparison']['annualized_mean_excess_ci']
        lines.append(f"| {item['cost_bps']} | {item['fold']} | {item['objective']} | {delta:.2%} | [{low:.2%}, {high:.2%}] |")
    lines += ['', '## Mechanistic validation diagnostics', '',
              'Table uses common fixed-equal50 phase-zero validation states; mature means at least 20 completed daily returns. Own-policy diagnostics are additionally retained in JSON.',
              '| Fold | Candidate | Mature tanh saturation mean | Probability SD maximum mean | Entropy mean |', '|---|---|---:|---:|---:|']
    for fold in manifest['folds']:
        for candidate in CANDIDATES:
            part = [x['common_state']['mature_decisions'] for x in diagnostics if x['fold'] == fold['name'] and x['candidate'] == candidate]
            lines.append(f"| {fold['name']} | {candidate} | {np.mean([x['mean_first_tanh_saturation'] for x in part]):.2%} | {np.mean([x['max_probability_std'] for x in part]):.4f} | {np.mean([x['mean_entropy_nats'] for x in part]):.4f} |")
    lines += ['', 'More switching or lower saturation alone is not financial improvement. Every candidate, seed and fold is retained.',
              'Intervals use 10-day moving blocks and 2000 resamples for annualized arithmetic excess, not CAGR; no multiplicity correction.',
              'No test-selected winner, live deployment, new assets, paid resources or external publication.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'import-controls', 'baselines', 'fit', 'select', 'evaluate', 'status', 'report'], required=True)
    parser.add_argument('--data-dir', default='data/v3')
    parser.add_argument('--output', default='runs/research_v8')
    parser.add_argument('--controls', default='runs/research_v7')
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--seed', type=int, choices=SEEDS, default=7)
    parser.add_argument('--fold', choices=[f'fold_{n:02}' for n in range(1, 5)])
    parser.add_argument('--cost-bps', type=float, choices=COSTS, default=5)
    args = parser.parse_args()
    if args.fold is not None and args.stage != 'fit':
        parser.error('--fold is permitted only for fit shards')
    common = (args.data_dir, args.output, args.controls, args.steps)
    stages = {'prepare': prepare, 'import-controls': import_controls, 'baselines': baseline_stage,
              'select': freeze_selection, 'status': status, 'report': report}
    if args.stage == 'fit':
        fit_seed(*common, args.seed, args.fold)
    elif args.stage == 'evaluate':
        evaluate(*common, args.cost_bps)
    else:
        result = stages[args.stage](*common)
        if args.stage == 'status':
            print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
