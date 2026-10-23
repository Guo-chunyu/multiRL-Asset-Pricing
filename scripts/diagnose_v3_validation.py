"""Describe completed validation allocations without opening any test result.

Usage::

    python scripts/diagnose_v3_validation.py --run runs/research_v3 \
        --output-dir runs/research_v3/diagnostics

The model JSON is the completion marker written after validation. Missing
markers/ledgers are reported as pending; inconsistent completed artifacts are
reported as errors. Model checkpoints are hashed, never loaded or modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weight_matrix(frame, column, width):
    matrix = np.asarray([json.loads(value) for value in frame[column]], dtype=float)
    if (matrix.shape != (len(frame), width) or not np.isfinite(matrix).all()
            or (matrix < -1e-10).any()
            or not np.allclose(matrix.sum(axis=1), 1, atol=1e-7, rtol=0)):
        raise ValueError(f'Invalid {column} simplex')
    return matrix


def inspect_record(directory, fold, candidate, seed, manifest):
    record_path = directory / f'{candidate}.json'
    ledger_path = directory / f'validation_{candidate}.csv'
    model_path = directory / f'{candidate}.zip'
    record = json.loads(record_path.read_text(encoding='utf-8'))
    if record['candidate'] != candidate or record['seed'] != seed:
        raise ValueError('Record candidate/seed mismatch')
    if record['requested_steps'] != manifest['identity']['steps']:
        raise ValueError('Record requested training budget differs from manifest')
    if sha256(model_path) != record['model_sha256']:
        raise ValueError('Checkpoint hash differs from completed record')
    frame = pd.read_csv(ledger_path)
    dates = pd.to_datetime(frame.date)
    expected_start, expected_end = map(pd.Timestamp, fold['dates']['validation'])
    expected_days = fold['validation'][1] - fold['validation'][0]
    if (not dates.is_unique or not dates.is_monotonic_increasing
            or dates.iloc[0] != expected_start or dates.iloc[-1] != expected_end
            or len(frame) != expected_days + 1):
        raise ValueError('Ledger does not match the registered validation interval')
    numeric = frame[['net_return', 'portfolio_value', 'turnover', 'cost_fraction']].to_numpy(float)
    if not np.isfinite(numeric).all() or (numeric[:, 1] <= 0).any():
        raise ValueError('Ledger contains nonfinite values or nonpositive NAV')
    if not np.isclose(frame.net_return.iloc[0], 0) or not np.isclose(frame.portfolio_value.iloc[0], 1):
        raise ValueError('Initial valuation must have zero return and unit NAV')
    if not np.allclose(np.cumprod(1 + frame.net_return.to_numpy()),
                       frame.portfolio_value.to_numpy(), atol=1e-9, rtol=1e-9):
        raise ValueError('Daily returns do not reconstruct NAV')
    if (frame[['turnover', 'cost_fraction']].to_numpy() < -1e-12).any():
        raise ValueError('Negative turnover or fee fraction')
    days = frame.iloc[1:].copy()
    weights = weight_matrix(days, 'weights', len(manifest['tickers']) + 1)
    targets = weight_matrix(days, 'target_weights', len(manifest['tickers']) + 1)
    if 'rebalance_executed' in days:
        flags = days.rebalance_executed.astype(str).str.lower()
        if not flags.isin(['true', 'false']).all():
            raise ValueError('Invalid scheduled rebalance flags')
        rebalance = flags.eq('true').to_numpy()
        hold = ~rebalance
        if not np.allclose(days.loc[hold, ['turnover', 'cost_fraction']], 0, atol=1e-12, rtol=0):
            raise ValueError('Hold day unexpectedly incurs turnover or fees')
    else:
        # The legacy policy submits a target at every valuation transition.
        rebalance = np.ones(len(days), dtype=bool)
    trade_targets = targets[rebalance]
    if not len(trade_targets):
        raise ValueError('Completed validation ledger has no allocation decision')
    if (trade_targets[:, 1:] > .5 + 1e-7).any():
        raise ValueError('Requested risky allocation exceeds the registered .5 cap')
    target_std = trade_targets.std(axis=0)
    target_l1_changes = np.abs(np.diff(trade_targets, axis=0)).sum(axis=1)
    cash = weights[:, 0]
    turnover = days.turnover.to_numpy(float)
    total_return = float(frame.portfolio_value.iloc[-1] - 1)
    if not np.isclose(total_return, record['validation_metrics']['total_return'], atol=1e-10, rtol=0):
        raise ValueError('Record return disagrees with validation ledger')
    max_std = float(target_std.max())
    flags = []
    if max_std < .005:
        flags.append('near_constant_executed_targets')
    if cash.mean() >= .95:
        flags.append('cash_heavy')
    if turnover.mean() > .20:
        flags.append('high_gross_turnover')
    mean_turnover = float(turnover.mean())
    return {
        'fold': fold['name'], 'candidate': candidate, 'seed': seed,
        'first_return_date': str(dates.iloc[1].date()), 'last_return_date': str(dates.iloc[-1].date()),
        'return_days': len(days), 'executed_allocation_days': int(rebalance.sum()),
        'actual_training_steps': record['actual_steps'], 'training_seconds': record['training_seconds'],
        'validation_total_return': total_return,
        'validation_max_drawdown': record['validation_metrics']['max_drawdown'],
        'mean_cash_weight_including_receivables': float(cash.mean()),
        'mean_equity_weight': float((1 - cash).mean()),
        'min_equity_weight': float((1 - cash).min()),
        'max_equity_weight': float((1 - cash).max()),
        'cash_weight_time_std': float(cash.std()),
        'fraction_days_cash_at_least_95pct': float((cash >= .95).mean()),
        'mean_liquid_cash_to_nav': float((days.cash / days.portfolio_value).mean()),
        'mean_receivable_to_nav': float((days.receivable / days.portfolio_value).mean()),
        'mean_actual_weights_by_asset': dict(zip(['cash', *manifest['tickers']], weights.mean(axis=0).tolist())),
        'mean_executed_target_weights_by_asset': dict(zip(['cash', *manifest['tickers']], trade_targets.mean(axis=0).tolist())),
        'actual_weight_time_std_mean': float(weights.std(axis=0).mean()),
        'actual_weight_time_std_by_asset': dict(zip(['cash', *manifest['tickers']], weights.std(axis=0).tolist())),
        'executed_target_time_std_mean': float(target_std.mean()),
        'executed_target_time_std_max': max_std,
        'executed_target_time_std_by_asset': dict(zip(['cash', *manifest['tickers']], target_std.tolist())),
        'mean_consecutive_executed_target_l1': float(target_l1_changes.mean()) if len(target_l1_changes) else None,
        'mean_daily_gross_turnover': mean_turnover,
        'annualized_gross_turnover_proxy': 252 * mean_turnover,
        'sum_daily_cost_fractions': float(days.cost_fraction.sum()),
        'mean_daily_cost_fraction': float(days.cost_fraction.mean()),
        'annual_cost_proxy_5bps': 252 * mean_turnover * .0005,
        'annual_cost_proxy_10bps': 252 * mean_turnover * .001,
        'annual_cost_proxy_20bps': 252 * mean_turnover * .002,
        'descriptive_flags': flags,
        'model_sha256': record['model_sha256'],
        'validation_ledger_sha256': sha256(ledger_path),
    }


def diagnose(run):
    run = Path(run)
    manifest_path = run / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows, pending, errors = [], [], []
    for fold in manifest['folds']:
        for seed in manifest['identity']['seeds']:
            directory = run / fold['name'] / f'seed_{seed}'
            for candidate in manifest['identity']['candidates']:
                identity = {'fold': fold['name'], 'seed': seed, 'candidate': candidate}
                required = [directory / f'{candidate}.json', directory / f'validation_{candidate}.csv',
                            directory / f'{candidate}.zip']
                missing = [path.name for path in required if not path.is_file()]
                if missing:
                    pending.append({**identity, 'missing_artifacts': missing})
                    continue
                try:
                    rows.append(inspect_record(directory, fold, candidate, seed, manifest))
                except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                    errors.append({**identity, 'error': str(error),
                                   'note': 'An actively written artifact may be incomplete; rerun after training.'})
    expected = len(manifest['folds']) * len(manifest['identity']['seeds']) * len(manifest['identity']['candidates'])
    return {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'complete' if len(rows) == expected and not errors else 'partial',
        'scope': 'completed validation records and validation ledgers only; test results are never opened',
        'manifest_sha256': sha256(manifest_path), 'expected_models': expected,
        'completed_models': len(rows), 'pending_models': len(pending), 'invalid_models': len(errors),
        'notes': [
            'No rankings, new strategy selection, model changes, or retraining are performed.',
            'Allocation variability uses submitted targets on scheduled rebalance days only; passive drift is separate.',
            'Near-constant means maximum component target standard deviation <0.005 (0.5 percentage points).',
            'Cash-heavy means mean cash weight >=0.95; the cash slot includes unpaid dividend receivables.',
            'High gross turnover means mean daily two-sided risky turnover >0.20; this is a descriptive threshold.',
            'Annual fee proxies are 252 * observed mean turnover * fee rate, holding the observed path fixed.',
            'Fee proxies are first-order diagnostics, not retraded cost-stress backtests or exact cumulative losses.',
            'Low drawdown or high return may reflect exposure differences; these diagnostics do not establish alpha.',
            'Partial snapshots must not be treated as comparisons over all registered seeds or folds.',
        ],
        'records': rows, 'pending': pending, 'errors': errors,
    }


def markdown(report):
    lines = [
        '# Validation allocation diagnostics', '',
        f'Status: **{report["status"]}**. Completed {report["completed_models"]}/{report["expected_models"]}; '
        f'pending {report["pending_models"]}; invalid {report["invalid_models"]}.', '',
        'Validation only. Columns are descriptive; rows follow the registered order rather than performance rank.', '',
        '| Fold | Seed | Candidate | Return | Mean equity | Max target std | Daily gross turnover | 5 bps annual fee proxy | Flags |',
        '|---|---:|---|---:|---:|---:|---:|---:|---|',
    ]
    for row in report['records']:
        lines.append(f'| {row["fold"]} | {row["seed"]} | {row["candidate"]} | '
                     f'{row["validation_total_return"]:.2%} | {row["mean_equity_weight"]:.2%} | '
                     f'{row["executed_target_time_std_max"]:.2%} | {row["mean_daily_gross_turnover"]:.2%} | '
                     f'{row["annual_cost_proxy_5bps"]:.2%} | {", ".join(row["descriptive_flags"]) or "-"} |')
    lines += ['', *[f'- {note}' for note in report['notes']]]
    if report['pending']:
        lines += ['', 'Pending artifacts:', '', *[
            f'- {row["fold"]} / seed {row["seed"]} / {row["candidate"]}: {", ".join(row["missing_artifacts"])}'
            for row in report['pending']]]
    if report['errors']:
        lines += ['', 'Invalid or concurrently incomplete artifacts:', '', *[
            f'- {row["fold"]} / seed {row["seed"]} / {row["candidate"]}: {row["error"]}'
            for row in report['errors']]]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/research_v3')
    parser.add_argument('--output-dir', help='Optional directory for validation_diagnostics.json/.csv/.md')
    args = parser.parse_args()
    report = diagnose(args.run)
    text = markdown(report)
    if args.output_dir:
        destination = Path(args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'validation_diagnostics.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        (destination / 'validation_diagnostics.md').write_text(text, encoding='utf-8')
        table = pd.DataFrame(report['records'])
        for column in ('descriptive_flags', 'executed_target_time_std_by_asset',
                       'mean_actual_weights_by_asset', 'mean_executed_target_weights_by_asset',
                       'actual_weight_time_std_by_asset'):
            if column in table:
                table[column] = table[column].map(json.dumps)
        table.to_csv(destination / 'validation_diagnostics.csv', index=False)
        print(f'{report["status"]}: completed={report["completed_models"]}/{report["expected_models"]}, '
              f'pending={report["pending_models"]}, invalid={report["invalid_models"]}; saved to {destination}')
    else:
        print(text)
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
