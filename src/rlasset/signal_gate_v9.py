"""Validation-only gate for one prespecified relative-allocation rule; no RL."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .robust_env import RobustPortfolioEnv
from .metrics import performance_metrics
from .study_v3 import digest, load_inputs, save_ledger, write_json
from .study_v7 import require_certified_data, audit_saved_ledger

ROOT = Path(__file__).resolve().parents[2]
TICKERS = ('510050', '510300', '510500')
YEARS = (2018, 2020, 2022, 2024)
COSTS = (5, 10, 20)
RULES = ('equal50', 'relative_momentum50')


def relative_momentum50(signal_close, t, tickers):
    """20-session total-return rank; ticker ascending breaks exact ties."""
    close = np.asarray(signal_close, dtype=float)
    tickers = tuple(tickers)
    if (close.ndim != 2 or close.shape[1] != 3 or len(tickers) != 3
            or len(set(tickers)) != 3 or any(not isinstance(x, str) for x in tickers)
            or isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer))
            or not 20 <= t < len(close)):
        raise ValueError('Require three distinct tickers and integer decision index >=20')
    prefix = close[t-20:t+1]
    if not np.isfinite(prefix).all() or np.any(prefix <= 0):
        raise ValueError('Decision prefix prices must be finite and positive')
    score = prefix[-1] / prefix[0] - 1
    order = np.lexsort((np.asarray(tickers), -score))
    weights = np.empty(3)
    weights[order] = np.array([1/4, 1/6, 1/12])
    return np.r_[.5, weights]


def validation_intervals(dates):
    dates = pd.DatetimeIndex(dates)
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError('Calendar must be unique and increasing')
    result = []
    for year in YEARS:
        indices = np.flatnonzero(dates.year == year)
        if len(indices) < 2 or indices[0] < 120:
            raise ValueError('Incomplete registered validation calendar')
        result.append({'year': year, 'interval': [int(indices[0]), int(indices[-1])],
                       'dates': [str(dates[indices[0]].date()), str(dates[indices[-1]].date())]})
    return result


def identity(data_dir):
    require_certified_data(data_dir)
    sources = [Path(__file__), ROOT/'tests/test_signal_gate_v9.py', ROOT/'docs/research-protocol-v9-signal.md',
               ROOT/'scripts/audit_research_v3.py']
    sources += [Path(__file__).with_name(n) for n in ('study_v3.py','study_v7.py','robust_env.py','corporate_actions.py','env.py','data.py','metrics.py')]
    return {'sources': {str(p.relative_to(ROOT)): digest(p) for p in sources},
            'data': {n:digest(Path(data_dir)/n) for n in ('ashare_broad_raw.csv','cash_actions.csv','splits.csv','restoration_verification.json','restoration_audit.json')},
            'tickers': list(TICKERS), 'years': list(YEARS), 'costs': list(COSTS), 'rules': list(RULES),
            'gate': {'strict_positive_score_years_per_cost':3, 'score':'total_return - 0.5 * max_drawdown',
                     'maximum_drawdown_deterioration_each_year_each_cost':.02},
            'new_rl_fits':0}


def prepare(data_dir, output):
    panel, _ = load_inputs(data_dir)
    if tuple(panel.tickers) != TICKERS:
        raise ValueError('Only the original three ETFs are registered')
    value = {'identity': identity(data_dir), 'validation_periods': validation_intervals(panel.dates)}
    path = Path(output)/'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Frozen validation-only protocol/input changed')
    else:
        write_json(path, value)
    return value


def check_approval(output):
    path = Path(output)/'preflight_review.json'
    if not path.exists():
        raise ValueError('Preflight approval required before financial evaluation')
    value = json.loads(path.read_text())
    if value.get('status') != 'passed' or value.get('manifest_sha256') != digest(Path(output)/'manifest.json'):
        raise ValueError('Preflight approval does not match current manifest')


def run_rule(panel, actions, interval, rule, cost):
    if rule not in RULES or cost not in COSTS:
        raise ValueError('Unregistered rule or cost')
    env = RobustPortfolioEnv(panel, *interval, actions=actions, rebalance_every=5, cost_bps=cost, risk_aversion=0)
    env.reset()
    done, targets = False, []
    while not done:
        if env.rebalance_phase == 0:
            target = np.r_[.5, np.full(3, 1/6)] if rule == 'equal50' else relative_momentum50(env.signal_close, env.t, panel.tickers)
            targets.append(target.copy())
        else:
            target = None
        _, _, term, trunc, _ = env.step_weights(target)
        done = term or trunc
        env.history[-1]['requested_rule_target'] = target.tolist() if target is not None else None
    frame = pd.DataFrame(env.history)
    metrics = performance_metrics(frame.net_return.to_numpy()[1:], frame.turnover.to_numpy()[1:])
    metrics['return_days'] = len(frame)-1
    metrics['mean_equity_weight'] = float(np.mean([1-row[0] for row in frame.weights.iloc[1:]]))
    metrics['executed_rebalances'] = len(targets)
    metrics['sum_daily_cost_fraction'] = float(frame.cost_fraction.fillna(0).sum())
    targets = np.asarray(targets)
    reference = np.r_[.5, np.full(3, 1/6)]
    metrics['mean_requested_target_l1_from_equal50'] = float(np.abs(targets-reference).sum(axis=1).mean())
    metrics['fraction_requested_targets_different_from_equal50'] = float(np.mean(np.any(np.abs(targets-reference)>1e-12, axis=1)))
    metrics['distinct_requested_targets'] = len(np.unique(targets, axis=0))
    return frame, metrics


def gate_decision(rows):
    expected = {(cost, year, rule) for cost in COSTS for year in YEARS for rule in RULES}
    keys = [(r['cost_bps'],r['year'],r['rule']) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Exactly all 24 cost/year/rule paths are mandatory')
    lookup = {k:r for k,r in zip(keys, rows)}
    comparisons, per_cost = [], []
    for cost in COSTS:
        wins, dd_ok = 0, True
        for year in YEARS:
            base, new = (lookup[(cost,year,rule)] for rule in RULES)
            vals = np.array([base['total_return'],new['total_return'],base['max_drawdown'],new['max_drawdown']],float)
            if not np.isfinite(vals).all() or np.any(vals[2:] < 0):
                raise ValueError('Invalid gate metrics')
            score_delta = (vals[1]-.5*vals[3])-(vals[0]-.5*vals[2])
            dd_delta = vals[3]-vals[2]
            wins += int(score_delta > 0)
            dd_ok = dd_ok and bool(dd_delta <= .02 + 1e-12)
            comparisons.append({'cost_bps':cost,'year':year,'score_delta':float(score_delta),
                'total_return_delta':float(vals[1]-vals[0]),'max_drawdown_delta':float(dd_delta)})
        per_cost.append({'cost_bps':cost,'strict_score_wins':wins,'drawdown_gate_passed':dd_ok,'passed':bool(wins>=3 and dd_ok)})
    passed = all(r['passed'] for r in per_cost)
    return {'passed':passed,'decision':'eligible_for_separately_reviewed_RL_experiment' if passed else 'do_not_train_new_selector',
            'interpretation':'Exploratory compute-budget gate only, not statistical evidence of alpha or permission to deploy',
            'cost_gates':per_cost,'comparisons':comparisons}


def evaluate(data_dir, output):
    output = Path(output)
    if not (output/'manifest.json').exists():
        raise ValueError('Prepare frozen manifest first')
    manifest = prepare(data_dir, output)
    check_approval(output)
    panel, actions = load_inputs(data_dir)
    rows, hashes = [], {}
    for cost in COSTS:
        for period in manifest['validation_periods']:
            for rule in RULES:
                path = output/f'validation_{cost}bps'/str(period['year'])/f'{rule}.csv'
                path.parent.mkdir(parents=True, exist_ok=True)
                ledger, metrics = run_rule(panel, actions, period['interval'], rule, cost)
                save_ledger(path, ledger)
                audit_saved_ledger(path, data_dir, cost)
                hashes[str(path.relative_to(output))] = digest(path)
                rows.append({'cost_bps':cost,'year':period['year'],'rule':rule,**metrics})
            print(f"Validation {period['year']} cost {cost} bps complete",flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'all_metrics.csv',index=False)
    decision = gate_decision(rows)
    write_json(output/'gate_decision.json', decision)
    write_json(output/'run_record.json', {'manifest_sha256':digest(output/'manifest.json'),
        'completed_at_utc':datetime.now(timezone.utc).isoformat(),'ledger_sha256':hashes,
        'claim':'Validation-only exploratory gate; no RL fit or new test-period evaluation'})
    lines = ['# v9 relative-allocation signal gate','', 'Validation-only historical development. No RL was trained and no test split was opened.',
        'One 20-session rule, fixed 50% equity, no parameter search. Same total equity does not imply same volatility or beta.',
        'Beta is deliberately not estimated; this study makes no risk-adjusted alpha claim. Each year starts independently; do not concatenate NAVs.',
        '',f"Budget gate: **{decision['decision']}**",'',
        '| Cost bps | Year | Rule | Net total return | Annual return | MDD | Annual volatility | Mean turnover | Target L1 vs equal50 |',
        '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['cost_bps']} | {r['year']} | {r['rule']} | {r['total_return']:.3%} | {r['annual_return']:.3%} | {r['max_drawdown']:.3%} | {r['annual_volatility']:.3%} | {r['mean_turnover']:.5f} | {r['mean_requested_target_l1_from_equal50']:.5f} |")
    lines += ['', 'Gate requires at least 3/4 strictly positive total-return-minus-half-MDD differences at EACH cost, and no annual MDD deterioration above 2 percentage points at ANY cost.',
              'This arbitrary prespecified research-budget hurdle is not a statistical test. A failure forbids new selector training under this plan; do not alter the rule after seeing results.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('prepare','evaluate'), required=True)
    parser.add_argument('--data-dir', default='data/v3')
    parser.add_argument('--output', default='runs/research_v9_signal')
    args = parser.parse_args()
    result = (prepare if args.stage=='prepare' else evaluate)(args.data_dir,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
