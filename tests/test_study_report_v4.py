"""Reports must compare balanced outcomes and not inflate rule sample counts."""
import json

import numpy as np
import pandas as pd
import pytest

from rlasset.metrics import performance_metrics


def fixture_run(root):
    from rlasset.study_v4 import CANDIDATES, BASELINES
    dates = pd.bdate_range('2021-01-04', periods=31)
    directory = root / 'evaluation_5bps' / 'fold_01'
    directory.mkdir(parents=True)
    manifest = {'identity': {'seeds': [7, 17, 29], 'candidates': list(CANDIDATES), 'baselines': list(BASELINES)},
                'tickers': ['A', 'B', 'C'], 'folds': [{'name': 'fold_01', 'test': [120, 150],
                'dates': {'test': [str(dates[0].date()), str(dates[-1].date())]}}]}
    frozen = {'folds': {'fold_01': {'candidate': 'joint_ppo', 'deploy_candidate': 'vol_target10',
                                  'deploy_kind': 'rule', 'selected_on': 'validation'}}}
    def ledger(name, returns):
        frame = pd.DataFrame({'date': dates, 'net_return': np.r_[0., returns],
                              'portfolio_value': np.r_[1., np.cumprod(1+returns)],
                              'turnover': np.r_[0., np.full(30, .01)],
                              'weights': [json.dumps([.5, 1/6, 1/6, 1/6])] * 31})
        frame.to_csv(directory / f'{name}.csv', index=False)
        return {**performance_metrics(returns, np.full(30, .01)), 'mean_cash_weight': .5, 'return_days': 30}
    baselines = {name: ledger(name, np.zeros(30)) for name in BASELINES}
    records = {}
    for j, seed in enumerate((7, 17, 29)):
        records[str(seed)] = {name: ledger(f'seed_{seed}_{name}', np.r_[.1, -.1, np.zeros(28)]*(j+1))
                              for name in (*CANDIDATES, 'joint_mixture')}
        records[str(seed)]['selected_on_validation'] = records[str(seed)]['joint_ppo']
        records[str(seed)]['selected_with_rules'] = baselines['vol_target10']
    summary = {'cost_bps': 5, 'folds': {'fold_01': {'selected': 'joint_ppo',
               'deploy_selected': 'vol_target10', 'deploy_kind': 'rule', 'seeds': records, 'baselines': baselines}}}
    for filename, content in (('manifest.json', manifest), ('frozen_selection.json', frozen),
                              ('evaluation_5bps/summary.json', summary)):
        (root / filename).write_text(json.dumps(content), encoding='utf-8')
    return root


def report():
    from rlasset import study_report_v4
    return study_report_v4


def test_rule_fallback_is_one_path_and_seed_metrics_are_not_mean_return_compounding(tmp_path):
    data = report().analyze_run(fixture_run(tmp_path), n_bootstrap=20)
    fold = data['evaluations']['5']['folds']['fold_01']
    assert fold['strategies']['selected_with_rules']['n_paths'] == 1
    row = fold['strategies']['joint_ppo']
    assert row['metrics']['total_return']['mean'] == pytest.approx(-.0466666666666667)
    assert row['metrics']['max_drawdown']['mean'] == pytest.approx(.2)
    assert row['metrics']['balance_score']['mean'] == pytest.approx(-.1466666666666667)
    assert row['metrics']['annual_volatility']['mean'] > 0
    assert row['comparisons']['vol_target10']['comparison']['n_observations'] == 30
    assert len(fold['ablations']) == 4
    assert fold['ablations']['joint_ppo-minus-risk_ppo']['total_return_difference'] == 0


@pytest.mark.parametrize('damage', ['summary', 'date', 'nav', 'selection'])
def test_invalid_report_evidence_is_rejected(tmp_path, damage):
    run = fixture_run(tmp_path)
    path = run / 'evaluation_5bps' / 'fold_01' / 'seed_7_joint_ppo.csv'
    if damage in ('date', 'nav'):
        frame = pd.read_csv(path)
        if damage == 'date':
            frame = frame.drop(index=5)
        else:
            frame.loc[5, 'portfolio_value'] = 12
        frame.to_csv(path, index=False)
    else:
        path = run / 'evaluation_5bps' / 'summary.json'
        value = json.loads(path.read_text())
        if damage == 'summary':
            value['folds']['fold_01']['seeds']['7']['joint_ppo']['total_return'] = 123
        else:
            value['folds']['fold_01']['deploy_selected'] = 'joint_sac'
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        report().analyze_run(run, n_bootstrap=20)


def test_report_exports_risk_ablation_and_rule_comparisons(tmp_path):
    report().build_report(fixture_run(tmp_path), n_bootstrap=20)
    assert (tmp_path / 'equity.png').stat().st_size > 1000
    metrics = pd.read_csv(tmp_path / 'metrics.csv')
    assert len(metrics) == 15
    assert 'annual_volatility_mean' in metrics and 'balance_score_mean' in metrics
    assert len(pd.read_csv(tmp_path / 'ablation_metrics.csv')) == 4
    intervals = pd.read_csv(tmp_path / 'excess_intervals.csv')
    assert len(intervals) == 36
    rule = intervals.loc[intervals.strategy == 'selected_with_rules'].iloc[0]
    assert rule.n_seeds == 0 and rule.n_paths == 1
    text = (tmp_path / 'report.md').read_text(encoding='utf-8')
    assert 'bootstrap 20 次' in text
    assert '已完成成本评估：5 bps' in text
    assert '5/10/20 bps 单边成本完整指标' not in text
