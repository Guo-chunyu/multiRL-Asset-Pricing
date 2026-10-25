"""Verified v4 factorial research reports; never train or choose on test returns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .statistics import compare_seeded_returns
from .study_report_v3 import (_aggregate, _check_summary, _digest, _export_csv,
                              _json, _ledger, _metric_cell)

BENCHMARKS = ('equal_50', 'equal_90', 'vol_target10', 'invvol_target10')
ABLATIONS = (('signal_ppo', 'base_ppo'), ('risk_ppo', 'base_ppo'),
             ('joint_ppo', 'risk_ppo'), ('joint_ppo', 'signal_ppo'))
DISPLAY_METRICS = ('total_return', 'annual_volatility', 'sharpe', 'max_drawdown',
                   'balance_score', 'mean_turnover', 'mean_cash_weight')


def aggregate(paths):
    row = _aggregate(paths)
    for key in ('annual_volatility', 'balance_score'):
        values = np.asarray([p['metrics']['annual_volatility'] if key == 'annual_volatility'
                             else p['metrics']['total_return'] - .5 * p['metrics']['max_drawdown']
                             for p in paths])
        row['metrics'][key] = {'mean': float(values.mean()), 'std': float(values.std(ddof=1)) if len(values) > 1 else None,
                               'n_defined': len(values)}
    return row


def analyze_run(run, *, n_bootstrap=2000):
    run = Path(run)
    manifest, frozen = _json(run / 'manifest.json'), _json(run / 'frozen_selection.json')
    candidates, seeds = manifest['identity']['candidates'], manifest['identity']['seeds']
    baselines = manifest['identity']['baselines']
    fold_names = [f['name'] for f in manifest['folds']]
    if (not candidates or not seeds or len(set(seeds)) != len(seeds)
            or len(set(candidates)) != len(candidates) or set(frozen['folds']) != set(fold_names)):
        raise ValueError('Incomplete unique candidate/seed/fold registry')
    strategies = candidates + ['joint_mixture', 'selected_on_validation', 'selected_with_rules']
    result = {'manifest': manifest, 'frozen_selection': frozen, 'evaluations': {},
              'source_hashes': {p: _digest(run / p) for p in ('manifest.json', 'frozen_selection.json')},
              'method': {'bootstrap_block_days': 10, 'bootstrap_resamples': n_bootstrap,
                         'seed_unit': 'Training randomness, not independent markets',
                         'bootstrap_unit': 'Paired dates after averaging seed returns',
                         'equity_curve': 'Mean initial-1 seed NAV; not compounded mean daily return',
                         'selection': 'Frozen validation choices only',
                         'claim': 'Retrospective development; historical periods were already seen'}}
    for cost in (5, 10, 20):
        directory = run / f'evaluation_{cost}bps'
        if not (directory / 'summary.json').exists():
            if cost == 5:
                raise ValueError('Primary evaluation is required')
            continue
        summary = _json(directory / 'summary.json')
        if summary['cost_bps'] != cost or set(summary['folds']) != set(fold_names):
            raise ValueError('Evaluation cost or fold coverage mismatch')
        evaluation = {'cost_bps': cost, 'folds': {}, 'summary_sha256': _digest(directory / 'summary.json')}
        previous_end = None
        for fold in manifest['folds']:
            name, selection = fold['name'], frozen['folds'][fold['name']]
            selected, deployed = selection['candidate'], selection['deploy_candidate']
            report = summary['folds'][name]
            start, end = map(pd.Timestamp, fold['dates']['test'])
            if (start >= end or (previous_end is not None and start <= previous_end)
                    or selection['selected_on'] != 'validation' or selected not in candidates
                    or deployed not in candidates + ['vol_target10', 'invvol_target10']
                    or report['selected'] != selected or report['deploy_selected'] != deployed
                    or report['deploy_kind'] != selection['deploy_kind']
                    or selection['deploy_kind'] != ('rule' if deployed in baselines else 'rl')
                    or set(report['seeds']) != {str(s) for s in seeds}
                    or set(report['baselines']) != set(baselines)):
                raise ValueError('Frozen choice, dates or coverage mismatch')
            previous_end = end
            dates, baseline_paths, paths, hashes = None, {}, {}, {}
            for baseline in baselines:
                file = directory / name / f'{baseline}.csv'
                ledger = _ledger(file, fold, len(manifest['tickers']), dates)
                dates = ledger['dates']
                _check_summary(ledger['metrics'], report['baselines'][baseline], str(file))
                baseline_paths[baseline], hashes[file.name] = ledger, _digest(file)
            if cost != 5 and dates.strftime('%Y-%m-%d').tolist() != result['evaluations']['5']['folds'][name]['dates']:
                raise ValueError('Cost stress calendars differ')
            for strategy in strategies:
                source = selected if strategy == 'selected_on_validation' else deployed if strategy == 'selected_with_rules' else strategy
                paths[strategy] = []
                for seed in seeds:
                    if set(report['seeds'][str(seed)]) != set(strategies):
                        raise ValueError('Incomplete seed strategies')
                    if source in baseline_paths:
                        ledger = baseline_paths[source]
                    else:
                        file = directory / name / f'seed_{seed}_{source}.csv'
                        ledger = _ledger(file, fold, len(manifest['tickers']), dates)
                        hashes[file.name] = _digest(file)
                    _check_summary(ledger['metrics'], report['seeds'][str(seed)][strategy], f'{name}/{seed}/{strategy}')
                    if source not in baseline_paths or not paths[strategy]:
                        paths[strategy].append(ledger)
            rows = {}
            for strategy, ledgers in paths.items():
                source = selected if strategy == 'selected_on_validation' else deployed if strategy == 'selected_with_rules' else strategy
                row = aggregate(ledgers)
                row.update(source_candidate=source, seeds=seeds if source not in baselines else [],
                           kind='rule' if source in baselines else 'rl')
                row['comparisons'] = {baseline: compare_seeded_returns(
                    np.asarray([p['returns'] for p in ledgers]), baseline_paths[baseline]['returns'],
                    block_length=10, n_bootstrap=n_bootstrap, seed=7) for baseline in BENCHMARKS}
                rows[strategy] = row
            ablations = {}
            for changed, control in ABLATIONS:
                paired = compare_seeded_returns(np.asarray([p['returns'] for p in paths[changed]]),
                                               np.mean([p['returns'] for p in paths[control]], axis=0),
                                               block_length=10, n_bootstrap=n_bootstrap, seed=7)
                ablations[f'{changed}-minus-{control}'] = {
                    'changed': changed, 'control': control,
                    **{f'{key}_difference': rows[changed]['metrics'][key]['mean'] - rows[control]['metrics'][key]['mean']
                       if rows[changed]['metrics'][key]['mean'] is not None and rows[control]['metrics'][key]['mean'] is not None else None
                       for key in DISPLAY_METRICS}, 'paired_daily_excess': paired['comparison']}
            evaluation['folds'][name] = {'selected_on_validation': selected, 'selected_with_rules': deployed,
                'dates': dates.strftime('%Y-%m-%d').tolist(), 'return_days': len(dates)-1,
                'strategies': rows, 'baselines': {b: aggregate([p]) for b, p in baseline_paths.items()},
                'ablations': ablations, 'ledger_sha256': hashes}
        result['evaluations'][str(cost)] = evaluation
    return result


def markdown(data):
    lines = ['# v4 信息与风险配置研究', '',
        '真实 A 股 ETF 历史回溯研究。历史测试区间已在 v3 看过，因此不是全新样本外证明。', '',
        '六种 RL 候选、三个种子、四折独立测试；原始价格、分红应收/到账与拆分记账。',
        '每折从 NAV=1 开始；不拼接历史。下表是各路径指标的均值±种子标准差；规则仅一条路径。',
        '平衡分数=累计收益−0.5×最大回撤。它用于固定验证选择，不保证任何偏好的投资者都获益。',
        '波动率和 Sharpe 按252天年化，无风险利率0；现金包含应收分红。目标波动不是实际风险保证。', '',
        '[指标CSV](metrics.csv) · [成对超额区间](excess_intervals.csv) · [消融差异](ablation_metrics.csv) · [信号诊断](signal_metrics.csv)', '']
    for name, fold in data['evaluations']['5']['folds'].items():
        lines += [f"## {name}: {fold['dates'][1]}—{fold['dates'][-1]}", '',
                  f"验证期 RL 选择：`{fold['selected_on_validation']}`；允许规则回退：`{fold['selected_with_rules']}`。", '',
                  '| 策略 | 累计收益 | 年化波动 | Sharpe | 最大回撤 | 平衡分数 | 日均换手 | 平均现金及应收 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for strategy, row in {**fold['strategies'], **fold['baselines']}.items():
            lines.append('| '+strategy+' | '+' | '.join(_metric_cell(row, key) for key in DISPLAY_METRICS)+' |')
        lines += ['', '消融差异（前者减后者；回撤差值为负表示回撤降低）：', '',
                  '| 消融 | 收益差 | 回撤差 | 平衡分数差 |', '|---|---:|---:|---:|']
        for label, row in fold['ablations'].items():
            lines.append(f"| {label} | {row['total_return_difference']:.2%} | {row['max_drawdown_difference']:.2%} | {row['balance_score_difference']:.2%} |")
        lines.append('')
    available_costs = ', '.join(f'{c} bps' for c in data['evaluations'])
    lines += ['## 成本与统计', '',
              f'已完成成本评估：{available_costs}。完整指标见 CSV。更高成本使用同一模型重新执行，持仓反馈可能变化。',
              f"成对区间按对齐市场日，种子先平均，每折独立10日移动分块 bootstrap {data['method']['bootstrap_resamples']} 次。估计252×日均算术超额，并非CAGR区间。没有多重比较校正，也没有消除历史反复开发的偏差。", '',
              '信息读出诊断固定 Ridge alpha=10，仅在训练期拟合缩放/系数，随后验证/测试不调整。未来5日标签完全在各自分区内。它不参与策略选择，也不代表RL实际使用了哪些信号。', '',
              '风险层与信息特征需结合消融、低仓位等权和无RL动态风险规则评估；收益减少与回撤减少必须同时展示。三只相关性高的长期存续宽基ETF不能代表全A股。研究使用分数份额，不验证券商成交或实时运行。', '',
              '![分折净值](equity.png)', '', '![风险收益和消融](comparison.png)', '']
    return '\n'.join(lines)


def plots(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folds = data['evaluations']['5']['folds']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    scatter, squares = plt.subplots(2, 2, figsize=(15, 10))
    styles = {'base_ppo': '#999999', 'joint_ppo': '#007c91', 'joint_sac': '#d59028',
              'joint_td3': '#9b59b6', 'selected_with_rules': '#141414',
              'vol_target10': '#c94c4c', 'equal_90': '#6e8a46'}
    for ax, square, (name, fold) in zip(axes.flat, squares.flat, folds.items()):
        all_rows = {**fold['strategies'], **fold['baselines']}
        dates = pd.to_datetime(fold['dates'])
        for strategy, color in styles.items():
            row = all_rows[strategy]
            ax.plot(dates, row['mean_nav'], label=strategy, color=color,
                    linestyle='--' if strategy in fold['baselines'] else '-',
                    linewidth=2 if strategy == 'selected_with_rules' else 1.2)
        for strategy in ('base_ppo', 'signal_ppo', 'risk_ppo', 'joint_ppo', 'joint_sac', 'joint_td3',
                         'equal_50', 'equal_90', 'vol_target10', 'invvol_target10'):
            row = all_rows[strategy]['metrics']
            x, y = row['max_drawdown']['mean']*100, row['total_return']['mean']*100
            square.scatter(x, y, label=strategy, marker='x' if strategy in fold['baselines'] else 'o')
        ax.set_title(f"{name}: {fold['dates'][1]} to {fold['dates'][-1]}")
        ax.set_ylabel('Mean initial-1 NAV')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=.2)
        ax.tick_params(axis='x', labelrotation=20)
        square.set_title(name)
        square.set_xlabel('Maximum drawdown (%)')
        square.set_ylabel('Cumulative return (%)')
        square.legend(fontsize=7, ncol=2)
        square.grid(alpha=.2)
    for ax in list(axes.flat)[len(folds):] + list(squares.flat)[len(folds):]:
        ax.set_visible(False)
    fig.tight_layout(); scatter.tight_layout()
    fig.savefig(output / 'equity.png', dpi=150)
    scatter.savefig(output / 'comparison.png', dpi=150)
    plt.close(fig); plt.close(scatter)


def build_report(run, *, n_bootstrap=2000):
    output = Path(run)
    data = analyze_run(output, n_bootstrap=n_bootstrap)
    (output / 'analysis.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (output / 'report.md').write_text(markdown(data), encoding='utf-8')
    _export_csv(data, output)
    intervals = pd.read_csv(output / 'excess_intervals.csv')
    for index, record in intervals.iterrows():
        strategy = data['evaluations'][str(record.cost_bps)]['folds'][record.fold]['strategies'][record.strategy]
        intervals.loc[index, 'n_seeds'] = len(strategy['seeds'])
        intervals.loc[index, 'n_paths'] = strategy['n_paths']
        intervals.loc[index, 'strategy_kind'] = strategy['kind']
    intervals.to_csv(output / 'excess_intervals.csv', index=False)
    ablations = []
    for cost, evaluation in data['evaluations'].items():
        for fold, row in evaluation['folds'].items():
            for label, result in row['ablations'].items():
                interval = result['paired_daily_excess']['annualized_mean_excess_ci']
                ablations.append({'cost_bps': int(cost), 'fold': fold, 'contrast': label,
                    **{key: value for key, value in result.items() if key != 'paired_daily_excess'},
                    'annualized_daily_excess': result['paired_daily_excess']['annualized_mean_excess_return'],
                    'ci_lower': interval[0], 'ci_upper': interval[1]})
    pd.DataFrame(ablations).to_csv(output / 'ablation_metrics.csv', index=False)
    plots(data, output)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/research_v4')
    args = parser.parse_args()
    result = build_report(args.run)
    print(f"Reported {len(result['evaluations']['5']['folds'])} v4 folds")


if __name__ == '__main__':
    main()
