"""Fixed, train-only Ridge readouts of candidate information; never policy selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_samples(signal_close, features, start, end, horizon=5):
    prices, features = np.asarray(signal_close, float), np.asarray(features, float)
    if (prices.ndim != 2 or features.ndim != 2 or len(prices) != len(features)
            or not np.isfinite(prices).all() or np.any(prices <= 0)
            or not 20 <= start < end < len(prices) or horizon < 2):
        raise ValueError('Invalid sample arrays or interval')
    indices = np.arange(start, end - horizon + 1, horizon)
    if not len(indices) or not np.isfinite(features[start:end + 1]).all():
        raise ValueError('No complete label horizon or nonfinite features')
    market = np.diff(np.log(prices), axis=0).mean(axis=1)
    y = np.asarray([[market[t:t + horizon].sum(),
                     market[t:t + horizon].std(ddof=1) * np.sqrt(252)] for t in indices])
    persistence = np.asarray([market[t - 20:t].std(ddof=1) * np.sqrt(252) for t in indices])
    return {'x': features[indices], 'y': y, 'decision_indices': indices,
            'label_end_indices': indices + horizon, 'persistence_vol': persistence}


def fit_readout(x, y, alpha=10.):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (x.ndim != 2 or y.shape != (len(x), 2) or len(x) < 2
            or not np.isfinite(x).all() or not np.isfinite(y).all() or alpha <= 0):
        raise ValueError('Need finite training features and two targets')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    z, target_mean = (x - mean) / scale, y.mean(axis=0)
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - target_mean))
    return {'mean': mean, 'scale': scale, 'coef': coef, 'intercept': target_mean.copy(),
            'target_mean': target_mean, 'alpha': alpha, 'n_train': len(x)}


def predict_readout(fitted, x):
    prediction = (np.asarray(x, float) - fitted['mean']) / fitted['scale'] @ fitted['coef'] + fitted['intercept']
    # A volatility readout cannot be negative. This fixed projection is also
    # applied on training diagnostics, never selected from evaluation outcomes.
    prediction[:, 1] = np.maximum(prediction[:, 1], 0.)
    return prediction


def readout_metrics(y, prediction, train_mean, persistence_vol):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    mse = np.mean((y - prediction) ** 2, axis=0)
    reference = np.mean((y - np.asarray(train_mean)) ** 2, axis=0)
    result = {}
    for i, target in enumerate(('return', 'volatility')):
        result[target] = {'mse': float(mse[i]), 'train_mean_mse': float(reference[i]),
                          'skill_vs_train_mean': float(1 - mse[i] / reference[i]) if reference[i] > 1e-20 else None}
    persistence = float(np.mean((y[:, 1] - persistence_vol) ** 2))
    result['volatility'].update(persistence_mse=persistence,
                               skill_vs_persistence=float(1 - mse[1] / persistence) if persistence > 1e-20 else None)
    return result


def diagnose(data_dir, run):
    from .risk_env import InformationRiskEnv
    from .study_v3 import digest, load_inputs, write_json
    from .study_v4 import checked_manifest
    run = Path(run)
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    manifest = checked_manifest(data_dir, run, manifest['identity']['steps'])
    if not (run / 'frozen_selection.json').exists():
        raise ValueError('Freeze all selections before test information diagnostics')
    panel, actions = load_inputs(data_dir)
    env = InformationRiskEnv(panel, 120, len(panel.dates) - 1, actions=actions,
                             enhanced_features=True, risk_control=True)
    output, flat, predictions = {}, [], []
    for enhanced in (False, True):
        group = 'expanded' if enhanced else 'base'
        features = np.zeros((len(panel.dates), len(env.market_features(120, enhanced=enhanced))))
        for t in range(120, len(features)):
            features[t] = env.market_features(t, enhanced=enhanced)
        for fold in manifest['folds']:
            samples = {split: make_samples(env.signal_close, features, *fold[split])
                       for split in ('train', 'validation', 'test')}
            fitted = fit_readout(samples['train']['x'], samples['train']['y'])
            record = {'fit': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fitted.items()},
                      'train_last_label_date': str(panel.dates[samples['train']['label_end_indices'][-1]].date()),
                      'evaluations': {}}
            for split in ('validation', 'test'):
                sample = samples[split]
                predicted = predict_readout(fitted, sample['x'])
                metrics = readout_metrics(sample['y'], predicted, fitted['target_mean'], sample['persistence_vol'])
                record['evaluations'][split] = {'n_samples': len(predicted), 'metrics': metrics}
                for target, row in metrics.items():
                    flat.append({'fold': fold['name'], 'features': group, 'split': split,
                                 'target': target, 'n_train': fitted['n_train'], 'n_eval': len(predicted), **row})
                for i, t in enumerate(sample['decision_indices']):
                    predictions.append({'fold': fold['name'], 'features': group, 'split': split,
                                        'decision_date': str(panel.dates[t].date()),
                                        'label_end_date': str(panel.dates[sample['label_end_indices'][i]].date()),
                                        'return_true': sample['y'][i, 0], 'return_predicted': predicted[i, 0],
                                        'vol_true': sample['y'][i, 1], 'vol_predicted': predicted[i, 1],
                                        'vol_persistence': sample['persistence_vol'][i]})
            output[f'{fold["name"]}/{group}'] = record
    write_json(run / 'signal_diagnostics.json', {
        'method': 'Fixed alpha=10 Ridge, train-only standardization, disjoint 5-day labels; volatility prediction floored at zero.',
        'label_market': 'Mean ETF log return (geometric aggregate), not an executable portfolio ledger.',
        'use': 'Descriptive information diagnostic only; no feedback to model selection or feature changes.',
        'source_sha256': digest(__file__), 'records': output})
    pd.DataFrame(flat).to_csv(run / 'signal_metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(run / 'signal_predictions.csv', index=False)
    return flat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/v3')
    parser.add_argument('--run', default='runs/research_v4')
    args = parser.parse_args()
    print(f'Wrote {len(diagnose(args.data_dir, args.run))} signal metric rows')


if __name__ == '__main__':
    main()
