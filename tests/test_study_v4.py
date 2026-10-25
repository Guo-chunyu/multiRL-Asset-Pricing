"""The v4 runner must keep ablations, selection, and test boundaries explicit."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _panel(days=400, assets=3):
    from rlasset.data import MarketPanel
    rng = np.random.default_rng(22)
    close = 100 * np.exp(np.cumsum(rng.normal(.0001, .01, (days, assets)), axis=0))
    opening = close * np.exp(rng.normal(0, .002, close.shape))
    return MarketPanel(pd.bdate_range('2020-01-01', periods=days),
                       tuple(f'ETF{i}' for i in range(assets)), opening, close,
                       np.maximum(close, opening) * 1.01, np.minimum(close, opening) * .99,
                       rng.uniform(10000, 20000, close.shape))


def test_factorial_candidates_change_only_registered_switches():
    # A swapped factor switch invalidates the entire attribution experiment.
    from rlasset.study_v4 import CANDIDATES, candidate_config
    assert CANDIDATES == ('base_ppo', 'signal_ppo', 'risk_ppo', 'joint_ppo', 'joint_sac', 'joint_td3')
    assert candidate_config('base_ppo') == (False, False, 'ppo')
    assert candidate_config('signal_ppo') == (True, False, 'ppo')
    assert candidate_config('risk_ppo') == (False, True, 'ppo')
    assert candidate_config('joint_ppo') == (True, True, 'ppo')
    assert candidate_config('joint_sac') == (True, True, 'sac')
    assert candidate_config('joint_td3') == (True, True, 'td3')
    with pytest.raises(ValueError, match='candidate'):
        candidate_config('unregistered_ppo')


def test_rule_fallback_is_reported_separately_from_best_rl():
    from rlasset.study_v4 import choose_candidates
    rl = {'base_ppo': [{'total_return': .2, 'max_drawdown': .3}] * 3,
          'joint_ppo': [{'total_return': .1, 'max_drawdown': .02}] * 3}
    rules = {'vol_target10': {'total_return': .11, 'max_drawdown': .02},
             'invvol_target10': {'total_return': .12, 'max_drawdown': .1}}
    choice = choose_candidates(rl, rules)
    assert choice['candidate'] == 'joint_ppo'
    assert choice['deploy_candidate'] == 'vol_target10'
    assert choice['deploy_scores']['vol_target10'] == pytest.approx(.1)
    assert choice['selected_on'] == 'validation'


def test_partial_seed_scores_cannot_sneak_into_selection():
    from rlasset.study_v4 import choose_candidates
    with pytest.raises(ValueError, match='seed'):
        choose_candidates({'base_ppo': [{'total_return': .1, 'max_drawdown': .1}]}, {})


def test_undefined_ratios_are_checked_without_inventing_zero_values():
    from rlasset.study_v4 import _check_metrics
    _check_metrics({'sharpe': None, 'total_return': 0.}, {'sharpe': None, 'total_return': 0.})
    with pytest.raises(ValueError, match='metrics disagree'):
        _check_metrics({'sharpe': 0.}, {'sharpe': None})


def test_record_signature_detects_validation_metric_edits():
    from rlasset.study_v4 import signed_record, check_record_signature
    source = {'candidate': 'joint_ppo', 'validation_metrics': {'total_return': .1}}
    signed = signed_record(source)
    check_record_signature(signed)
    assert 'record_sha256' not in source
    signed['validation_metrics']['total_return'] = .9
    with pytest.raises(ValueError, match='record'):
        check_record_signature(signed)


def test_fold_filter_never_changes_the_full_manifest():
    from rlasset.study_v4 import selected_folds
    manifest = {'folds': [{'name': 'fold_01'}, {'name': 'fold_02'}]}
    assert selected_folds(manifest, 'fold_02') == [{'name': 'fold_02'}]
    assert len(manifest['folds']) == 2
    with pytest.raises(ValueError, match='fold'):
        selected_folds(manifest, 'fold_99')


def test_unregistered_cost_is_rejected_before_any_io():
    from rlasset.study_v4 import evaluate
    with pytest.raises(ValueError, match='cost'):
        evaluate('missing', 'missing', 20000, -1)


def test_ledger_metrics_are_recomputed_and_dates_must_match(tmp_path):
    from rlasset.study_v3 import save_ledger
    from rlasset.study_v4 import validation_metrics
    ledger = pd.DataFrame({'date': pd.bdate_range('2020-01-01', periods=4),
                           'net_return': [0., .1, -.1, .01],
                           'portfolio_value': [1., 1.1, .99, .9999],
                           'turnover': [0., 1., 0., .1],
                           'weights': [[1., 0., 0.], [.2, .4, .4], [.2, .4, .4], [.2, .4, .4]]})
    path = tmp_path / 'validation.csv'
    save_ledger(path, ledger)
    expected_dates = pd.DatetimeIndex(ledger.date)
    result = validation_metrics(path, expected_dates)
    assert result['total_return'] == pytest.approx(-.0001)
    assert result['return_days'] == 3
    assert result['mean_cash_weight'] == pytest.approx(.2)
    with pytest.raises(ValueError, match='calendar'):
        validation_metrics(path, expected_dates[:-1])
    ledger.loc[2, 'portfolio_value'] = 5
    save_ledger(path, ledger)
    with pytest.raises(ValueError, match='NAV'):
        validation_metrics(path, expected_dates)


def _record_fixture(directory):
    from rlasset.study_v3 import digest, save_ledger, write_json
    from rlasset.study_v4 import signed_record, validation_metrics
    panel = _panel()
    fold = {'name': 'fold_01', 'validation': [200, 203]}
    ledger = pd.DataFrame({'date': panel.dates[200:204], 'net_return': [0., .1, -.1, .01],
                           'portfolio_value': [1., 1.1, .99, .9999], 'turnover': [0., 1., 0., .1],
                           'weights': [[1., 0., 0., 0.], [.1, .3, .3, .3], [.1, .3, .3, .3], [.1, .3, .3, .3]]})
    model_path, ledger_path = directory / 'base_ppo.zip', directory / 'validation_base_ppo.csv'
    model_path.write_bytes(b'model checksum fixture, not a trained model')
    save_ledger(ledger_path, ledger)
    record = signed_record({'candidate': 'base_ppo', 'seed': 7, 'fold': fold['name'],
                            'actual_steps': 100, 'requested_steps': 100,
                            'model_sha256': digest(model_path), 'validation_sha256': digest(ledger_path),
                            'validation_metrics': validation_metrics(ledger_path, panel.dates[200:204])})
    write_json(directory / 'base_ppo.json', record)
    return panel, fold, record


def test_resume_checks_model_and_validation_file_contents(tmp_path):
    from rlasset.study_v4 import checked_record
    panel, fold, record = _record_fixture(tmp_path)
    assert checked_record(tmp_path, 'base_ppo', 7, fold, panel, 100) == record
    path = tmp_path / 'base_ppo.zip'
    original = path.read_bytes()
    path.write_bytes(b'changed model')
    with pytest.raises(ValueError, match='Model checksum'):
        checked_record(tmp_path, 'base_ppo', 7, fold, panel, 100)
    path.write_bytes(original)
    path = tmp_path / 'validation_base_ppo.csv'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='Validation ledger checksum'):
        checked_record(tmp_path, 'base_ppo', 7, fold, panel, 100)


def test_resigned_metadata_still_must_match_the_ledger(tmp_path):
    from rlasset.study_v3 import write_json
    from rlasset.study_v4 import checked_record, signed_record
    panel, fold, record = _record_fixture(tmp_path)
    record['validation_metrics']['total_return'] = .9
    write_json(tmp_path / 'base_ppo.json', signed_record(record))
    with pytest.raises(ValueError, match='metrics disagree'):
        checked_record(tmp_path, 'base_ppo', 7, fold, panel, 100)


def test_test_evaluation_requires_frozen_selection(tmp_path):
    from rlasset.study_v4 import evaluate
    with pytest.raises(ValueError, match='Freeze all'):
        evaluate(tmp_path / 'data', tmp_path / 'run', 20000, 5)


@pytest.mark.parametrize('candidate', ['base_ppo', 'signal_ppo', 'risk_ppo', 'joint_ppo', 'joint_sac', 'joint_td3'])
def test_future_price_volume_and_corporate_events_cannot_change_fit_or_validation(candidate):
    """A whole-panel factor scaler would change parameters despite equal pasts."""
    from dataclasses import replace
    import torch
    from rlasset.corporate_actions import CorporateActions, Dividend
    from rlasset.finrl_adapter import make_agent
    from rlasset.study_v4 import candidate_config, make_env, model_policy, run_policy
    torch.set_num_threads(1)
    panel = _panel()
    future = replace(panel, **{name: getattr(panel, name).copy()
                              for name in ('open', 'close', 'high', 'low', 'volume')})
    for name in ('open', 'close', 'high', 'low', 'volume'):
        getattr(future, name)[341:] *= np.linspace(1.2, 10, 59)[:, None]
    actions = CorporateActions.empty(panel)
    changed = CorporateActions.empty(future)
    changed.cash_per_share[350, 0] = 5.
    changed.split_ratio[360, 1] = 2.
    changed = replace(changed, dividends=(Dividend(0, 350, 355, 5.),))
    outcomes = []
    for source, events in ((panel, actions), (future, changed)):
        env = make_env(source, events, (120, 300), candidate, training=True)
        model = make_agent(env, algorithm=candidate_config(candidate)[2], seed=41, total_timesteps=100,
                           learning_rate=1e-4, policy='MlpPolicy')
        model.learn(total_timesteps=100)
        parameters = {name: tensor.detach().clone() for name, tensor in model.policy.state_dict().items()}
        ledger, metrics = run_policy(make_env(source, events, (301, 340), candidate), model_policy(model))
        assert model.num_timesteps == 100
        assert all(torch.equal(parameters[name], tensor) for name, tensor in model.policy.state_dict().items())
        outcomes.append((parameters, ledger, metrics))
    for name, tensor in outcomes[0][0].items():
        assert torch.equal(tensor, outcomes[1][0][name]), name
    pd.testing.assert_frame_equal(outcomes[0][1], outcomes[1][1], check_exact=True)
    assert outcomes[0][2] == outcomes[1][2]
