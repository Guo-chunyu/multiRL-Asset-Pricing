import numpy as np
import pandas as pd
import pytest


def test_calendar_folds_do_not_use_test_in_validation_or_training():
    from rlasset.study_v3 import calendar_folds
    dates = pd.bdate_range('2015-04-15', '2026-10-22')
    folds = calendar_folds(dates)
    assert len(folds) == 4
    for f in folds:
        assert 120 <= f['train'][0] < f['train'][1] < f['validation'][0]
        assert f['validation'][1] < f['test'][0] < f['test'][1]
    assert dates[folds[0]['test'][0]].year == 2019
    assert dates[folds[-1]['test'][1]].year == 2026


def test_selection_averages_all_seeds_and_penalizes_drawdown():
    from rlasset.study_v3 import select_candidate
    validation = {
        'first': [{'total_return': .5, 'max_drawdown': .8}, {'total_return': -.1, 'max_drawdown': .2}],
        'second': [{'total_return': .1, 'max_drawdown': .05}, {'total_return': .1, 'max_drawdown': .05}],
    }
    selected = select_candidate(validation)
    assert selected['candidate'] == 'second'
    assert selected['scores']['second'] == pytest.approx(.075)


def test_selection_rejects_missing_seed_and_nonfinite_metrics():
    from rlasset.study_v3 import select_candidate
    with pytest.raises(ValueError):
        select_candidate({'a': [{'total_return': 0., 'max_drawdown': 0.}], 'b': []})
    with pytest.raises(ValueError):
        select_candidate({'a': [{'total_return': np.nan, 'max_drawdown': 0.}]})


def test_zero_rl_residual_has_matching_non_rl_control():
    from rlasset.study_v3 import constant_weights
    np.testing.assert_allclose(constant_weights(3, .9), [.1, .3, .3, .3])
    with pytest.raises(ValueError):
        constant_weights(1, .9)


def test_fold_requires_sufficient_calendar_coverage():
    from rlasset.study_v3 import calendar_folds
    with pytest.raises(ValueError):
        calendar_folds(pd.bdate_range('2022-01-01', '2026-10-22'))


@pytest.mark.parametrize('candidate', ['legacy_ppo', 'robust_ppo', 'robust_sac', 'robust_td3'])
def test_future_market_and_actions_cannot_change_training_or_validation(candidate):
    """Exercise real learners: future rows must not alter parameters or selection inputs."""
    from dataclasses import replace

    import torch

    from rlasset.corporate_actions import CorporateActions, Dividend
    from rlasset.data import MarketPanel
    from rlasset.extractors import SharedTemporalExtractor
    from rlasset.finrl_adapter import make_agent
    from rlasset.study_v3 import make_env, model_policy, run_policy

    torch.set_num_threads(1)
    rng = np.random.default_rng(23)
    closes = 100 * np.exp(np.cumsum(rng.normal(.0002, .01, size=(400, 4)), axis=0))
    opening = closes * np.exp(rng.normal(0, .002, size=closes.shape))
    panel = MarketPanel(
        dates=pd.bdate_range('2020-01-01', periods=400),
        tickers=('a', 'b', 'c', 'd'), open=opening, close=closes,
        high=np.maximum(opening, closes) * 1.01,
        low=np.minimum(opening, closes) * .99,
        volume=np.full_like(closes, 100000.),
    )
    future = replace(panel, **{name: getattr(panel, name).copy()
                              for name in ('open', 'close', 'high', 'low')})
    for name in ('open', 'close', 'high', 'low'):
        getattr(future, name)[341:] *= np.linspace(1.2, 10, 59)[:, None]
    original_actions = CorporateActions.empty(panel)
    changed_actions = CorporateActions.empty(future)
    changed_actions.cash_per_share[350, 0] = 5.
    changed_actions.split_ratio[360, 1] = 2.
    changed_actions = replace(changed_actions, dividends=(Dividend(0, 350, 355, 5.),))
    outcomes = []
    legacy = candidate == 'legacy_ppo'
    for source, actions in ((panel, original_actions), (future, changed_actions)):
        env = make_env(source, actions, (120, 300), candidate, training=True)
        model = make_agent(
            env, algorithm=candidate.split('_')[1], seed=41, total_timesteps=100,
            learning_rate=3e-4 if legacy else 1e-4,
            policy='MultiInputPolicy' if legacy else 'MlpPolicy',
            policy_kwargs={'features_extractor_class': SharedTemporalExtractor} if legacy else None,
            backend='sb3',
        )
        model.learn(total_timesteps=100)
        parameters = {name: tensor.detach().clone() for name, tensor in model.policy.state_dict().items()}
        ledger, metrics = run_policy(
            make_env(source, actions, (301, 340), candidate), model_policy(model),
        )
        assert model.num_timesteps == 100
        assert all(torch.equal(parameters[name], tensor) for name, tensor in model.policy.state_dict().items())
        outcomes.append((parameters, ledger, metrics))
    assert outcomes[0][0].keys() == outcomes[1][0].keys()
    for name in outcomes[0][0]:
        assert torch.equal(outcomes[0][0][name], outcomes[1][0][name]), name
    pd.testing.assert_frame_equal(outcomes[0][1], outcomes[1][1], check_exact=True)
    assert outcomes[0][2] == outcomes[1][2]
