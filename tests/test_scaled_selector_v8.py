"""Synthetic count-only scaling, frozen-v7 parity and backend smoke checks."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from rlasset.corporate_actions import CorporateActions, Dividend
from rlasset.data import MarketPanel
from rlasset.masked_selector_v7 import MaskedSelectorEnvV7
from rlasset.scaled_selector_v8 import ScaledSelectorEnvV8


def panel_fixture(days=300, assets=3):
    rng = np.random.default_rng(89)
    returns = rng.normal(.0003, .018, (days, 1)) + rng.normal(0, .006, (days, assets))
    close = 100 * np.exp(np.cumsum(returns, axis=0))
    return MarketPanel(pd.bdate_range('2020-01-01', periods=days),
                       tuple(chr(65+i) for i in range(assets)),
                       close.copy(), close, close*1.01, close*.99,
                       np.full_like(close, 10000))


def env(panel=None, **kwargs):
    config = dict(start=120, end=280)
    config.update(kwargs)
    return ScaledSelectorEnvV8(panel if panel is not None else panel_fixture(), **config)


def assert_only_count_changed(scaled, original, index, divisor):
    expected = original.copy()
    expected[index] /= np.float32(divisor)
    np.testing.assert_array_equal(scaled, expected)
    assert scaled.dtype == original.dtype == np.float32


@pytest.mark.parametrize('scale', [1, 20])
@pytest.mark.parametrize('penalty', [False, True])
def test_exact_v7_account_reward_mask_and_single_feature_parity(scale, penalty):
    panel = panel_fixture()
    a = env(panel, count_scale=scale, drawdown_penalty=penalty)
    b = MaskedSelectorEnvV7(panel, 120, 280, mask_mode='phase', drawdown_penalty=penalty)
    assert a.count_index == 57
    assert_only_count_changed(a.reset(seed=4)[0], b.reset(seed=4)[0], 57, scale)
    for day in range(160):
        np.testing.assert_array_equal(a.action_masks(), b.action_masks())
        x, y = a.step(day % 4), b.step(day % 4)
        assert_only_count_changed(x[0], y[0], 57, scale)
        assert x[0][57] == np.float32(min(day+1, 20)) / np.float32(scale)
        assert x[1:] == y[1:]
        assert a.history == b.history
        assert a.performance_history == b.performance_history
        for sa, sb in zip(a.shadows, b.shadows):
            assert sa.history == sb.history
    assert a.objective == b.objective
    assert a.history[-1]['rebalance_executed'] is False


@pytest.mark.parametrize('scale', [1, 20])
def test_split_dividend_entitlement_receivable_and_payment_parity(scale):
    panel = panel_fixture()
    actions = CorporateActions.empty(panel)
    for key in ('open', 'close', 'high', 'low'):
        getattr(panel, key)[125:, 0] /= 2
        getattr(panel, key)[130:, 1] -= 3
    actions.split_ratio[125, 0] = 2
    actions.cash_per_share[130, 1] = 3
    actions = replace(actions, dividends=(Dividend(1, 130, 139, 3.),))
    a = env(panel, count_scale=scale, actions=actions, end=160)
    b = MaskedSelectorEnvV7(panel, 120, 160, actions=actions, mask_mode='phase')
    for day in range(40):
        x, y = a.step(1 if day < 30 else 2), b.step(1 if day < 30 else 2)
        assert_only_count_changed(x[0], y[0], 57, scale)
        assert x[1:] == y[1:]
        assert a.history == b.history
        for sa, sb in zip(a.shadows, b.shadows):
            assert sa.history == sb.history
    assert any(row['receivable'] > 0 for row in a.history)
    assert any(row['dividend_paid'] > 0 for row in a.history)


def test_future_prices_and_events_do_not_affect_observation_or_history():
    panel = panel_fixture()
    future = replace(panel, **{key:getattr(panel, key).copy()
                              for key in ('open', 'close', 'high', 'low', 'volume')})
    for key in ('open', 'close', 'high', 'low', 'volume'):
        getattr(future, key)[161:] *= 3
    actions = CorporateActions.empty(future)
    actions.cash_per_share[180, 1] = 5
    actions.split_ratio[190, 0] = 1.25
    actions = replace(actions, dividends=(Dividend(1, 180, 190, 5.),))
    a, b = env(panel, end=160), env(future, end=160, actions=actions)
    np.testing.assert_array_equal(a.reset()[0], b.reset()[0])
    for day in range(40):
        x, y = a.step(day % 4), b.step(day % 4)
        np.testing.assert_array_equal(x[0], y[0])
        np.testing.assert_array_equal(a.action_masks(), b.action_masks())
        assert x[1:] == y[1:]


def test_reset_dynamic_dispatch_count_cold_start_and_fragment_schedule():
    a, b = env(training=True), env(training=True, count_scale=1)
    for e in (a, b):
        observation, _ = e.reset(seed=19)
        assert observation[e.count_index] == 0
        assert e.training_episode_schedule == [(e.base.start, e.base.end)]
    for episode in range(3):
        assert a.training_episode_schedule == b.training_episode_schedule
        for day in range(126):
            x, y = a.step(day % 4), b.step(day % 4)
            assert_only_count_changed(x[0], y[0], 57, 20)
            assert x[1:] == y[1:]
            assert (x[2], x[3]) == (False, day == 125)
        for e in (a, b):
            with pytest.raises(RuntimeError):
                e.step(0)
            observation, _ = e.reset()
            assert observation[e.count_index] == 0
            assert e.previous_executed_expert is None
            assert e.performance_history == [[], [], [], []]
            assert e.action_masks().all()
            assert len(e.training_episode_schedule) == episode + 2
    a.reset(seed=19)
    assert a.training_episode_schedule == b.training_episode_schedule[:1]
    e = env(end=125)
    for _ in range(5):
        result = e.step(0)
    assert result[2:4] == (True, False)


@pytest.mark.parametrize('assets', [2, 3, 4])
def test_count_index_derived_from_asset_count(assets):
    a, b = env(panel_fixture(assets=assets)), env(panel_fixture(assets=assets), count_scale=1)
    assert a.count_index == 13*assets+18
    for _ in range(21):
        x, y = a.step(1), b.step(1)
        assert_only_count_changed(x[0], y[0], a.count_index, 20)
    assert x[0][a.count_index] == 1


@pytest.mark.parametrize('bad', [0, -1, 2, 19, 21, True, False, np.bool_(True),
                                 np.nan, np.inf, '20', None, [20], np.array([20])])
def test_invalid_count_scale(bad):
    with pytest.raises(ValueError, match='count_scale'):
        env(count_scale=bad)


@pytest.mark.parametrize('scale', [1, 20])
def test_gym_contract(scale):
    check_env(env(training=True, count_scale=scale), skip_render_check=True)


def test_official_maskable_ppo_synthetic_smoke():
    import sb3_contrib
    import torch
    from sb3_contrib import MaskablePPO
    assert sb3_contrib.__version__ == '2.7.1'
    torch.set_num_threads(1)
    e = env(training=True)
    model = MaskablePPO('MlpPolicy', e, n_steps=10, batch_size=10, n_epochs=1,
                       ent_coef=0.0, policy_kwargs=dict(net_arch=[8]), seed=7,
                       device='cpu', verbose=0)
    model.learn(total_timesteps=20)  # Synthetic API test, not market evidence.
    assert np.isfinite(model.rollout_buffer.advantages).all()
    observations = model.rollout_buffer.observations.reshape(-1, e.observation_space.shape[0])
    assert np.all((observations[:, e.count_index] >= 0) & (observations[:, e.count_index] <= 1))
    obs, _ = e.reset(seed=7)
    for day in range(12):
        masks = e.action_masks()
        for deterministic in (False, True):
            action, _ = model.predict(obs, action_masks=masks, deterministic=deterministic)
            assert masks[int(action)]
        obs, _, _, _, info = e.step(action)
        assert info['action_ignored'] == (day % 5 != 0)
