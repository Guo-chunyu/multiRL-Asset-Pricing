"""Synthetic v7 causal masks, frozen-v6 parity and official-backend smoke."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from rlasset.corporate_actions import CorporateActions, Dividend
from rlasset.data import MarketPanel
from rlasset.masked_selector_v7 import MaskedSelectorEnvV7
from rlasset.selector_env_v6 import FixedExpertSelectorEnv


def panel_fixture(days=300):
    rng = np.random.default_rng(89)
    returns = rng.normal(.0003, .018, (days, 1)) + rng.normal(0, .006, (days, 3))
    close = 100 * np.exp(np.cumsum(returns, axis=0))
    return MarketPanel(pd.bdate_range('2020-01-01', periods=days), ('A', 'B', 'C'),
                       close.copy(), close, close*1.01, close*.99, np.full_like(close, 10000))


def env(panel=None, **kwargs):
    config = dict(start=120, end=280)
    config.update(kwargs)
    return MaskedSelectorEnvV7(panel if panel is not None else panel_fixture(), **config)


@pytest.mark.parametrize('mode', ['phase', 'control'])
def test_masks_shape_type_pre_step_phase_and_freshness(mode):
    e = env(mask_mode=mode)
    for day in range(16):
        mask = e.action_masks()
        assert mask.shape == (4,) and mask.dtype == np.bool_
        assert e.rebalance_phase == day % 5
        expected = np.array([True]*4 if mode == 'control' or day % 5 == 0
                            else [True, False, False, False])
        np.testing.assert_array_equal(mask, expected)
        assert mask.sum() == (4 if mode == 'control' or day % 5 == 0 else 1)
        mask[:] = False
        np.testing.assert_array_equal(e.action_masks(), expected)
        info = e.step(1)[-1]
        assert info['rebalance_executed'] == (day % 5 == 0)


@pytest.mark.parametrize('penalty', [False, True])
@pytest.mark.parametrize('mode', ['phase', 'control'])
def test_exact_v6_observation_reward_and_ledger_parity(mode, penalty):
    panel = panel_fixture()
    a = env(panel, mask_mode=mode, drawdown_penalty=penalty)
    b = FixedExpertSelectorEnv(panel, 120, 280, drawdown_penalty=penalty)
    np.testing.assert_array_equal(a.reset(seed=4)[0], b.reset(seed=4)[0])
    for day in range(160):
        x, y = a.step(day % 4), b.step(day % 4)
        np.testing.assert_array_equal(x[0], y[0])
        assert x[1:] == y[1:]
        assert a.history == b.history
        for sa, sb in zip(a.shadows, b.shadows):
            assert sa.history == sb.history
    assert a.objective == b.objective


def test_identical_executions_ignore_other_requests_with_corporate_actions():
    panel = panel_fixture()
    actions = CorporateActions.empty(panel)
    for key in ('open', 'close', 'high', 'low'):
        getattr(panel, key)[125:, 0] /= 2
        getattr(panel, key)[130:, 1] -= 3
    actions.split_ratio[125, 0] = 2
    actions.cash_per_share[130, 1] = 3
    actions = replace(actions, dividends=(Dividend(1, 130, 139, 3.),))
    a, b = (env(panel, mask_mode=mode, actions=actions, end=160)
            for mode in ('phase', 'control'))
    for day in range(40):
        executed_choice = 1 if day < 30 else 2
        x = a.step(executed_choice if day % 5 == 0 else 0)
        y = b.step(executed_choice if day % 5 == 0 else 1 + day % 3)
        np.testing.assert_array_equal(x[0], y[0])
        assert x[1:4] == y[1:4]
        # Only logged ignored request labels may differ, never accounting.
        ignored_labels = {'requested_expert', 'requested_expert_index'}
        assert {k:v for k,v in x[-1].items() if k not in ignored_labels} == {
            k:v for k,v in y[-1].items() if k not in ignored_labels}
        np.testing.assert_array_equal(a.base.shares, b.base.shares)
        for sa, sb in zip(a.shadows, b.shadows):
            assert sa.history == sb.history
    assert any(r['receivable'] > 0 for r in a.history)
    assert any(r['dividend_paid'] > 0 for r in a.history)


@pytest.mark.parametrize('mode', ['phase', 'control'])
def test_future_data_and_events_do_not_change_prefix_masks(mode):
    panel = panel_fixture()
    future = replace(panel, **{k:getattr(panel, k).copy()
                              for k in ('open', 'close', 'high', 'low', 'volume')})
    for key in ('open', 'close', 'high', 'low', 'volume'):
        getattr(future, key)[161:] *= 3
    actions = CorporateActions.empty(future)
    actions.cash_per_share[180, 1] = 5
    actions.split_ratio[190, 0] = 1.25
    actions = replace(actions, dividends=(Dividend(1, 180, 190, 5.),))
    a = env(panel, end=160, mask_mode=mode)
    b = env(future, end=160, actions=actions, mask_mode=mode)
    for day in range(40):
        np.testing.assert_array_equal(a.action_masks(), b.action_masks())
        x, y = a.step(day % 4), b.step(day % 4)
        np.testing.assert_array_equal(x[0], y[0])
        assert x[1:] == y[1:]


@pytest.mark.parametrize('mode', ['phase', 'control'])
def test_reset_training_truncation_and_eval_termination_masks(mode):
    e = env(training=True, mask_mode=mode)
    for seed in (4, 9):
        e.reset(seed=seed)
        assert e.base.end - e.base.start == 126
        assert e.rebalance_phase == 0 and e.action_masks().all()
        for day in range(126):
            _, _, term, trunc, _ = e.step(0)
            assert (term, trunc) == (False, day == 125)
        assert e.rebalance_phase == 1
        assert e.action_masks().sum() == (1 if mode == 'phase' else 4)
        with pytest.raises(RuntimeError):
            e.step(0)
    e = env(end=125, mask_mode=mode)
    for _ in range(5):
        result = e.step(0)
    assert result[2:4] == (True, False)
    assert e.action_masks().all()  # phase 0, even at terminal observation
    e.reset()
    assert e.action_masks().all()


def test_invalid_mode_and_gym_contract():
    with pytest.raises(ValueError, match='mask_mode'):
        env(mask_mode='future')
    check_env(env(training=True), skip_render_check=True)


@pytest.mark.parametrize('mode', ['phase', 'control'])
def test_official_maskable_ppo_smoke_and_explicit_predict_masks(mode):
    import sb3_contrib
    import torch
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    assert sb3_contrib.__version__ == '2.7.1'
    assert MaskablePPO.__module__ == 'sb3_contrib.ppo_mask.ppo_mask'
    torch.set_num_threads(1)
    e = env(mask_mode=mode, training=True)
    model = MaskablePPO('MlpPolicy', e, n_steps=10, batch_size=10, n_epochs=1,
                        ent_coef=0.0, policy_kwargs=dict(net_arch=[8]), seed=7,
                        device='cpu', verbose=0)
    model.learn(total_timesteps=20)  # Synthetic backend check, not an experiment.
    masks = model.rollout_buffer.action_masks.reshape(-1, 4)
    assert np.isfinite(model.rollout_buffer.advantages).all()
    assert set(np.sum(masks, axis=-1)) == ({1, 4} if mode == 'phase' else {4})
    obs, _ = e.reset(seed=7)
    for day in range(12):
        mask = get_action_masks(e)
        for deterministic in (False, True):
            action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
            assert mask[int(action)]
            if mode == 'phase' and e.rebalance_phase != 0:
                assert int(action) == 0
        obs, _, _, _, info = e.step(action)
        assert info['action_ignored'] == (day % 5 != 0)


def test_paired_training_fragment_schedule_seeded_reset_discards_constructor():
    a, b = (env(training=True, mask_mode=mode) for mode in ('phase', 'control'))
    for e in (a, b):
        e.reset(seed=19)
        assert e.training_episode_schedule == [(e.base.start, e.base.end)]
    for episode in range(3):
        assert a.training_episode_schedule == b.training_episode_schedule
        for day in range(126):
            a.step(1 if day % 5 == 0 else 0)
            b.step(1)
        a.reset()
        b.reset()
        assert len(a.training_episode_schedule) == episode + 2
    assert a.training_episode_schedule == b.training_episode_schedule
    a.reset(seed=19)
    assert a.training_episode_schedule == b.training_episode_schedule[:1]
    evaluation = env()
    evaluation.reset(seed=19)
    assert evaluation.training_episode_schedule == []
