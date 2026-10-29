"""v5 bounded residual, causal anchor and inherited raw-ledger contracts."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from rlasset.corporate_actions import CorporateActions, Dividend
from rlasset.data import MarketPanel
from rlasset.residual_env import ResidualRiskEnv, trend_volatility_target
from rlasset.risk_env import InformationRiskEnv


def panel_fixture(days=280, mode='random'):
    rng = np.random.default_rng(91)
    returns = rng.normal(.0003, .018, (days, 1)) + rng.normal(0, .006, (days, 3))
    if mode != 'random':
        returns[:] = {'up': .001, 'down': -.001, 'flat': 0}[mode]
    close = 100 * np.exp(np.cumsum(returns, axis=0))
    return MarketPanel(pd.bdate_range('2020-01-01', periods=days), ('A', 'B', 'C'),
                       close.copy(), close, close*1.01, close*.99, np.full_like(close, 10000))


def env(panel=None, **kwargs):
    options = dict(start=120, end=260)
    options.update(kwargs)
    return ResidualRiskEnv(panel if panel is not None else panel_fixture(), **options)


@pytest.mark.parametrize('gate', [True, False])
def test_zero_residual_is_exact_anchor_and_standalone_rule(gate):
    e = env(trend_gate=gate)
    for t in range(120, 260):
        e.t = t
        np.testing.assert_array_equal(e.action_to_weights(np.zeros(4)), e.anchor_target())
        np.testing.assert_array_equal(e.anchor_target(), trend_volatility_target(e.signal_close, t, gate))


def test_gate_off_anchor_equals_existing_volatility_rule():
    panel = panel_fixture()
    a, b = env(panel, trend_gate=False), InformationRiskEnv(panel, 120, 260, risk_control=True)
    for t in range(120, 260):
        a.t = b.t = t
        np.testing.assert_array_equal(a.anchor_target(), b.risk_target(np.ones(3)))


@pytest.mark.parametrize('mode, expected', [('up', 1.), ('down', 0.), ('flat', 0.)])
def test_trend_gate_uses_strict_close_above_fixed_trailing_means(mode, expected):
    e = env(panel_fixture(mode=mode))
    assert e.trend_participation() == expected
    for action in (np.ones(4), -np.ones(4), np.zeros(4)):
        assert sum(e.action_to_weights(action)[1:]) == pytest.approx(expected)


def test_partial_gate_is_outside_exposure_cap_and_mean_across_assets():
    panel = panel_fixture(mode='up')
    for name in ('open', 'close', 'high', 'low'):
        getattr(panel, name)[:, 0] = 100 * np.exp(-.001 * np.arange(len(panel.dates)))
    e = env(panel)
    assert e.trend_participation() == pytest.approx(2/3)
    assert e.anchor_target()[0] == pytest.approx(1/3)


def test_residual_bounds_and_budget_mapping_are_independently_recomputed():
    e = env()
    rng = np.random.default_rng(9)
    for t in range(120, 260):
        e.t = t
        action = rng.uniform(-1, 1, 4)
        q = np.exp(.10*action[1:]); q /= q.sum()
        sigma = max(.01, *(np.sqrt(252*q @ e.estimated_covariance(t,h) @ q) for h in (20,60)))
        g = np.mean([e.signal_close[t] > e.signal_close[t-h+1:t+1].mean(axis=0) for h in (60,120)])
        budget = .1*(1+.25*action[0])
        exposure = g*min(1,budget/sigma)
        target = e.action_to_weights(action)
        np.testing.assert_allclose(target, np.r_[1-exposure,exposure*q], atol=1e-15)
        assert .075 <= budget <= .125
        assert q.max()/q.min() <= np.exp(.2)+1e-14
        assert target.min() >= 0 and target[1:].max() <= .5
        assert target.sum() == pytest.approx(1.)
        assert target[1:].sum() <= g+1e-14


def test_zero_rule_account_equivalence_with_dividends_splits_and_weekly_hold():
    panel = panel_fixture(mode='up')
    actions = CorporateActions.empty(panel)
    # Real raw-price ex/split mechanics; signal adjusts but execution stays raw.
    for name in ('open', 'close', 'high', 'low'):
        getattr(panel, name)[125:, 0] /= 2
        getattr(panel, name)[130:, 1] -= 3
    actions.split_ratio[125, 0] = 2
    actions.cash_per_share[130, 1] = 3
    actions = replace(actions, dividends=(Dividend(1,130,135,3.),))
    learned, rule = env(panel, actions=actions, end=160), env(panel, actions=actions, end=160)
    for i in range(40):
        a, b = learned.step(np.zeros(4)), rule.step_weights(rule.anchor_target())
        np.testing.assert_array_equal(a[0], b[0]); assert a[1:] == b[1:]
        assert a[-1]['rebalance_executed'] == (i % 5 == 0)
    assert any(row.get('receivable',0) > 0 for row in learned.history)


def test_future_prices_volume_corporate_actions_cannot_change_past_decisions_or_ledger():
    panel = panel_fixture()
    future = replace(panel, **{k:getattr(panel,k).copy() for k in ('open','close','high','low','volume')})
    for name in ('open','close','high','low','volume'):
        getattr(future, name)[161:] *= 3
    actions = CorporateActions.empty(future)
    actions.cash_per_share[180,1] = 5; actions.split_ratio[190,0] = 1.25
    actions = replace(actions, dividends=(Dividend(1,180,190,5.),))
    a,b = env(panel,end=160), env(future,end=160,actions=actions)
    for t in range(120,161):
        np.testing.assert_array_equal(a.anchor_target(t), b.anchor_target(t))
    for _ in range(40):
        action = np.array([.3,-.7,.2,.1])
        np.testing.assert_array_equal(a.action_to_weights(action),b.action_to_weights(action))
        x,y = a.step(action),b.step(action)
        np.testing.assert_array_equal(x[0],y[0]); assert x[1:] == y[1:]


def test_current_close_is_used_but_next_open_never_enters_decision():
    panel = panel_fixture(mode='flat')
    changed = replace(panel, close=panel.close.copy(), open=panel.open.copy())
    changed.close[120] *= 1.1
    changed.open[121] *= 5
    first, second = env(panel), env(changed)
    assert first.trend_participation() == 0 and second.trend_participation() == 1
    target = second.anchor_target()
    changed.open[121] *= 2
    np.testing.assert_array_equal(target, env(changed).anchor_target())


@pytest.mark.parametrize('action', [[0], [0,0,np.nan,0], [0,2,0,0]])
def test_invalid_action_cannot_advance_state(action):
    e = env()
    with pytest.raises(ValueError): e.step(action)
    assert e.t == 120 and len(e.history) == 1


@pytest.mark.parametrize('gate', [True,False])
def test_gym_contract_and_pure_anchor(gate):
    e = env(trend_gate=gate, training=True, episode_length=20)
    check_env(e, skip_render_check=True)
    obs,_ = e.reset(seed=17)
    assert obs.shape == (34,) and e.observation_space.contains(obs)
    before = (e.t,e.value,e.cash,e.receivable,len(e.history))
    e.anchor_target()[:] = 999
    assert not np.any(e.anchor_target() == 999)
    assert before == (e.t,e.value,e.cash,e.receivable,len(e.history))


@pytest.mark.parametrize('gate', [1, 'yes', None])
def test_gate_flag_must_be_boolean(gate):
    with pytest.raises(ValueError): env(trend_gate=gate)
