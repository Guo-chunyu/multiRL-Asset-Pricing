"""Causal information ablation and ex-ante risk-budget accounting contracts."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from rlasset.corporate_actions import CorporateActions, Dividend
from rlasset.data import MarketPanel
from rlasset.robust_env import RobustPortfolioEnv
from rlasset.risk_env import InformationRiskEnv


def market_panel(days=280, flat=False):
    rng = np.random.default_rng(91)
    common = rng.normal(.0003, .018, (days, 1))
    returns = common + rng.normal(0, .006, (days, 3))
    close = np.full((days, 3), 100.) if flat else 100 * np.exp(np.cumsum(returns, axis=0))
    return MarketPanel(
        pd.bdate_range("2020-01-01", periods=days), ("A", "B", "C"),
        close.copy(), close, close * 1.01, close * .99,
        np.full((days, 3), 10000.),
    )


def env(panel=None, **kwargs):
    options = dict(start=120, end=260, enhanced_features=True, risk_control=True)
    options.update(kwargs)
    return InformationRiskEnv(panel or market_panel(), **options)


def test_both_ablations_off_are_step_for_step_identical_to_v3():
    panel = market_panel()
    baseline = RobustPortfolioEnv(panel, 120, 150, cost_bps=13)
    changed = env(panel, end=150, cost_bps=13, enhanced_features=False, risk_control=False)
    np.testing.assert_array_equal(baseline.reset(seed=7)[0], changed.reset(seed=7)[0])
    rng = np.random.default_rng(7)
    for _ in range(30):
        action = rng.uniform(-1, 1, 4)
        np.testing.assert_array_equal(baseline.action_to_weights(action), changed.action_to_weights(action))
        original, actual = baseline.step(action), changed.step(action)
        np.testing.assert_array_equal(original[0], actual[0])
        assert original[1:] == actual[1:]


def test_feature_dimensions_and_market_features_exclude_account_state():
    e = env()
    assert e.observation_space.shape == (45,)
    assert e.market_features(120, enhanced=False).shape == (24,)
    assert e.market_features(120).shape == (37,)
    before = e.market_features(120)
    e.cash = .2
    e.receivable = .1
    np.testing.assert_array_equal(before, e.market_features(120))
    before[:] = 99  # Public arrays cannot mutate the internal feature cache.
    assert not np.any(e.market_features(120) == 99)
    assert len(e.market_feature_names(enhanced=False)) == 24
    assert len(e.market_feature_names()) == 37


def test_risk_budget_mapping_and_portfolio_covariance_match_manual_calculation():
    e = env()
    e.reset()
    action = np.array([0., .8, -.2, -.7])
    q = np.exp(.75 * action[1:])
    from rlasset.robust_env import capped_risky_weights
    q = capped_risky_weights(q, 1., .5)
    estimated = []
    for window in (20, 60):
        prices = e.signal_close[120-window:121]
        returns = np.log(prices[1:] / prices[:-1])
        sample = np.cov(returns, rowvar=False, ddof=1)
        covariance = .75 * sample + .25 * np.diag(np.diag(sample))
        estimated.append(np.sqrt(252 * (q @ covariance @ q)))
    sigma = max(*estimated, .01)
    exposure = min(1., .1 / sigma)
    np.testing.assert_allclose(e.action_to_weights(action), np.r_[1-exposure, exposure*q])
    assert e.estimated_risk(q) == pytest.approx(sigma)
    targets = [e.action_to_weights(np.r_[a, action[1:]]) for a in (-1, 0, 1)]
    assert [1-target[0] for target in targets] == sorted(1-target[0] for target in targets)
    assert 1-targets[0][0] == pytest.approx(min(1, .05/sigma))
    assert 1-targets[-1][0] == pytest.approx(min(1, .20/sigma))


def test_flat_market_covariance_has_floor_and_finite_full_investment():
    e = env(market_panel(flat=True))
    assert e.estimated_risk(np.ones(3)/3) == .01
    np.testing.assert_allclose(e.risk_target(np.ones(3)), [0, 1/3, 1/3, 1/3])
    assert np.isfinite(e.reset()[0]).all()


def test_risk_diagnostics_describe_the_proposal_at_decision_close_not_holdings():
    e = env()
    target = e.risk_target([1, 1, 1], .1)
    sigma = e.estimated_risk(np.ones(3)/3)
    first = e.step_weights(target)[-1]
    assert first["rebalance_executed"]
    assert first["requested_exposure"] == pytest.approx(1-target[0])
    assert first["estimated_risky_vol"] == pytest.approx(sigma)
    assert first["requested_target_vol"] == pytest.approx(.1)
    # The following cash proposal is observed but ignored during the hold phase.
    ignored = e.step_weights([1, 0, 0, 0])[-1]
    assert not ignored["rebalance_executed"]
    assert ignored["requested_exposure"] == 0
    assert ignored["requested_risky_mix"] == [0, 0, 0]
    assert ignored["requested_target_vol"] == 0
    assert sum(ignored["target_weights"][1:]) > 0
    hold = e.hold()[-1]
    assert hold["requested_exposure"] is None
    assert hold["requested_target_vol"] is None
    assert e.history[-1]["requested_exposure"] is None


def test_risk_target_is_pure_and_public_covariance_cannot_mutate_future_actions():
    e = env()
    before = (e.t, e.value, e.cash, e.receivable, e._peak_value, len(e.history))
    target = e.risk_target([1, 2, 3])
    e.risk_target([3, 2, 1], .2)
    copy = e.estimated_covariance(e.t, 20)
    copy[:] = 999
    np.testing.assert_array_equal(target, e.risk_target([1, 2, 3]))
    assert before == (e.t, e.value, e.cash, e.receivable, e._peak_value, len(e.history))


@pytest.mark.parametrize("index", [119, 280, -1, 120.0, True])
def test_features_require_a_valid_history_index(index):
    with pytest.raises(ValueError):
        env().market_features(index)


def test_volume_surprise_uses_previous_window_and_split_compatible_share_units():
    original = market_panel(flat=True)
    changed = replace(original, **{name: getattr(original, name).copy()
                                  for name in ("open", "close", "high", "low", "volume")})
    for name in ("open", "close", "high", "low"):
        getattr(changed, name)[125:, 0] /= 2
    changed.volume[125:, 0] *= 2
    actions = CorporateActions.empty(changed)
    actions.split_ratio[125, 0] = 2
    split_env = env(changed, actions=actions)
    unchanged_env = env(original)
    np.testing.assert_allclose(split_env.market_features(130), unchanged_env.market_features(130), atol=1e-12)
    changed.volume[130, 1] *= 3
    surprise_env = env(changed, actions=actions)
    names = surprise_env.market_feature_names()
    index = names.index("B:volume_surprise_60")
    assert surprise_env.market_features(130)[index] == pytest.approx(np.log(3))


def test_future_ohlcv_dividends_and_splits_cannot_change_past_features_actions_or_account():
    panel = market_panel()
    future = replace(panel, **{name: getattr(panel, name).copy()
                               for name in ("open", "close", "high", "low", "volume")})
    for name in ("open", "close", "high", "low", "volume"):
        getattr(future, name)[161:] *= 3
    actions = CorporateActions.empty(future)
    actions.cash_per_share[180, 1] = 5
    actions.split_ratio[190, 0] = 1.25
    actions = replace(actions, dividends=(Dividend(1, 180, 190, 5.),))
    first, second = env(panel, end=160), env(future, end=160, actions=actions)
    for t in range(120, 161):
        np.testing.assert_array_equal(first.market_features(t), second.market_features(t))
        np.testing.assert_array_equal(first.estimated_covariance(t, 60), second.estimated_covariance(t, 60))
    first.reset(seed=7); second.reset(seed=7)
    for _ in range(40):
        action = np.array([.3, -.7, .2, .1])
        np.testing.assert_array_equal(first.action_to_weights(action), second.action_to_weights(action))
        a, b = first.step(action), second.step(action)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]


def test_account_features_show_current_drawdown_and_unspendable_receivable():
    panel = market_panel(flat=True)
    for name in ("open", "close", "high", "low"):
        getattr(panel, name)[122:, 0] *= .9
        getattr(panel, name)[123:, 1] *= .8
    actions = CorporateActions.empty(panel)
    actions.cash_per_share[122, 0] = 10
    actions = replace(actions, dividends=(Dividend(0, 122, 125, 10.),))
    e = env(panel, actions=actions, cost_bps=0, risk_aversion=0)
    e.step_weights([.1, .3, .3, .3])
    obs, _, _, _, info = e.hold()
    assert obs[-3:] == pytest.approx([0, .1, .03])
    obs, _, _, _, info = e.hold()
    assert obs[-3:] == pytest.approx([.06, .1/.94, .03/.94])
    e.hold()
    obs, _, _, _, info = e.hold()
    assert obs[-3:] == pytest.approx([.06, .13/.94, 0])
    obs, _ = e.reset()
    assert obs[-3:] == pytest.approx([0, 1, 0])


@pytest.mark.parametrize("kwargs", [{"enhanced_features":1}, {"risk_control":"yes"}])
def test_ablation_flags_require_booleans(kwargs):
    with pytest.raises(ValueError):
        env(**kwargs)


@pytest.mark.parametrize("action", [[0], [0,0,np.nan,0], [0,2,0,0]])
def test_invalid_risk_actions_cannot_advance_state(action):
    e = env()
    with pytest.raises(ValueError):
        e.step(action)
    assert e.t == 120


@pytest.mark.parametrize("budget", [-1, 0, np.nan, np.inf])
def test_invalid_risk_budget_is_rejected(budget):
    with pytest.raises(ValueError):
        env().risk_target(np.ones(3), budget)


@pytest.mark.parametrize("enhanced,risk", [(False,False), (False,True), (True,False), (True,True)])
def test_gym_contract_including_seeded_random_fragments(enhanced, risk):
    e = env(enhanced_features=enhanced, risk_control=risk, training=True, episode_length=20)
    check_env(e, skip_render_check=True)
