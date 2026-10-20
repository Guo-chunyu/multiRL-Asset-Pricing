"""Accounting and causality checks with independently hand-computed paths."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from rlasset.data import MarketPanel


def panel_from_prices(close, open_prices=None):
    close = np.asarray(close, dtype=float)
    if close.ndim == 1:
        close = close[:, None]
    opening = close.copy() if open_prices is None else np.asarray(open_prices, dtype=float).reshape(close.shape)
    return MarketPanel(
        dates=pd.bdate_range("2020-01-01", periods=len(close)),
        tickers=tuple(f"ETF{i}" for i in range(close.shape[1])),
        open=opening,
        close=close,
        high=np.maximum(opening, close) * 1.01,
        low=np.minimum(opening, close) * 0.99,
        volume=np.full(close.shape, 100_000.0),
    )


def make_env(panel=None, **kwargs):
    from rlasset.env import PortfolioEnv

    if panel is None:
        panel = panel_from_prices([100, 100, 100, 100, 100, 100])
    config = dict(start=2, end=len(panel.dates) - 1, lookback=2, max_weight=1.0)
    config.update(kwargs)
    return PortfolioEnv(panel, **config)


def test_observation_uses_only_the_window_ending_at_decision_close():
    # Catches future prices in windows and normalization by a future close.
    panel = panel_from_prices([[80, 60], [90, 40], [100, 50], [150, 90]])
    env = make_env(panel)
    obs, info = env.reset(seed=7)
    assert obs["prices"].shape == (3, 2, 2)
    assert obs["prices"].dtype == np.float32
    np.testing.assert_allclose(obs["prices"][0], [[0.9, 1.0], [0.8, 1.0]])
    np.testing.assert_allclose(obs["prices"][1], [[0.909, 1.01], [0.808, 1.01]])
    np.testing.assert_allclose(obs["weights"], [1.0, 0.0, 0.0])
    assert info["portfolio_value"] == 1.0
    assert len(env.history) == 1


def test_new_position_is_bought_at_next_open_after_the_gap():
    # Cash cannot earn the 20% gap before its order executes at next open.
    panel = panel_from_prices([100, 100, 100, 132], [100, 100, 100, 120])
    env = make_env(panel, cost_bps=100, reward_scale=1)
    env.reset()
    obs, reward, terminated, truncated, info = env.step_weights([0.5, 0.5])
    # Starting at 1, a half-risky allocation costs .01 * .5 * mu.
    # mu = 1/1.005; intraday growth = .5 + .5 * 1.1 = 1.05.
    assert info["portfolio_value"] == pytest.approx(1.044776119402985)
    assert info["turnover"] == pytest.approx(0.4975124378109453)
    assert info["cost_fraction"] == pytest.approx(0.004975124378109453)
    assert info["net_return"] == pytest.approx(0.044776119402985)
    assert reward == pytest.approx(np.log(1.044776119402985))
    np.testing.assert_allclose(obs["weights"], [10 / 21, 11 / 21])
    assert info["decision_date"] == panel.dates[2].isoformat()
    assert info["date"] == panel.dates[3].isoformat()
    assert terminated and not truncated


def test_old_position_earns_overnight_gap_before_it_is_sold():
    panel = panel_from_prices([100, 100, 100, 100, 198], [100, 100, 100, 100, 200])
    env = make_env(panel, cost_bps=100)
    env.reset()
    env.step_weights([0, 1])
    obs, _, done, _, info = env.step_weights([1, 0])
    # Buy at t3 costs 1/1.01, double overnight, sell at t4 for .99.
    assert info["portfolio_value"] == pytest.approx(1.9603960396039604)
    assert info["net_return"] == pytest.approx(0.98)
    assert info["turnover"] == pytest.approx(1.0)
    assert info["cost_fraction"] == pytest.approx(0.01)
    np.testing.assert_array_equal(obs["weights"], [1.0, 0.0])
    assert done


def test_switching_between_two_assets_charges_both_the_sale_and_purchase():
    panel = panel_from_prices(np.full((5, 2), 100.0))
    env = make_env(panel, cost_bps=100)
    env.reset()
    env.step_weights([0, 1, 0])
    _, _, _, _, info = env.step_weights([0, 0, 1])
    # Post-cost wealth mu solves mu=1-.01*(1+mu), so mu=99/101.
    assert info["net_return"] == pytest.approx(-2 / 101)
    assert info["turnover"] == pytest.approx(200 / 101)
    assert info["cost_fraction"] == pytest.approx(2 / 101)


def test_rebalancing_uses_drifted_holdings_and_not_the_last_target():
    panel = panel_from_prices([100, 100, 100, 200, 200], [100, 100, 100, 100, 200])
    env = make_env(panel, cost_bps=100, max_weight=0.5)
    env.reset()
    obs, _, _, _, first = env.step_weights([0.5, 0.5])
    np.testing.assert_allclose(obs["weights"], [1 / 3, 2 / 3])
    # Holdings may drift beyond the cap; only target weights are capped.
    _, _, _, _, second = env.step_weights([0.5, 0.5])
    # Sell 2/3 - mu/2: mu=(1-.01*2/3)/(1-.01/2)=596/597.
    assert second["portfolio_value"] / first["portfolio_value"] == pytest.approx(596 / 597)
    assert second["turnover"] == pytest.approx(100 / 597)


def test_unchanged_holdings_have_zero_turnover_and_no_repeated_fee():
    env = make_env(cost_bps=100)
    env.reset()
    env.step_weights([0.5, 0.5])
    _, _, _, _, info = env.step_weights([0.5, 0.5])
    assert info["turnover"] == pytest.approx(0, abs=1e-12)
    assert info["cost_fraction"] == pytest.approx(0, abs=1e-12)
    assert info["net_return"] == pytest.approx(0, abs=1e-12)


def test_hold_keeps_shares_through_the_gap_without_rebalancing_or_fees():
    # A hold order must not re-establish yesterday's weight at today's open.
    panel = panel_from_prices([100, 100, 100, 100, 300], [100, 100, 100, 100, 200])
    env = make_env(panel, cost_bps=100, max_weight=0.5)
    env.reset()
    env.step_weights([0.5, 0.5])
    obs, _, _, _, info = env.hold()
    # Half remains cash, half triples: 2/(1+.01*.5).
    assert info["portfolio_value"] == pytest.approx(1.9900497512437811)
    assert info["turnover"] == 0.0
    assert info["cost_fraction"] == 0.0
    np.testing.assert_allclose(obs["weights"], [0.25, 0.75])


def test_cash_does_not_pay_fees_or_earn_risky_returns():
    env = make_env(panel_from_prices([100, 90, 100, 300, 30]), cost_bps=200)
    env.reset()
    for _ in range(2):
        _, reward, _, _, info = env.step_weights([1, 0])
        assert info["portfolio_value"] == 1.0
        assert info["turnover"] == 0.0
        assert reward == 0.0


def test_action_mapping_caps_risky_assets_and_sends_excess_to_cash():
    panel = panel_from_prices(np.full((5, 2), 100.0))
    env = make_env(panel, max_weight=0.3)
    np.testing.assert_allclose(env.action_to_weights([0, 0, 0]), [0.4, 0.3, 0.3])
    weights = env.action_to_weights([-1, 1, -1])
    assert weights[1] == pytest.approx(0.3)
    assert weights.sum() == pytest.approx(1)
    assert weights[0] > 0.69
    env.reset()
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(info["target_weights"], [0.4, 0.3, 0.3])


@pytest.mark.parametrize("action", [[np.nan, 0], [0, np.inf], [0], [2, 0]])
def test_malformed_actions_are_rejected_without_advancing_time(action):
    env = make_env()
    env.reset()
    with pytest.raises(ValueError):
        env.step(action)
    assert env.t == 2
    assert env.value == 1


@pytest.mark.parametrize("weights", [[-0.1, 1.1], [0.2, 0.2], [0.1, 0.9], [np.nan, 0], [1, 0, 0]])
def test_direct_targets_must_be_feasible_before_trading(weights):
    env = make_env(max_weight=0.5)
    env.reset()
    with pytest.raises(ValueError):
        env.step_weights(weights)
    assert env.t == 2


def test_future_data_changes_cannot_change_an_earlier_transition():
    panel = panel_from_prices([100, 90, 100, 105, 200, 300], [100, 90, 100, 100, 200, 300])
    changed = replace(panel, **{name: getattr(panel, name).copy() for name in ("open", "close", "high", "low")})
    for name in ("open", "close", "high", "low"):
        getattr(changed, name)[4:] *= 50
    env1 = make_env(panel, risk_aversion=10)
    env2 = make_env(changed, risk_aversion=10)
    obs1, _ = env1.reset(seed=99)
    obs2, _ = env2.reset(seed=99)
    np.testing.assert_array_equal(obs1["prices"], obs2["prices"])
    out1 = env1.step_weights([0.4, 0.6])
    out2 = env2.step_weights([0.4, 0.6])
    assert out1[1] == out2[1]
    assert out1[4] == out2[4]
    np.testing.assert_array_equal(out1[0]["prices"], out2[0]["prices"])


def test_risk_penalty_uses_returns_known_at_decision_time():
    # Historical returns are -.1 and +.1, sample variance=.02.
    # The next close return is 0 and must not be included in this variance.
    panel = panel_from_prices([100, 90, 99, 99, 990], [100, 90, 99, 99, 990])
    env = make_env(panel, cost_bps=0, risk_aversion=2, reward_scale=100)
    env.reset()
    _, reward, _, _, info = env.step_weights([0.5, 0.5])
    assert info["risk_penalty"] == pytest.approx(0.01)
    assert reward == pytest.approx(-1.0)
    assert info["portfolio_value"] == 1.0


def test_single_return_lookback_has_finite_zero_sample_risk():
    env = make_env(lookback=1, risk_aversion=10)
    env.reset()
    _, reward, _, _, info = env.step_weights([0.5, 0.5])
    assert np.isfinite(reward)
    assert info["risk_penalty"] == 0.0


def test_termination_and_reset_cannot_replay_the_last_return():
    env = make_env(end=3, cost_bps=0)
    env.reset()
    assert env.step_weights([0.5, 0.5])[2]
    assert len(env.history) == 2
    with pytest.raises(RuntimeError):
        env.step_weights([0.5, 0.5])
    obs, info = env.reset(seed=22)
    assert env.t == 2 and len(env.history) == 1
    assert info["portfolio_value"] == 1
    np.testing.assert_array_equal(obs["weights"], [1, 0])


def test_seeded_rollouts_and_gym_contract_are_reproducible():
    env = make_env(cost_bps=10)
    check_env(env, skip_render_check=True)
    def rollout():
        env.reset(seed=45)
        env.action_space.seed(45)
        rewards = []
        for _ in range(3):
            rewards.append(env.step(env.action_space.sample())[1])
        return rewards, env.history
    first, first_history = rollout()
    second, second_history = rollout()
    np.testing.assert_array_equal(first, second)
    assert first_history == second_history


@pytest.mark.parametrize("kwargs", [
    {"start": 1}, {"end": 2}, {"end": 6}, {"lookback": 0},
    {"cost_bps": -1}, {"cost_bps": 10000}, {"max_weight": 0},
    {"max_weight": 1.1}, {"risk_aversion": -1}, {"reward_scale": 0},
])
def test_invalid_environment_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        make_env(**kwargs)
