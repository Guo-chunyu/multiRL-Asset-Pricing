"""Causal features, bounded allocations, and scheduled self-financing trades."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from rlasset.data import MarketPanel


def panel_fixture(days=400, assets=4):
    close = np.full((days, assets), 100.0)
    return MarketPanel(
        dates=pd.bdate_range("2019-01-01", periods=days),
        tickers=tuple(f"ETF{i}" for i in range(assets)),
        open=close.copy(), close=close,
        high=close * 1.01, low=close * .99,
        volume=np.full_like(close, 100000.),
    )


def make_env(panel=None, **kwargs):
    from rlasset.robust_env import RobustPortfolioEnv
    panel = panel_fixture() if panel is None else panel
    options = dict(start=120, end=len(panel.dates) - 1)
    options.update(kwargs)
    return RobustPortfolioEnv(panel, **options)


def test_flat_market_has_zero_factors_and_observable_rebalance_phase():
    # Catches uncentred prices and a hidden time-to-next-rebalance state.
    env = make_env()
    observation, _ = env.reset(seed=7)
    assert observation.shape == (38,)
    assert observation.dtype == np.float32
    np.testing.assert_array_equal(observation[:32], 0.)
    np.testing.assert_array_equal(observation[32:37], [1., 0., 0., 0., 0.])
    assert observation[-1] == 0
    observation, *_ = env.step(np.zeros(5))
    assert observation[-1] == pytest.approx(.25)
    assert env.observation_space.contains(observation)


def test_future_prices_do_not_change_observation_target_or_first_transition():
    # Catches full-panel normalization and future rows in factors/covariance.
    panel = panel_fixture()
    other = replace(panel, **{name: getattr(panel, name).copy()
                              for name in ("open", "close", "high", "low")})
    for name in ("open", "close", "high", "low"):
        getattr(other, name)[122:] *= np.linspace(2, 50, 278)[:, None]
    first, second = make_env(panel), make_env(other)
    obs1, _ = first.reset(seed=17)
    obs2, _ = second.reset(seed=17)
    np.testing.assert_array_equal(obs1, obs2)
    action1, action2 = np.tanh(obs1[:5]), np.tanh(obs2[:5])
    np.testing.assert_array_equal(first.action_to_weights(action1), second.action_to_weights(action2))
    out1, out2 = first.step(action1), second.step(action2)
    np.testing.assert_array_equal(out1[0], out2[0])
    assert out1[1:] == out2[1:]


def test_factors_detect_recent_price_change_without_future_lookahead():
    # A known jump at the decision close must enter all momentum horizons.
    panel = panel_fixture()
    panel.close[120, 0] = 110.
    panel.high[120, 0] = 111.
    env = make_env(panel)
    obs, _ = env.reset()
    assert obs[0] == pytest.approx(10 * np.log(1.1))
    assert obs[4] == pytest.approx(10 * np.log(1.1) / np.sqrt(120))
    assert obs[5] > 0 and obs[6] > 0 and obs[7] > 0
    np.testing.assert_array_equal(obs[8:32], 0)


@pytest.mark.parametrize("cash_action, expected_cash", [(-1., 0.), (0., .1), (1., .2)])
def test_equal_logits_respect_explicit_equity_exposure(cash_action, expected_cash):
    env = make_env()
    weights = env.action_to_weights([cash_action, 0, 0, 0, 0])
    np.testing.assert_allclose(weights, [expected_cash, *([(1 - expected_cash) / 4] * 4)])


def test_capped_action_redistributes_to_other_assets_instead_of_extra_cash():
    # Extreme logits initially allocate >.5 to asset 1; cap excess must stay risky.
    env = make_env()
    weights = env.action_to_weights([-1., 1., -1., -1., -1.])
    np.testing.assert_allclose(weights, [0., .5, 1 / 6, 1 / 6, 1 / 6])
    for action in np.random.default_rng(41).uniform(-1, 1, size=(100, 5)):
        weights = env.action_to_weights(action)
        assert 0 <= weights[0] <= .2
        assert np.all(weights[1:] <= .5 + 1e-12)
        assert np.all(weights >= 0)
        assert weights.sum() == pytest.approx(1.)


def test_two_asset_zero_cash_target_is_feasible_under_each_asset_cap():
    env = make_env(panel_fixture(assets=2))
    np.testing.assert_allclose(env.action_to_weights([-1, 1, -1]), [0, .5, .5])


def test_four_intervening_days_hold_shares_even_when_new_actions_disagree():
    # Catches rebalancing every day or reusing target weights during hold days.
    panel = panel_fixture()
    for day in range(122, 128):
        panel.open[day, 0] = panel.close[day, 0] = 100. + (day - 121) * 10
        panel.high[day, 0] = panel.close[day, 0] * 1.01
        panel.low[day, 0] = panel.close[day, 0] * .99
    env = make_env(panel, cost_bps=100, risk_aversion=0)
    env.reset()
    _, _, _, _, first = env.step_weights([0, .25, .25, .25, .25])
    assert first["portfolio_value"] == pytest.approx(1 / 1.01)
    for offset in range(1, 5):
        _, _, _, _, info = env.step_weights([1, 0, 0, 0, 0])
        assert info["turnover"] == 0 and info["cost_fraction"] == 0
        assert not info["rebalance_executed"]
        assert info["portfolio_value"] == pytest.approx((1 + .025 * offset) / 1.01)
    _, _, _, _, sixth = env.step_weights([1, 0, 0, 0, 0])
    assert sixth["rebalance_executed"]
    assert sixth["cost_fraction"] > 0
    np.testing.assert_allclose(sixth["weights"], [1, 0, 0, 0, 0])


def test_explicit_hold_skips_even_a_scheduled_rebalance():
    env = make_env()
    env.reset()
    for _ in range(6):
        _, _, _, _, info = env.hold()
        assert info["portfolio_value"] == 1
        assert info["turnover"] == 0


def test_ex_dividend_drop_is_not_a_false_momentum_signal_or_lost_cash():
    # A 10/share ex-price drop is offset by the existing holder's receivable;
    # features use the causal signal index while shares retain raw-price value.
    from rlasset.corporate_actions import CorporateActions, Dividend
    panel = panel_fixture()
    for name in ("open", "close", "high", "low"):
        getattr(panel, name)[122:, 0] *= .9
    actions = CorporateActions.empty(panel)
    actions.cash_per_share[122, 0] = 10.
    actions = replace(actions, dividends=(Dividend(0, 122, 125, 10.),))
    env = make_env(panel, actions=actions, cost_bps=0, risk_aversion=0)
    env.reset()
    env.step(np.zeros(5))
    observation, reward, _, _, ex = env.step(np.zeros(5))
    np.testing.assert_allclose(observation[:8], 0., atol=1e-12)
    assert ex['portfolio_value'] == pytest.approx(1.)
    assert ex['receivable'] == pytest.approx(.0225)
    assert ex['cash'] == pytest.approx(.1)
    assert reward == pytest.approx(0.)
    for _ in range(3):
        _, _, _, _, paid = env.step(np.zeros(5))
    assert paid['dividend_paid'] == pytest.approx(.0225)
    assert paid['receivable'] == 0
    assert paid['cash'] == pytest.approx(.1225)
    assert paid['portfolio_value'] == pytest.approx(1.)


def test_random_training_starts_remain_in_bounds_and_reproduce_seed():
    # Catches reseeding on every reset and sampling a future validation date.
    env = make_env(start=140, end=350, training=True, episode_length=126)
    sequences = []
    for _ in range(2):
        env.reset(seed=29)
        starts = [env.t]
        for _ in range(15):
            env.reset()
            starts.append(env.t)
            assert 140 <= env.t <= 224
            assert env.end - env.t == 126
        sequences.append(starts)
    assert sequences[0] == sequences[1]
    assert len(set(sequences[0])) > 1


def test_training_fragment_truncates_but_full_evaluation_terminates():
    # A training fragment must bootstrap rather than learn an artificial terminal value.
    training = make_env(training=True, episode_length=3)
    training.reset(seed=7)
    for step in range(3):
        obs, _, terminal, truncated, _ = training.step(np.zeros(5))
        assert terminal is False
        assert truncated is (step == 2)
    assert training.observation_space.contains(obs)
    with pytest.raises(RuntimeError):
        training.step(np.zeros(5))
    evaluation = make_env(start=120, end=123)
    evaluation.reset(seed=7)
    for step in range(3):
        _, _, terminal, truncated, _ = evaluation.step(np.zeros(5))
        assert terminal is (step == 2)
        assert truncated is False
    evaluation.reset(seed=99)
    assert evaluation.t == 120


@pytest.mark.parametrize("action", [[0], [0, 0, 0, 0, 2], [0, 0, np.nan, 0, 0]])
def test_invalid_action_is_rejected_before_any_market_step(action):
    env = make_env()
    env.reset()
    with pytest.raises(ValueError):
        env.step(action)
    assert env.t == 120


def test_invalid_direct_target_is_rejected_on_a_hold_day_too():
    env = make_env()
    env.reset()
    env.step(np.zeros(5))
    with pytest.raises(ValueError):
        env.step_weights([0, 1, 0, 0, 0])
    assert env.t == 121


@pytest.mark.parametrize("kwargs", [
    {"start": 119}, {"rebalance_every": 0}, {"rebalance_every": 1.5},
    {"episode_length": 0}, {"training": True, "end": 200, "episode_length": 126},
])
def test_invalid_history_or_fragment_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        make_env(**kwargs)


def test_one_asset_cannot_satisfy_the_equity_exposure_and_asset_cap_together():
    with pytest.raises(ValueError):
        make_env(panel_fixture(assets=1))


def test_environment_seed_and_step_contract():
    check_env(make_env(), skip_render_check=True)
    check_env(make_env(training=True, episode_length=20), skip_render_check=True)


def test_capped_risky_weights_redistributes_without_changing_exposure():
    from rlasset.robust_env import capped_risky_weights
    # 9:1:1 with .9 exposure starts at .736:.082:.082; the first cap
    # leaves .4 total for the two equal-score uncapped positions.
    np.testing.assert_allclose(capped_risky_weights([9, 1, 1], .9, .5), [.5, .2, .2])
    np.testing.assert_allclose(capped_risky_weights([1, 1], 1, .5), [.5, .5])
    np.testing.assert_array_equal(capped_risky_weights([1, 2], 0, .5), [0, 0])


@pytest.mark.parametrize("scores, exposure, cap", [
    ([1], .9, .5), ([0, 0], .9, .5), ([np.nan, 1], .9, .5),
    ([1, 2], 1.1, .5), ([1, 2], .9, 0),
])
def test_capped_risky_weights_rejects_infeasible_inputs(scores, exposure, cap):
    from rlasset.robust_env import capped_risky_weights
    with pytest.raises(ValueError):
        capped_risky_weights(scores, exposure, cap)
