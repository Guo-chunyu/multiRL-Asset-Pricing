import numpy as np
import pytest


def make_env():
    from rlasset.data import MarketPanel, synthetic_frame
    from rlasset.env import PortfolioEnv
    return PortfolioEnv(MarketPanel.from_frame(synthetic_frame(70, 2, 9)), start=20, end=40, lookback=20)


def test_selector_only_observes_realized_performance_and_executes_selected_policy():
    from rlasset.meta import Candidate, SelectorEnv
    candidates = [Candidate("cash", lambda env, obs: np.array([1., 0., 0.])), Candidate("equal", lambda env, obs: np.array([0., .5, .5]))]
    meta = SelectorEnv(make_env, candidates)
    obs, _ = meta.reset()
    np.testing.assert_array_equal(obs["performance"], 0)
    next_obs, _, done, _, info = meta.step(1)
    assert info["selected_policy"] == "equal"
    np.testing.assert_allclose(info["target_weights"], [0., .5, .5])
    assert next_obs["performance"].shape == (6,)
    assert np.isfinite(next_obs["performance"]).all()
    assert next_obs["performance"][0] == 0  # shadow cash earns zero
    assert not done
    _, _ = meta.reset()
    assert all(not returns for returns in meta.performance_history)
    with pytest.raises(ValueError):
        meta.step(9)


def test_meta_does_not_advance_or_update_candidates_during_observation():
    from rlasset.meta import Candidate, SelectorEnv
    calls = []
    def cash(env, obs):
        calls.append(env.t)
        return np.array([1., 0., 0.])
    meta = SelectorEnv(make_env, [Candidate("cash", cash)])
    meta.reset()
    assert calls == []
    meta.step(0)
    assert calls == [20, 20]  # actual portfolio and independently held shadow


def test_baseline_buy_hold_requests_no_further_rebalance():
    from rlasset.meta import baseline_candidates
    env = make_env()
    obs, _ = env.reset()
    baseline = baseline_candidates()["buy_hold"]
    env.step_weights(baseline.weights(env, obs))
    assert baseline.weights(env, obs) is None


def test_future_prices_cannot_change_earlier_selector_features_or_choices():
    # Catches precomputing performance from later returns or future normalization.
    from dataclasses import replace

    from rlasset.data import MarketPanel, synthetic_frame
    from rlasset.env import PortfolioEnv
    from rlasset.meta import SelectorEnv, baseline_candidates

    panel = MarketPanel.from_frame(synthetic_frame(70, 2, 9))
    changed = replace(panel, **{
        field: getattr(panel, field).copy()
        for field in ("open", "close", "high", "low")
    })
    for field in ("open", "close", "high", "low"):
        getattr(changed, field)[23:] *= 100
    candidates = baseline_candidates()
    pool = [candidates["momentum"], candidates["reversal"], candidates["cash"]]
    meta1 = SelectorEnv(lambda: PortfolioEnv(panel, 20, 40, lookback=20), pool)
    meta2 = SelectorEnv(lambda: PortfolioEnv(changed, 20, 40, lookback=20), pool)
    obs1, _ = meta1.reset(seed=7)
    obs2, _ = meta2.reset(seed=7)
    for key in obs1:
        np.testing.assert_array_equal(obs1[key], obs2[key])
    for action in (0, 1):
        out1 = meta1.step(action)
        out2 = meta2.step(action)
        for key in out1[0]:
            np.testing.assert_array_equal(out1[0][key], out2[0][key])
        assert out1[1:] == out2[1:]
        assert meta1.performance_history == meta2.performance_history


def test_selected_policy_receives_actual_holdings_while_shadow_uses_its_own():
    # Catches running the real action with a shadow's unrelated portfolio state.
    import pandas as pd

    from rlasset.data import MarketPanel
    from rlasset.env import PortfolioEnv
    from rlasset.meta import Candidate, SelectorEnv

    prices = np.full((6, 2), 100.0)
    panel = MarketPanel(
        dates=pd.bdate_range("2020-01-01", periods=6), tickers=("A", "B"),
        open=prices.copy(), close=prices.copy(), high=prices.copy(),
        low=prices.copy(), volume=np.full((6, 2), 10000.),
    )
    def toggle(env, obs):
        # Buy when in cash, otherwise sell; every decision is causal.
        return np.array([0., .5, .5]) if obs["weights"][0] > .9 else np.array([1., 0., 0.])
    pool = [
        Candidate("cash", lambda env, obs: np.array([1., 0., 0.])),
        Candidate("toggle", toggle),
    ]
    meta = SelectorEnv(lambda: PortfolioEnv(panel, 2, 5, lookback=2, cost_bps=100), pool)
    meta.reset()
    next_obs, _, _, _, first = meta.step(0)
    # Actual ledger stays cash; toggle shadow has bought its own risky assets.
    np.testing.assert_allclose(first["weights"], [1., 0., 0.])
    np.testing.assert_allclose(meta.shadows[1].weights, [0., .5, .5])
    assert next_obs["performance"][3] == pytest.approx(-100 / 101)
    _, _, _, _, second = meta.step(1)
    # Selecting toggle should buy in the actual account, but sell in its shadow.
    np.testing.assert_allclose(second["target_weights"], [0., .5, .5])
    np.testing.assert_allclose(meta.shadows[1].weights, [1., 0., 0.])
    assert second["net_return"] == pytest.approx(-1 / 101)
    assert meta.performance_history[1][-1] == pytest.approx(-.01)


def test_shadow_performance_window_drops_old_returns_and_reset_clears_it():
    # Catches accidental expanding statistics after a configured rolling window.
    from rlasset.meta import Candidate, SelectorEnv

    candidates = [Candidate("equal", lambda env, obs: np.array([0., .5, .5]))]
    meta = SelectorEnv(make_env, candidates, performance_window=1)
    meta.reset()
    meta.step(0)
    obs, _, _, _, _ = meta.step(0)
    last_return = meta.shadows[0].history[-1]["net_return"]
    np.testing.assert_allclose(obs["performance"],
                               [last_return * 100, 0., np.log1p(last_return) * 10], rtol=1e-6)
    reset_obs, _ = meta.reset()
    np.testing.assert_array_equal(reset_obs["performance"], np.zeros(3))


@pytest.mark.parametrize("scores", [
    # Real ETF reversal scores at 2022-06-01, t=96, lookback=20.
    [0.5255809426307678, 0.9974937438964844, 0.8367301821708679, 0.5293209552764893],
    # Real ETF momentum scores at 2023-07-07, t=364, lookback=20.
    [1.2502527236938477, 0.9737067818641663, 1.0889452695846558, 1.0696032047271729],
])
def test_capped_weights_cannot_create_negative_cash_from_normalization_roundoff(scores):
    # These real inputs previously produced cash=-2.220446049250313e-16,
    # crashing the independently simulated shadow inside SelectorEnv.step.
    from rlasset.data import MarketPanel, synthetic_frame
    from rlasset.env import PortfolioEnv
    from rlasset.meta import capped_weights

    target = capped_weights(scores, .5)
    assert (target >= 0).all()
    assert target.sum() == pytest.approx(1, abs=1e-15)
    assert (target[1:] <= .5).all()
    panel = MarketPanel.from_frame(synthetic_frame(24, 4, 3))
    env = PortfolioEnv(panel, 20, 23, lookback=20)
    _, reward, _, _, info = env.step_weights(target)
    assert np.isfinite(reward)
    assert info["portfolio_value"] > 0
