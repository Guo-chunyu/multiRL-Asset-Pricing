"""Hand-accounted dividend/split cases; no provider data or network required."""
import numpy as np
import pandas as pd
import pytest

from rlasset.data import MarketPanel


def make_panel(prices=None):
    prices = prices or [10., 10., 10., 9., 9., 9., 9.]
    dates = pd.bdate_range("2024-01-01", periods=len(prices))
    return MarketPanel.from_frame(pd.DataFrame({
        "date": dates, "tic": "ETF", "open": prices, "close": prices,
        "high": prices, "low": prices, "volume": 10000.,
    }))


def actions_from_files(tmp_path, panel, cash=None, splits=None):
    from rlasset.corporate_actions import CorporateActions
    cash_path = tmp_path / "cash.csv"
    splits_path = tmp_path / "splits.csv"
    pd.DataFrame(cash or [], columns=["ticker", "ex_date", "pay_date", "cash_per_share"]).to_csv(cash_path, index=False)
    pd.DataFrame(splits or [], columns=["ticker", "effective_date", "ratio"]).to_csv(splits_path, index=False)
    return CorporateActions.from_csv(panel, cash_path, splits_path)


def test_ex_date_owner_receives_dividend_and_receivable_cannot_be_spent(tmp_path):
    from rlasset.corporate_actions import CorporateActionPortfolioEnv
    panel = make_panel()
    actions = actions_from_files(tmp_path, panel, [("ETF", "2024-01-04", "2024-01-08", 1.)])
    env = CorporateActionPortfolioEnv(panel, start=1, end=6, lookback=1, cost_bps=0, max_weight=1, actions=actions)
    env.step_weights([0, 1])  # Buy .1 shares on January 3 before ex-date.
    _, _, _, _, info = env.step_weights([0, 1])
    assert info["dividend_entitlement"] == pytest.approx(.1)
    assert info["receivable"] == pytest.approx(.1)
    assert info["cash"] == pytest.approx(0)
    assert info["shares"] == pytest.approx([.1])
    assert info["portfolio_value"] == pytest.approx(1)
    assert info["weights"] == pytest.approx([.1, .9])
    assert info["net_return"] == pytest.approx(0)
    env.hold()  # January 5, still unpaid.
    _, _, _, _, paid = env.step_weights([0, 1])  # Pay-date January 8; cash now investable.
    assert paid["dividend_paid"] == pytest.approx(.1)
    assert paid["receivable"] == 0
    assert paid["shares"] == pytest.approx([1/9])
    assert paid["portfolio_value"] == pytest.approx(1)


def test_new_buyer_on_ex_date_has_no_entitlement(tmp_path):
    from rlasset.corporate_actions import CorporateActionPortfolioEnv
    panel = make_panel()
    actions = actions_from_files(tmp_path, panel, [("ETF", "2024-01-04", "2024-01-08", 1.)])
    env = CorporateActionPortfolioEnv(panel, start=2, end=6, lookback=1, cost_bps=0, max_weight=1, actions=actions)
    _, _, _, _, info = env.step_weights([0, 1])
    assert info["dividend_entitlement"] == 0
    assert info["receivable"] == 0
    assert info["shares"] == pytest.approx([1/9])


def test_ex_date_seller_keeps_entitlement_and_weekend_pay_date_rolls_forward(tmp_path):
    from rlasset.corporate_actions import CorporateActionPortfolioEnv
    panel = make_panel()
    actions = actions_from_files(tmp_path, panel, [("ETF", "2024-01-04", "2024-01-06", 1.)])
    env = CorporateActionPortfolioEnv(panel, start=1, end=6, lookback=1, cost_bps=0, max_weight=1, actions=actions)
    env.step_weights([0, 1])
    info = env.step_weights([1, 0])[-1]
    assert info["cash"] == pytest.approx(.9)
    assert info["receivable"] == pytest.approx(.1)
    assert info["shares"] == [0]
    assert env.hold()[-1]["dividend_paid"] == 0
    paid = env.hold()[-1]
    assert paid["date"].startswith("2024-01-08")
    assert paid["dividend_paid"] == pytest.approx(.1)
    assert paid["cash"] == pytest.approx(1)


def test_split_preserves_value_and_held_shares_with_no_turnover(tmp_path):
    from rlasset.corporate_actions import CorporateActionPortfolioEnv
    panel = make_panel([10., 10., 10., 5., 5., 5., 5.])
    actions = actions_from_files(tmp_path, panel, splits=[("ETF", "2024-01-04", 2.)])
    env = CorporateActionPortfolioEnv(panel, start=1, end=6, lookback=1, cost_bps=0, max_weight=1, actions=actions)
    env.step_weights([0, 1])
    info = env.hold()[-1]
    assert info["shares"] == pytest.approx([.2])
    assert info["split_ratio"] == [2.]
    assert info["portfolio_value"] == pytest.approx(1)
    assert info["turnover"] == 0
    assert info["cost_fraction"] == 0
    assert env.signal_close[:, 0] == pytest.approx(np.full(7, 10.))


def test_fee_accounting_matches_original_environment_without_actions():
    from rlasset.corporate_actions import CorporateActionPortfolioEnv
    from rlasset.env import PortfolioEnv
    from rlasset.data import synthetic_frame
    panel = MarketPanel.from_frame(synthetic_frame(periods=30, assets=2))
    kwargs = dict(start=3, end=12, lookback=3, cost_bps=17, risk_aversion=2, max_weight=.7)
    actual = CorporateActionPortfolioEnv(panel, **kwargs)
    expected = PortfolioEnv(panel, **kwargs)
    for target in ([.2,.3,.5], None, [.1,.6,.3], [.5,.2,.3]):
        a = actual.step_weights(target)
        e = expected.step_weights(target)
        assert a[1] == pytest.approx(e[1], abs=1e-12)
        for field in ("portfolio_value", "net_return", "turnover", "cost_fraction", "risk_penalty"):
            assert a[-1][field] == pytest.approx(e[-1][field], abs=1e-12)
        assert a[-1]["weights"] == pytest.approx(e[-1]["weights"])


def test_future_actions_do_not_change_prior_observations_or_signal(tmp_path):
    from rlasset.corporate_actions import CorporateActionPortfolioEnv
    panel = make_panel()
    actions = actions_from_files(tmp_path, panel, [("ETF", "2024-01-09", "2024-01-15", 1.)])
    first = CorporateActionPortfolioEnv(panel, 1, 6, lookback=1, actions=actions)
    second = CorporateActionPortfolioEnv(panel, 1, 6, lookback=1)
    np.testing.assert_array_equal(first.signal_close[:6], second.signal_close[:6])
    for _ in range(3):
        a = first.hold()[0]
        b = second.hold()[0]
        np.testing.assert_array_equal(a["prices"], b["prices"])


def test_reset_clears_all_pending_receivables_and_holdings(tmp_path):
    from rlasset.corporate_actions import CorporateActionPortfolioEnv
    panel = make_panel()
    actions = actions_from_files(tmp_path, panel, [("ETF", "2024-01-04", "2024-01-20", 1.)])
    env = CorporateActionPortfolioEnv(panel, 1, 6, lookback=1, max_weight=1, actions=actions)
    env.step_weights([0, 1]); env.hold()
    _, info = env.reset()
    assert info["receivable"] == 0
    assert info["cash"] == 1
    assert info["shares"] == [0]


@pytest.mark.parametrize("event", [("ETF", "2024-01-04", "2024-01-03", 1),
                                   ("ETF", "2024-01-04", "2024-01-08", -1),
                                   ("ETF", "2024-01-06", "2024-01-08", 1)])
def test_invalid_dividend_events_fail(tmp_path, event):
    with pytest.raises(ValueError):
        actions_from_files(tmp_path, make_panel(), [event])


def test_duplicate_action_and_bad_split_fail(tmp_path):
    event = ("ETF", "2024-01-04", "2024-01-08", 1)
    with pytest.raises(ValueError):
        actions_from_files(tmp_path, make_panel(), [event, event])
    with pytest.raises(ValueError):
        actions_from_files(tmp_path, make_panel(), splits=[("ETF", "2024-01-04", 0)])


def test_sale_fees_apply_to_shares_without_charging_or_spending_receivables(tmp_path):
    from rlasset.corporate_actions import CorporateActionPortfolioEnv
    panel = make_panel()
    actions = actions_from_files(tmp_path, panel, [("ETF", "2024-01-04", "2024-01-08", 1.)])
    env = CorporateActionPortfolioEnv(panel, 1, 6, lookback=1, cost_bps=0, max_weight=1, actions=actions)
    env.step_weights([0, 1])
    env.cost_rate = .01
    info = env.step_weights([1, 0])[-1]
    assert info["cash"] == pytest.approx(.891)
    assert info["receivable"] == pytest.approx(.1)
    assert info["portfolio_value"] == pytest.approx(.991)
    assert info["cost_fraction"] == pytest.approx(.009)
    assert info["turnover"] == pytest.approx(.9)


def test_same_day_split_and_dividend_use_post_split_share_entitlement(tmp_path):
    from rlasset.corporate_actions import CorporateActionPortfolioEnv
    panel = make_panel([10., 10., 10., 4.5, 4.5, 4.5, 4.5])
    actions = actions_from_files(tmp_path, panel, [("ETF", "2024-01-04", "2024-01-08", .5)],
                                 [("ETF", "2024-01-04", 2.)])
    env = CorporateActionPortfolioEnv(panel, 1, 6, lookback=1, cost_bps=0, max_weight=1, actions=actions)
    env.step_weights([0, 1])
    info = env.hold()[-1]
    assert info["shares"] == pytest.approx([.2])
    assert info["dividend_entitlement"] == pytest.approx(.1)
    assert info["portfolio_value"] == pytest.approx(1)
    assert env.signal_close[:, 0] == pytest.approx(np.full(7, 10.))
