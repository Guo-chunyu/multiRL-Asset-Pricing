"""Hand-accounted event-driven execution checks, independent of RL policies."""
import importlib
import json
import math

import pytest


def api():
    return importlib.import_module("rlasset.execution")


def bar(date="2024-01-02", ticker="510300", price=10.0, close=None, volume=100000, **kwargs):
    return api().ExecutionBar(date=date, ticker=ticker, open=price,
                              close=price if close is None else close, volume=volume, **kwargs)


def broker(cash=10000, positions=None, **kwargs):
    return api().PaperBroker(cash, config=api().ExecutionConfig(**kwargs),
                             initial_positions=positions)


def test_fill_fee_and_slippage_conserve_open_marked_equity():
    account = broker(slippage_bps=10)
    record = account.submit_order("buy-1", "510300", "buy", 100, bar(close=11))
    assert record.status == "filled"
    assert record.filled_price == pytest.approx(10.01)
    assert record.commission == 5
    assert record.slippage_cost == pytest.approx(1)
    assert account.cash == pytest.approx(8994)
    result = account.close_snapshot({"510300": bar(close=11)})
    assert result["nav"] == pytest.approx(10094)
    assert result["weights"]["510300"] == pytest.approx(1100 / 10094)
    assert result["date"] == "2024-01-02"
    assert result["reserved_cash"] == 0
    json.dumps(record.to_dict(), allow_nan=False)
    json.dumps(result, allow_nan=False)


def test_t_plus_one_book_prevents_intraday_sale_and_matures_once():
    account = broker()
    account.submit_order("b", "510300", "buy", 100, bar())
    rejected = account.submit_order("s0", "510300", "sell", 100, bar())
    assert rejected.status == "rejected"
    assert rejected.reject_reason == "t_plus_one"
    assert account.positions == {"510300": 100}
    account.advance_date("2024-01-03")
    account.advance_date("2024-01-03")
    sold = account.submit_order("s1", "510300", "sell", 100, bar("2024-01-03"))
    assert sold.filled_quantity == 100
    assert account.cash == pytest.approx(9990)
    assert account.positions == {}
    with pytest.raises(ValueError, match="backward"):
        account.advance_date("2024-01-02")


def test_old_position_can_sell_while_new_purchase_remains_unsettled():
    account = broker(positions={"510300": 100})
    account.submit_order("b", "510300", "buy", 100, bar())
    sold = account.submit_order("s", "510300", "sell", 200, bar())
    assert sold.status == "partially_filled"
    assert sold.filled_quantity == 100
    assert sold.cancelled_quantity == 100
    assert sold.reject_reason == "t_plus_one"
    assert account.positions == {"510300": 100}
    assert account.sellable == {"510300": 0}


def test_participation_is_shared_across_orders_and_unfilled_cash_is_released():
    account = broker(participation_rate=.1)
    first = account.submit_order("a", "510300", "buy", 500, bar(volume=2500))
    second = account.submit_order("b", "510300", "buy", 100, bar(volume=2500))
    assert first.filled_quantity == 200
    assert first.status == "partially_filled"
    assert first.cancelled_quantity == 300
    assert first.reject_reason == "participation_limit"
    assert second.filled_quantity == 0
    assert second.reject_reason == "participation_limit"
    assert account.cash == 7995
    assert account.available_cash == account.cash
    assert account.reserved_cash == 0


def test_minimum_commission_reduces_fill_before_cash_can_go_negative():
    account = broker(cash=2000)
    record = account.submit_order("buy", "510300", "buy", 200, bar())
    assert record.filled_quantity == 100
    assert record.reject_reason == "insufficient_cash"
    assert account.cash == 995
    assert account.reserved_cash == 0


def test_duplicate_order_is_idempotent_and_conflicting_id_is_rejected():
    account = broker()
    first = account.submit_order("same", "510300", "buy", 100, bar())
    cash_after = account.cash
    assert account.submit_order("same", "510300", "buy", 100, bar()) == first
    assert account.cash == cash_after
    assert len(account.orders) == 1
    with pytest.raises(ValueError, match="client_order_id"):
        account.submit_order("same", "510300", "buy", 200, bar())
    assert account.cash == cash_after


@pytest.mark.parametrize("side,kwargs,reason", [
    ("buy", {"halted": True}, "halted"),
    ("sell", {"halted": True}, "halted"),
    ("buy", {"limit_up": 10}, "limit_up"),
    ("sell", {"limit_down": 10}, "limit_down"),
    ("buy", {"volume": 0}, "no_volume"),
])
def test_untradeable_directions_are_recorded_without_fees(side, kwargs, reason):
    account = broker(positions={"510300": 100})
    record = account.submit_order("x", "510300", side, 100, bar(**kwargs))
    assert record.status == "rejected"
    assert record.reject_reason == reason
    assert record.commission == 0
    assert account.cash == 10000
    assert account.positions == {"510300": 100}


def test_limit_only_blocks_adverse_side_and_halted_position_can_be_marked():
    account = broker(positions={"510300": 100})
    assert account.submit_order("sell", "510300", "sell", 100, bar(limit_up=10)).status == "filled"
    other = broker()
    assert other.submit_order("buy", "510300", "buy", 100, bar(limit_down=10)).status == "filled"
    snap = other.close_snapshot({"510300": bar(close=9, halted=True)})
    assert snap["positions"]["510300"] == 100
    assert snap["nav"] == 9895


def test_buy_lots_and_odd_lot_liquidation():
    account = broker(positions={"510300": 150})
    assert account.submit_order("invalid", "510300", "buy", 50, bar()).reject_reason == "invalid_lot"
    assert account.submit_order("liquidate", "510300", "sell", 150, bar()).filled_quantity == 150
    assert account.positions == {}


def test_rebalance_sells_before_buying_and_has_batch_idempotency():
    account = broker(cash=0, positions={"A": 1000})
    bars = {"A": bar(ticker="A"), "B": bar(ticker="B")}
    orders = account.rebalance({"B": 1.0}, bars, decision_date="2024-01-01",
                               client_order_prefix="day-1")
    assert [(o.ticker, o.side) for o in orders] == [("A", "sell"), ("B", "buy")]
    assert orders[1].requested_quantity == 1000
    assert orders[1].filled_quantity == 900
    assert account.cash == 990
    assert account.positions == {"B": 900}
    again = account.rebalance({"B": 1.0}, bars, decision_date="2024-01-01",
                              client_order_prefix="day-1")
    assert orders == again and len(account.orders) == 2
    with pytest.raises(ValueError, match="prefix"):
        account.rebalance({"B": .5}, bars, decision_date="2024-01-01",
                          client_order_prefix="day-1")


def test_next_close_and_volume_do_not_change_requested_order_sizing():
    low, high = broker(), broker()
    low_orders = low.rebalance({"510300": 1.0}, {"510300": bar(close=1, volume=1000)},
                              decision_date="2024-01-01", client_order_prefix="low")
    high_orders = high.rebalance({"510300": 1.0}, {"510300": bar(close=100, volume=1000000)},
                                decision_date="2024-01-01", client_order_prefix="high")
    assert low_orders[0].requested_quantity == high_orders[0].requested_quantity == 1000
    assert low_orders[0].filled_quantity == 100
    assert high_orders[0].filled_quantity == 900


def test_decision_must_precede_execution_and_close_prevents_late_orders():
    account = broker()
    with pytest.raises(ValueError, match="decision"):
        account.rebalance({"510300": 1}, {"510300": bar()},
                          decision_date="2024-01-02", client_order_prefix="lookahead")
    account.close_snapshot({"510300": bar()})
    record = account.submit_order("late", "510300", "buy", 100, bar())
    assert record.reject_reason == "session_closed"


def test_adjusted_price_bars_and_invalid_target_batch_fail_before_mutation():
    with pytest.raises(ValueError, match="unadjusted"):
        bar(price_basis="hfq")
    account = broker()
    with pytest.raises(ValueError):
        account.rebalance({"510300": 1.1}, {"510300": bar()},
                          decision_date="2024-01-01", client_order_prefix="invalid")
    assert account.cash == 10000 and not account.orders


def test_oversell_and_tiny_negative_proceeds_cannot_create_cash_or_short_positions():
    account = broker(cash=0, positions={"510300": 100})
    assert account.submit_order("too-many", "510300", "sell", 200, bar()).reject_reason == "insufficient_position"
    tiny = account.submit_order("fee-exceeds-proceeds", "510300", "sell", 100, bar(price=.01))
    assert tiny.reject_reason == "insufficient_cash"
    assert account.cash == 0 and account.positions == {"510300": 100}


def test_date_replay_does_not_unlock_new_shares_or_repeat_a_historical_fill():
    account = broker()
    first = account.submit_order("original", "510300", "buy", 100, bar())
    account.advance_date("2024-01-02")
    assert account.sellable == {"510300": 0}
    account.advance_date("2024-01-03")
    before = account.cash
    assert account.submit_order("original", "510300", "buy", 100, bar()) == first
    assert account.cash == before and account.sellable == {"510300": 100}


def test_configurable_same_day_trading_and_zero_fees_preserve_round_trip_cash():
    account = broker(t_plus_one=False, commission_rate=0, min_commission=0)
    account.submit_order("buy", "510300", "buy", 100, bar())
    assert account.sellable == {"510300": 100}
    account.submit_order("sell", "510300", "sell", 100, bar())
    assert account.cash == 10000 and account.positions == {}


def test_multi_day_receipts_reconcile_cash_positions_and_marked_nav():
    account = broker(cash=30000, slippage_bps=20, participation_rate=.1)
    initial_cash = account.cash
    for day, price, close, weights in [
        ("2024-01-02", 10., 11., {"510300": .8}),
        ("2024-01-03", 12., 13., {"510300": .2}),
        ("2024-01-04", 8., 7., {"510300": 1.}),
        ("2024-01-05", 9., 9., {}),
    ]:
        day_bars = {"510300": bar(day, price=price, close=close, volume=3500)}
        account.rebalance(weights, day_bars, decision_date="2024-01-01", client_order_prefix=day)
        snapshot = account.close_snapshot(day_bars)
        expected_cash = initial_cash
        expected_shares = 0
        for order in account.orders:
            direction = 1 if order.side == "buy" else -1
            expected_cash -= direction * order.notional + order.commission
            expected_shares += direction * order.filled_quantity
        assert account.cash == pytest.approx(expected_cash)
        assert account.positions.get("510300", 0) == expected_shares
        assert snapshot["nav"] == pytest.approx(expected_cash + expected_shares * close)
        assert sum(snapshot["weights"].values()) == pytest.approx(1)
        assert account.cash >= 0
        assert snapshot["reserved_cash"] == 0


def test_floating_point_lot_boundary_never_commits_an_overdraft():
    cash = math.nextafter(100 * 1e9 + 5, 0)
    account = broker(cash=cash, commission_rate=0)
    record = account.submit_order("boundary", "510300", "buy", 100, bar(price=1e9))
    assert record.filled_quantity == 0
    assert record.reject_reason == "insufficient_cash"
    assert account.cash == cash
