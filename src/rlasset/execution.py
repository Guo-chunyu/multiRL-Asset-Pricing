"""Daily-bar, immediate-or-cancel paper execution on unadjusted prices.

This is an offline fill model, not an exchange queue simulator. Whole-session
volume caps fills but is never consulted when sizing requested target orders.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import math
from numbers import Integral
from typing import Mapping


def _date(value: str | date) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _finite(value: float, name: str, *, positive: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


@dataclass(frozen=True)
class ExecutionBar:
    date: str | date
    ticker: str
    open: float
    close: float
    volume: float
    halted: bool = False
    limit_up: float | None = None
    limit_down: float | None = None
    price_basis: str = "raw"

    def __post_init__(self):
        object.__setattr__(self, "date", _date(self.date))
        if not isinstance(self.ticker, str) or not self.ticker or self.ticker == "cash":
            raise ValueError("ticker must be a nonempty security identifier")
        for field in ("open", "close", "volume"):
            object.__setattr__(self, field, _finite(getattr(self, field), field,
                                                  positive=field != "volume"))
        for field in ("limit_up", "limit_down"):
            if getattr(self, field) is not None:
                object.__setattr__(self, field, _finite(getattr(self, field), field, positive=True))
        if self.limit_down is not None and self.limit_up is not None and self.limit_down > self.limit_up:
            raise ValueError("limit_down cannot exceed limit_up")
        if self.price_basis not in {"raw", "unadjusted"}:
            raise ValueError("execution requires raw unadjusted prices")


@dataclass(frozen=True)
class ExecutionConfig:
    lot_size: int = 100
    commission_rate: float = .0003
    min_commission: float = 5.
    slippage_bps: float = 0.
    participation_rate: float = .1
    t_plus_one: bool = True

    def __post_init__(self):
        if isinstance(self.lot_size, bool) or not isinstance(self.lot_size, Integral) or self.lot_size <= 0:
            raise ValueError("lot_size must be a positive integer")
        for field in ("commission_rate", "min_commission", "slippage_bps", "participation_rate"):
            object.__setattr__(self, field, _finite(getattr(self, field), field))
        if self.slippage_bps >= 10000:
            raise ValueError("slippage_bps must be less than 10000")
        if not 0 < self.participation_rate <= 1:
            raise ValueError("participation_rate must be in (0, 1]")


@dataclass(frozen=True)
class OrderRecord:
    client_order_id: str
    date: str
    ticker: str
    side: str
    requested_quantity: int
    filled_quantity: int
    cancelled_quantity: int
    filled_price: float | None
    notional: float
    commission: float
    slippage_cost: float
    status: str
    reject_reason: str | None
    cash_after: float
    position_after: int
    sellable_after: int

    def to_dict(self) -> dict:
        return asdict(self)


class PaperBroker:
    """Long-only share/cash ledger; orders are IOC and never remain pending.

    Imported initial positions are assumed settled. ``advance_date`` takes
    trading-session dates supplied by the caller; it does not own a calendar.
    """

    def __init__(self, initial_cash: float, *, config: ExecutionConfig | None = None,
                 initial_positions: Mapping[str, int] | None = None):
        self.config = config or ExecutionConfig()
        self._cash = _finite(initial_cash, "initial_cash")
        self._positions: dict[str, int] = {}
        for ticker, quantity in (initial_positions or {}).items():
            if not isinstance(ticker, str) or not ticker or ticker == "cash":
                raise ValueError("invalid initial ticker")
            if isinstance(quantity, bool) or not isinstance(quantity, Integral) or quantity < 0:
                raise ValueError("initial positions must be nonnegative integer shares")
            if quantity:
                self._positions[ticker] = int(quantity)
        self._bought_today: dict[str, int] = {}
        self._date: date | None = None
        self._closed = False
        self._volume_used: dict[str, int] = {}
        self._orders: dict[str, OrderRecord] = {}
        self._intents: dict[str, tuple] = {}
        self._batches: dict[str, tuple[tuple, tuple[OrderRecord, ...]]] = {}
        self._commission = 0.
        self._slippage = 0.

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def available_cash(self) -> float:
        return self._cash

    @property
    def reserved_cash(self) -> float:
        return 0.

    @property
    def positions(self) -> dict[str, int]:
        return dict(self._positions)

    @property
    def sellable(self) -> dict[str, int]:
        return {ticker: quantity - self._bought_today.get(ticker, 0)
                if self.config.t_plus_one else quantity
                for ticker, quantity in self._positions.items()}

    @property
    def orders(self) -> tuple[OrderRecord, ...]:
        return tuple(self._orders.values())

    def advance_date(self, session_date: str | date) -> None:
        session_date = _date(session_date)
        if self._date is not None and session_date < self._date:
            raise ValueError("cannot move the trading date backward")
        if session_date != self._date:
            self._bought_today.clear()
            self._volume_used.clear()
            self._closed = False
            self._date = session_date

    def _fee(self, quantity: int, price: float) -> float:
        return max(self.config.min_commission, quantity * price * self.config.commission_rate) if quantity else 0.

    def submit_order(self, client_order_id: str, ticker: str, side: str,
                     quantity: int, bar: ExecutionBar) -> OrderRecord:
        """Execute once or return the original receipt for an identical retry.

        Conflicting reuse of an ID is an API error (ValueError); market/risk
        rejections produce immutable OrderRecords and consume that ID.
        """
        if not isinstance(client_order_id, str) or not client_order_id:
            raise ValueError("client_order_id must be nonempty")
        if ticker != bar.ticker:
            raise ValueError("order ticker does not match its bar")
        if side not in {"buy", "sell"}:
            raise ValueError("side must be buy or sell")
        if isinstance(quantity, bool) or not isinstance(quantity, Integral) or quantity <= 0:
            raise ValueError("quantity must be a positive integer")
        quantity = int(quantity)
        intent = (bar.date, ticker, side, quantity)
        if client_order_id in self._orders:
            if self._intents[client_order_id] != intent:
                raise ValueError("conflicting client_order_id")
            return self._orders[client_order_id]
        self.advance_date(bar.date)
        position = self._positions.get(ticker, 0)
        reason = None
        lot = self.config.lot_size
        if self._closed:
            reason = "session_closed"
        elif quantity % lot and (side == "buy" or quantity != position):
            reason = "invalid_lot"
        elif bar.halted:
            reason = "halted"
        elif side == "buy" and bar.limit_up is not None and bar.open >= bar.limit_up - 1e-10:
            reason = "limit_up"
        elif side == "sell" and bar.limit_down is not None and bar.open <= bar.limit_down + 1e-10:
            reason = "limit_down"
        elif bar.volume == 0:
            reason = "no_volume"
        elif side == "sell" and quantity > position:
            reason = "insufficient_position"

        price = bar.open * (1 + (1 if side == "buy" else -1) * self.config.slippage_bps / 10000)
        if bar.limit_up is not None:
            price = min(price, bar.limit_up)
        if bar.limit_down is not None:
            price = max(price, bar.limit_down)
        filled = quantity if reason is None else 0
        if filled and side == "sell":
            settled = self.sellable.get(ticker, 0)
            if settled < filled:
                filled, reason = settled, "t_plus_one"
        if filled:
            capacity = max(0, math.floor(bar.volume * self.config.participation_rate + 1e-9)
                           - self._volume_used.get(ticker, 0))
            if capacity < filled:
                filled, reason = capacity, "participation_limit"
        if filled:
            # An odd residual may be sold only when the complete position can
            # actually be liquidated; partial IOC sales still use board lots.
            if side == "buy" or filled != position:
                filled = filled // lot * lot
            if side == "buy":
                cash_bound = min(self._cash / (price * (1 + self.config.commission_rate)),
                                 max(0., self._cash - self.config.min_commission) / price)
                affordable = max(0, math.floor(cash_bound / lot) * lot)
                # Recheck the actual cash operation, without an epsilon that
                # could turn a nearly affordable lot into a real overdraft.
                if affordable and affordable * price + self._fee(affordable, price) > self._cash:
                    affordable -= lot
                if affordable < filled:
                    filled, reason = affordable, "insufficient_cash"
            elif self._cash + filled * price - self._fee(filled, price) < 0:
                filled, reason = 0, "insufficient_cash"

        notional = filled * price
        commission = self._fee(filled, price)
        slippage = filled * (price - bar.open) * (1 if side == "buy" else -1)
        if filled:
            self._cash = (self._cash - (notional + commission) if side == "buy"
                          else self._cash + notional - commission)
            if self._cash < 0:
                raise ArithmeticError("cash conservation failed")
            new_position = position + (filled if side == "buy" else -filled)
            if new_position:
                self._positions[ticker] = new_position
            else:
                self._positions.pop(ticker, None)
            if side == "buy":
                self._bought_today[ticker] = self._bought_today.get(ticker, 0) + filled
            elif not self.config.t_plus_one:
                self._bought_today[ticker] = min(self._bought_today.get(ticker, 0), new_position)
            self._volume_used[ticker] = self._volume_used.get(ticker, 0) + filled
            self._commission += commission
            self._slippage += slippage
        status = "filled" if filled == quantity else "partially_filled" if filled else "rejected"
        record = OrderRecord(client_order_id, bar.date.isoformat(), ticker, side,
                             quantity, filled, quantity - filled, price if filled else None,
                             notional, commission, slippage, status,
                             None if filled == quantity else reason or "participation_limit",
                             self._cash, self._positions.get(ticker, 0), self.sellable.get(ticker, 0))
        self._orders[client_order_id] = record
        self._intents[client_order_id] = intent
        return record

    @staticmethod
    def _bar_date(bars: Mapping[str, ExecutionBar]) -> date:
        if not bars or any(ticker != bar.ticker for ticker, bar in bars.items()):
            raise ValueError("bars must be nonempty with matching ticker keys")
        dates = {bar.date for bar in bars.values()}
        if len(dates) != 1:
            raise ValueError("all bars must belong to the same trading date")
        return next(iter(dates))

    def rebalance(self, target_weights: Mapping[str, float], bars: Mapping[str, ExecutionBar],
                  *, decision_date: str | date, client_order_prefix: str) -> list[OrderRecord]:
        """Translate prior-session targets at today's open; sell then buy.

        Targets contain risky securities only, with implicit residual cash.
        Sizing uses current cash/positions and opening prices, never daily
        close, halt, limits or volume. Fees and fill constraints may leave drift.
        """
        session_date = self._bar_date(bars)
        decision_date = _date(decision_date)
        if decision_date >= session_date:
            raise ValueError("decision_date must precede the execution date")
        if not isinstance(client_order_prefix, str) or not client_order_prefix:
            raise ValueError("client_order_prefix must be nonempty")
        targets = {ticker: _finite(weight, "target weight") for ticker, weight in target_weights.items()}
        if any(not isinstance(ticker, str) or not ticker or ticker == "cash" for ticker in targets):
            raise ValueError("targets must contain security tickers; cash is implicit")
        if sum(targets.values()) > 1 + 1e-12:
            raise ValueError("target weights must sum to at most one")
        fingerprint = (decision_date, session_date, tuple(sorted(targets.items())))
        if client_order_prefix in self._batches:
            old_intent, records = self._batches[client_order_prefix]
            if old_intent != fingerprint:
                raise ValueError("conflicting client_order_prefix")
            return list(records)
        securities = set(targets) | set(self._positions)
        if not securities <= set(bars):
            raise ValueError("missing bars for targets or current holdings")
        if self._date is not None and session_date < self._date:
            raise ValueError("cannot move the trading date backward")
        nav = self._cash + sum(quantity * bars[ticker].open for ticker, quantity in self._positions.items())
        lot = self.config.lot_size
        deltas = {}
        for ticker in sorted(securities):
            target = math.floor(nav * targets.get(ticker, 0.) / bars[ticker].open / lot) * lot
            difference = target - self._positions.get(ticker, 0)
            # Retaining an odd residual is preferable to requesting an invalid
            # partial odd-lot sale. Full liquidation may dispose of all shares.
            if difference < 0 and target:
                difference = -((-difference) // lot * lot)
            elif difference > 0:
                difference = difference // lot * lot
            if difference:
                deltas[ticker] = difference
        ids = {ticker: f"{client_order_prefix}:{ticker}:{'buy' if delta > 0 else 'sell'}"
               for ticker, delta in deltas.items()}
        if any(order_id in self._orders for order_id in ids.values()):
            raise ValueError("client_order_prefix overlaps existing client_order_id")
        self.advance_date(session_date)
        records = []
        for direction in (-1, 1):
            for ticker, delta in deltas.items():
                if (delta > 0) == (direction > 0):
                    records.append(self.submit_order(ids[ticker], ticker,
                                                     "buy" if delta > 0 else "sell",
                                                     abs(delta), bars[ticker]))
        self._batches[client_order_prefix] = (fingerprint, tuple(records))
        return records

    def close_snapshot(self, bars: Mapping[str, ExecutionBar]) -> dict:
        """Mark all holdings at supplied raw closes and end today's session."""
        session_date = self._bar_date(bars)
        if not set(self._positions) <= set(bars):
            raise ValueError("missing closing bars for current holdings")
        self.advance_date(session_date)
        values = {ticker: self._positions.get(ticker, 0) * bar.close for ticker, bar in bars.items()}
        nav = self._cash + sum(values.values())
        weights = {ticker: value / nav if nav else 0. for ticker, value in values.items()}
        weights["cash"] = self._cash / nav if nav else 0.
        self._closed = True
        return {"date": session_date.isoformat(), "nav": nav, "cash": self._cash,
                "available_cash": self.available_cash, "reserved_cash": self.reserved_cash,
                "positions": self.positions, "sellable": self.sellable, "weights": weights,
                "mark_prices": {ticker: bar.close for ticker, bar in bars.items()},
                "cumulative_commission": self._commission, "cumulative_slippage": self._slippage}
