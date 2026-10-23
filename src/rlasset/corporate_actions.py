"""Raw-price portfolio accounting with dividend receivables and share splits.

The feature-only total-return index reinvests on the ex-date. The actual account
does not: dividends remain unavailable receivables until their payment session.
All share quantities are fractional, as in PortfolioEnv; exchange fills belong
to a separate execution simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import MarketPanel
from .env import PortfolioEnv


@dataclass(frozen=True)
class Dividend:
    asset: int
    ex_index: int
    pay_index: int
    cash_per_share: float


@dataclass(frozen=True)
class CorporateActions:
    dates: pd.DatetimeIndex
    tickers: tuple[str, ...]
    cash_per_share: np.ndarray
    split_ratio: np.ndarray
    dividends: tuple[Dividend, ...] = ()

    @classmethod
    def empty(cls, panel: MarketPanel):
        shape = panel.close.shape
        return cls(panel.dates.copy(), panel.tickers, np.zeros(shape), np.ones(shape))

    @classmethod
    def from_csv(cls, panel: MarketPanel, cash_path=None, splits_path=None):
        """Read explicit cash and split events aligned to the raw price panel.

        Cash columns: ticker, ex_date, pay_date, cash_per_share. Split columns:
        ticker, effective_date, ratio (new shares per old share). Events before
        the panel cannot entitle the initially empty account; events after it
        cannot change its history. Nontrading payment dates roll to the next
        observed session, including the index just beyond this panel. Ex/split
        dates inside the panel must be observed trading sessions.
        """
        result = cls.empty(panel)
        dividends = []
        lookup = {ticker: i for i, ticker in enumerate(panel.tickers)}
        seen_cash, seen_splits = set(), set()

        def read(path, columns):
            if path is None:
                return []
            frame = pd.read_csv(Path(path), dtype={"ticker": str})
            if not set(columns) <= set(frame.columns) or frame[columns].isna().any().any():
                raise ValueError(f"Corporate-action CSV requires nonmissing columns {columns}")
            return frame.to_dict("records")

        def index_on(date):
            if date < panel.dates[0] or date > panel.dates[-1]:
                return None
            index = panel.dates.get_indexer([date])[0]
            if index < 0:
                raise ValueError("Ex-dividend/split date must be a panel trading session")
            return int(index)

        for row in read(cash_path, ["ticker", "ex_date", "pay_date", "cash_per_share"]):
            ticker = row["ticker"]
            if ticker not in lookup:
                raise ValueError(f"Unknown corporate-action ticker: {ticker}")
            ex_date = pd.Timestamp(row["ex_date"]).normalize()
            pay_date = pd.Timestamp(row["pay_date"]).normalize()
            amount = float(row["cash_per_share"])
            if pd.isna(ex_date) or pd.isna(pay_date) or pay_date < ex_date or not np.isfinite(amount) or amount <= 0:
                raise ValueError("Dividend needs a positive amount and payment on/after ex-date")
            key = (ticker, ex_date)
            if key in seen_cash:
                raise ValueError("Duplicate dividend event")
            seen_cash.add(key)
            ex_index = index_on(ex_date)
            if ex_index is None:
                continue
            asset = lookup[ticker]
            pay_index = int(panel.dates.searchsorted(pay_date, side="left"))
            result.cash_per_share[ex_index, asset] = amount
            dividends.append(Dividend(asset, ex_index, pay_index, amount))
        for row in read(splits_path, ["ticker", "effective_date", "ratio"]):
            ticker = row["ticker"]
            if ticker not in lookup:
                raise ValueError(f"Unknown corporate-action ticker: {ticker}")
            date = pd.Timestamp(row["effective_date"]).normalize()
            ratio = float(row["ratio"])
            if pd.isna(date) or not np.isfinite(ratio) or ratio <= 0:
                raise ValueError("Split needs a valid date and positive finite ratio")
            key = (ticker, date)
            if key in seen_splits:
                raise ValueError("Duplicate split event")
            seen_splits.add(key)
            index = index_on(date)
            if index is not None:
                result.split_ratio[index, lookup[ticker]] = ratio
        return cls(result.dates, result.tickers, result.cash_per_share, result.split_ratio,
                   tuple(sorted(dividends, key=lambda event: (event.ex_index, event.asset))))

    def signal_close(self, panel: MarketPanel) -> np.ndarray:
        """Causal ex-date reinvestment index, for observations and risk only."""
        if self.tickers != panel.tickers or not self.dates.equals(panel.dates):
            raise ValueError("Corporate actions must match the exact panel calendar and tickers")
        for values in (self.cash_per_share, self.split_ratio):
            if values.shape != panel.close.shape or not np.isfinite(values).all():
                raise ValueError("Corporate-action arrays must match the panel and be finite")
        if (self.cash_per_share < 0).any() or (self.split_ratio <= 0).any():
            raise ValueError("Dividend amounts must be nonnegative and split ratios positive")
        signal = panel.close.copy()
        gross = self.split_ratio[1:] * (panel.close[1:] + self.cash_per_share[1:]) / panel.close[:-1]
        signal[1:] = signal[0] * np.cumprod(gross, axis=0)
        return signal


class CorporateActionPortfolioEnv(PortfolioEnv):
    """Trade raw prices and keep ex-date entitlements until actual payment.

    Actions target the investable NAV (cash plus shares at next open), excluding
    receivables. Reported portfolio weights include receivables in the cash slot;
    they are not a promise that the entire cash weight can be spent. On a combined
    split/dividend date, the dividend amount is per post-split share.
    """

    def __init__(self, panel, start, end, lookback=20, cost_bps=5, risk_aversion=0,
                 max_weight=.5, reward_scale=100, actions=None):
        self.actions = actions if actions is not None else CorporateActions.empty(panel)
        self.signal_close = self.actions.signal_close(panel)
        self._events_by_ex = {}
        for event in self.actions.dividends:
            self._events_by_ex.setdefault(event.ex_index, []).append(event)
        super().__init__(panel, start, end, lookback, cost_bps, risk_aversion, max_weight, reward_scale)

    def _observation(self):
        window = slice(self.t - self.lookback + 1, self.t + 1)
        base = self.signal_close[window]
        current = self.signal_close[self.t]
        prices = np.stack([
            (base / current).T,
            (base * (self.panel.high[window] / self.panel.close[window]) / current).T,
            (base * (self.panel.low[window] / self.panel.close[window]) / current).T,
        ]).astype(np.float32)
        return {"prices": prices, "weights": self.weights.astype(np.float32).copy()}

    def reset(self, *, seed=None, options=None):
        self.cash = 1.0
        self.shares = np.zeros(self.n_assets)
        self.receivable = 0.0
        self._pending_dividends = []
        observation, info = super().reset(seed=seed, options=options)
        info.update(cash=self.cash, shares=self.shares.tolist(), receivable=0.,
                    dividend_entitlement=0., dividend_paid=0., split_ratio=np.ones(self.n_assets).tolist())
        self.history[0] = info.copy()
        return observation, info

    def _risk_penalty(self, target):
        if self.risk_aversion == 0 or self.lookback < 2:
            return 0.0
        prices = self.signal_close[self.t - self.lookback:self.t + 1]
        returns = prices[1:] / prices[:-1] - 1
        covariance = np.atleast_2d(np.cov(returns, rowvar=False, ddof=1))
        return self.risk_aversion * max(float(target[1:] @ covariance @ target[1:]), 0.0)

    def step_weights(self, weights):
        if self._terminated:
            raise RuntimeError("Episode is terminated; call reset before another step")
        target = None
        if weights is not None:
            target = np.asarray(weights, dtype=float)
            if target.shape != (self.n_assets + 1,) or not np.isfinite(target).all():
                raise ValueError("weights must be a finite cash-first vector of length N+1")
            if (target < 0).any() or not np.isclose(target.sum(), 1, atol=1e-7, rtol=0):
                raise ValueError("weights must be nonnegative and sum to 1")
            if (target[1:] > self.max_weight + 1e-7).any():
                raise ValueError("risky target weights exceed max_weight")
            target = target / target.sum()

        decision_t, next_t = self.t, self.t + 1
        previous_value = self.value
        split_ratio = self.actions.split_ratio[next_t]
        self.shares = self.shares * split_ratio
        entitlement = 0.0
        for event in self._events_by_ex.get(next_t, []):
            amount = float(self.shares[event.asset] * event.cash_per_share)
            entitlement += amount
            self._pending_dividends.append((event.pay_index, amount))
        self.receivable += entitlement
        paid = sum(amount for pay_index, amount in self._pending_dividends if pay_index <= next_t)
        self._pending_dividends = [(pay_index, amount) for pay_index, amount in self._pending_dividends if pay_index > next_t]
        self.cash += paid
        self.receivable = sum(amount for _, amount in self._pending_dividends)

        open_prices = self.panel.open[next_t]
        open_positions = self.shares * open_prices
        tradable = float(self.cash + open_positions.sum())
        open_value = tradable + self.receivable
        pretrade = np.r_[self.cash + self.receivable, open_positions] / open_value
        tradable_pretrade = np.r_[self.cash, open_positions] / tradable
        if target is None:
            mu, turnover = 1.0, 0.0
            actual_target = tradable_pretrade.copy()
        else:
            actual_target = target
            mu, turnover = self._transaction_factor(tradable_pretrade, target)
            after_fee = tradable * mu
            self.cash = float(target[0] * after_fee)
            self.shares = target[1:] * after_fee / open_prices
        fees = tradable * (1 - mu)
        effective_target = np.r_[self.cash + self.receivable, self.shares * open_prices] / (open_value - fees)
        risk_penalty = self._risk_penalty(effective_target)
        close_positions = self.shares * self.panel.close[next_t]
        self.value = float(self.cash + close_positions.sum() + self.receivable)
        self.weights = np.r_[self.cash + self.receivable, close_positions] / self.value
        net_growth = self.value / previous_value
        reward = self.reward_scale * (np.log(net_growth) - risk_penalty)
        self.t = next_t
        self._terminated = self.t >= self.end
        info = {
            "date": self.panel.dates[next_t].isoformat(),
            "decision_date": self.panel.dates[decision_t].isoformat(),
            "net_return": float(net_growth - 1), "portfolio_value": self.value,
            "turnover": float(turnover * tradable / open_value),
            "cost_fraction": float(fees / open_value),
            "weights": self.weights.tolist(), "target_weights": actual_target.tolist(),
            "pretrade_weights": pretrade.tolist(), "gap_growth": float(open_value / previous_value),
            "intraday_growth": float(self.value / (open_value - fees)),
            "risk_penalty": float(risk_penalty), "cash": self.cash,
            "shares": self.shares.tolist(), "receivable": float(self.receivable),
            "dividend_entitlement": float(entitlement), "dividend_paid": float(paid),
            "split_ratio": split_ratio.tolist(),
        }
        self.history.append(info)
        return self._observation(), float(reward), self._terminated, False, info.copy()
