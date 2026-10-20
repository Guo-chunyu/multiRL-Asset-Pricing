"""Self-financing, long-only portfolios decided at close and traded next open.

Weights are cash-first. A step starts at close[t], lets existing positions earn
the overnight gap, rebalances at open[t+1], then marks to close[t+1]. Returns
are net of proportional fees on risky-asset purchases and sales; cash is free.
This is a fractional-share research simulator, not an exchange execution engine.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlasset.data import MarketPanel


class PortfolioEnv(gym.Env):
    """A deterministic historical market with an inclusive final valuation day.

    ``start`` is the first decision close; ``end`` is the final valuation close.
    Thus an episode contains exactly ``end - start`` transitions. ``max_weight``
    applies to risky target allocations, not to passive price-driven drift.

    ``cost_bps`` is a one-way notional fee: selling one asset and buying another
    incurs both legs. ``turnover`` is gross risky notional traded / pre-fee NAV,
    without the half-turnover convention, and ``cost_fraction`` has the same
    pre-fee, next-open NAV denominator. Cash has zero interest in this model.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        panel: MarketPanel,
        start: int,
        end: int,
        lookback: int = 20,
        cost_bps: float = 5,
        risk_aversion: float = 0,
        max_weight: float = 0.5,
        reward_scale: float = 100,
    ):
        super().__init__()
        if not all(isinstance(x, (int, np.integer)) for x in (start, end, lookback)):
            raise ValueError("start, end, and lookback must be integers")
        if lookback < 1 or not lookback <= start < end < len(panel.dates):
            raise ValueError("Require 1 <= lookback <= start < end < number of dates")
        if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000:
            raise ValueError("cost_bps must be finite and in [0, 10000)")
        if not np.isfinite(max_weight) or not 0 < max_weight <= 1:
            raise ValueError("max_weight must be finite and in (0, 1]")
        if not np.isfinite(risk_aversion) or risk_aversion < 0:
            raise ValueError("risk_aversion must be finite and nonnegative")
        if not np.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError("reward_scale must be finite and positive")
        self.panel = panel
        self.start = int(start)
        self.end = int(end)
        self.lookback = int(lookback)
        self.cost_bps = float(cost_bps)
        self.cost_rate = self.cost_bps / 10000
        self.risk_aversion = float(risk_aversion)
        self.max_weight = float(max_weight)
        self.reward_scale = float(reward_scale)
        self.n_assets = len(panel.tickers)
        self.action_space = spaces.Box(-1, 1, shape=(self.n_assets + 1,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "prices": spaces.Box(0, np.finfo(np.float32).max,
                                 shape=(3, self.n_assets, self.lookback), dtype=np.float32),
            "weights": spaces.Box(0, 1, shape=(self.n_assets + 1,), dtype=np.float32),
        })
        self.reset()

    def _observation(self) -> dict[str, np.ndarray]:
        window = slice(self.t - self.lookback + 1, self.t + 1)
        current_close = self.panel.close[self.t]
        prices = np.stack([
            (series[window] / current_close).T
            for series in (self.panel.close, self.panel.high, self.panel.low)
        ]).astype(np.float32)
        return {"prices": prices, "weights": self.weights.astype(np.float32).copy()}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.start
        self.value = 1.0
        self.weights = np.zeros(self.n_assets + 1, dtype=float)
        self.weights[0] = 1.0
        self._terminated = False
        initial = {
            "date": self.panel.dates[self.t].isoformat(),
            "decision_date": self.panel.dates[self.t].isoformat(),
            "net_return": 0.0,
            "portfolio_value": 1.0,
            "turnover": 0.0,
            "cost_fraction": 0.0,
            "weights": self.weights.tolist(),
            "target_weights": self.weights.tolist(),
            "pretrade_weights": self.weights.tolist(),
            "gap_growth": 1.0,
            "intraday_growth": 1.0,
            "risk_penalty": 0.0,
        }
        self.history = [initial]
        return self._observation(), initial.copy()

    def action_to_weights(self, action) -> np.ndarray:
        """Map bounded logits to a capped simplex; excess risk weight is cash."""
        action = np.asarray(action, dtype=float)
        if action.shape != (self.n_assets + 1,) or not np.isfinite(action).all():
            raise ValueError("action must be a finite vector with one cash and N risky entries")
        if np.any(action < -1) or np.any(action > 1):
            raise ValueError("action logits must lie in [-1, 1]")
        logits = action * 5
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        probabilities[1:] = np.minimum(probabilities[1:], self.max_weight)
        probabilities[0] = 1.0 - probabilities[1:].sum()
        return probabilities

    def step(self, action):
        return self.step_weights(self.action_to_weights(action))

    def hold(self):
        """Keep shares and cash unchanged through next close, with zero trades."""
        return self.step_weights(None)

    def step_weights(self, weights):
        """Execute feasible target weights; ``None`` means no trades at all.

        A target is a desired *post-fee* next-open weight. Supplying yesterday's
        weights does rebalance after an overnight move; use ``hold()`` to keep
        the share quantities unchanged. Prices never enter action generation.
        """
        if self._terminated:
            raise RuntimeError("Episode is terminated; call reset before another step")
        target = None
        if weights is not None:
            target = np.asarray(weights, dtype=float)
            if target.shape != (self.n_assets + 1,) or not np.isfinite(target).all():
                raise ValueError("weights must be a finite cash-first vector of length N+1")
            if np.any(target < 0) or not np.isclose(target.sum(), 1, atol=1e-7, rtol=0):
                raise ValueError("weights must be nonnegative and sum to 1")
            if np.any(target[1:] > self.max_weight + 1e-7):
                raise ValueError("risky target weights exceed max_weight")
            # Accept float32 simplex roundoff while preserving self-financing.
            target = target / target.sum()

        decision_t = self.t
        next_t = decision_t + 1
        gap_relatives = np.r_[1.0, self.panel.open[next_t] / self.panel.close[decision_t]]
        gap_growth = float(self.weights @ gap_relatives)
        pretrade = self.weights * gap_relatives / gap_growth

        if target is None:
            target = pretrade.copy()
            mu, turnover = 1.0, 0.0
        else:
            mu, turnover = self._transaction_factor(pretrade, target)

        intraday_relatives = np.r_[1.0, self.panel.close[next_t] / self.panel.open[next_t]]
        intraday_growth = float(target @ intraday_relatives)
        net_growth = gap_growth * mu * intraday_growth
        risk_penalty = self._risk_penalty(target)
        reward = self.reward_scale * (np.log(net_growth) - risk_penalty)

        self.value *= net_growth
        self.weights = target * intraday_relatives / intraday_growth
        self.t = next_t
        self._terminated = self.t >= self.end
        info = {
            "date": self.panel.dates[self.t].isoformat(),
            "decision_date": self.panel.dates[decision_t].isoformat(),
            "net_return": float(net_growth - 1),
            "portfolio_value": float(self.value),
            "turnover": float(turnover),
            "cost_fraction": float(1 - mu),
            "weights": self.weights.tolist(),
            "target_weights": target.tolist(),
            "pretrade_weights": pretrade.tolist(),
            "gap_growth": gap_growth,
            "intraday_growth": intraday_growth,
            "risk_penalty": risk_penalty,
        }
        self.history.append(info)
        return self._observation(), float(reward), self._terminated, False, info.copy()

    def _transaction_factor(self, pretrade: np.ndarray, target: np.ndarray):
        """Solve mu = 1 - c * sum_i |mu * target_i - pretrade_i|, risky i.

        For 0 <= c < 1, the residual is strictly increasing on [0,1] and
        brackets a unique root. Both purchases and sales use the same NAV;
        deducting an approximate fee after setting weights would break this.
        """
        if np.allclose(pretrade, target, rtol=0, atol=1e-14):
            return 1.0, 0.0
        if self.cost_rate == 0:
            return 1.0, float(np.abs(target[1:] - pretrade[1:]).sum())
        lower, upper = 0.0, 1.0
        for _ in range(55):
            mu = (lower + upper) / 2
            turnover = np.abs(mu * target[1:] - pretrade[1:]).sum()
            residual = mu + self.cost_rate * turnover - 1
            if residual > 0:
                upper = mu
            else:
                lower = mu
        mu = (lower + upper) / 2
        turnover = float(np.abs(mu * target[1:] - pretrade[1:]).sum())
        return mu, turnover

    def _risk_penalty(self, target: np.ndarray) -> float:
        """Past lookback close-to-close sample covariance; no future rows."""
        if self.risk_aversion == 0 or self.lookback < 2:
            return 0.0
        closes = self.panel.close[self.t - self.lookback:self.t + 1]
        returns = closes[1:] / closes[:-1] - 1
        covariance = np.atleast_2d(np.cov(returns, rowvar=False, ddof=1))
        variance = float(target[1:] @ covariance @ target[1:])
        return self.risk_aversion * max(variance, 0.0)
