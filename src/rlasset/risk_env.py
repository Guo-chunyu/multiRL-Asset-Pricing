"""Compact causal market information and a long-only volatility-budget action.

The two switches support a 2x2 ablation against RobustPortfolioEnv. Volatility
is an estimate from completed returns, not a guarantee on realized volatility
or drawdown. No leverage, price prediction, or future fitted scaling is used.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from .corporate_actions import CorporateActions
from .robust_env import RobustPortfolioEnv, capped_risky_weights


class InformationRiskEnv(RobustPortfolioEnv):
    """Add 3 features per asset, 4 market features and 3 account state values.

    New asset features, in asset-major order, are split-consistent volume
    surprise, downside semivolatility and volatility-scaled relative momentum.
    Market features are mean correlation, three-ETF trend participation,
    current 120-day basket drawdown and the short/long volatility ratio.
    Account values are current drawdown, spendable cash/NAV and receivable/NAV.

    Risk-controlled actions select annual risk budgets between 5% and 20%,
    centered on 10%, then scale a capped risky composition by a conservative
    20/60-day covariance estimate. Target exposure can range from zero to one;
    the actual lower bound depends on observed volatility. Direct benchmark
    targets retain the parent's validation and weekly execution schedule.
    """

    BASE_FEATURES = (
        "momentum_1", "momentum_5", "momentum_20", "momentum_60", "momentum_120",
        "volatility_20", "volatility_60", "ma_deviation_60",
    )
    EXTRA_ASSET_FEATURES = (
        "volume_surprise_60", "downside_vol_20", "relative_momentum_20",
    )
    EXTRA_MARKET_FEATURES = (
        "mean_correlation_60", "breadth_ma60", "market_drawdown_120",
        "market_vol_ratio_20_60",
    )

    def __init__(
        self, panel, start, end, training=False, episode_length=126,
        rebalance_every=5, cost_bps=5, risk_aversion=2, actions=None,
        enhanced_features=False, risk_control=False,
    ):
        for name, flag in (("enhanced_features", enhanced_features),
                           ("risk_control", risk_control)):
            if not isinstance(flag, (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
        self.enhanced_features = bool(enhanced_features)
        self.risk_control = bool(risk_control)
        self._peak_value = 1.
        actions = actions if actions is not None else CorporateActions.empty(panel)
        # Build row t from prefixes ending at t only. Precomputation changes
        # runtime cost, never which information is available at a decision.
        self._build_market_cache(panel, actions)
        super().__init__(panel, start, end, training=training,
                         episode_length=episode_length, rebalance_every=rebalance_every,
                         cost_bps=cost_bps, risk_aversion=risk_aversion, actions=actions)
        if self.enhanced_features:
            self.observation_space = spaces.Box(
                -np.finfo(np.float32).max, np.finfo(np.float32).max,
                shape=(12 * self.n_assets + 9,), dtype=np.float32,
            )

    def _build_market_cache(self, panel, actions):
        close = actions.signal_close(panel)
        days, assets = close.shape
        log_returns = np.zeros_like(close)
        log_returns[1:] = np.log(close[1:] / close[:-1])
        split_units = np.cumprod(actions.split_ratio, axis=0)
        volume = panel.volume / split_units
        # This is a feature index for a frictionless daily equal-weight basket,
        # not the net returns of the actual account or a backtest benchmark.
        basket = np.ones(days)
        basket[1:] = np.cumprod(np.mean(close[1:] / close[:-1], axis=1))
        basket_returns = np.zeros(days)
        basket_returns[1:] = np.log(basket[1:] / basket[:-1])
        self._base_market = np.zeros((days, 8 * assets))
        self._extra_market = np.zeros((days, 3 * assets + 4))
        self._covariances = {h: np.zeros((days, assets, assets)) for h in (20, 60)}
        upper = np.triu_indices(assets, k=1)
        for t in range(120, days):
            base = [10 * np.log(close[t] / close[t-h]) / np.sqrt(h)
                    for h in self.MOMENTUM_HORIZONS]
            for h in (20, 60):
                sample = log_returns[t-h+1:t+1]
                base.append(100 * sample.std(axis=0, ddof=1))
                covariance = np.atleast_2d(np.cov(sample, rowvar=False, ddof=1))
                self._covariances[h][t] = .75 * covariance + .25 * np.diag(np.diag(covariance))
            moving_mean = close[t-59:t+1].mean(axis=0)
            base.append(10 * (close[t] / moving_mean - 1))
            self._base_market[t] = np.stack(base, axis=1).reshape(-1)
            volume_surprise = np.clip(np.log(volume[t] / volume[t-60:t].mean(axis=0)), -5, 5)
            short = log_returns[t-19:t+1]
            long = log_returns[t-59:t+1]
            downside = np.sqrt(252 * np.mean(np.minimum(short, 0) ** 2, axis=0))
            momentum = np.log(close[t] / close[t-20])
            relative = np.clip((momentum - momentum.mean()) /
                               (np.maximum(long.std(axis=0, ddof=1), .001) * np.sqrt(20)), -5, 5)
            covariance = np.atleast_2d(np.cov(long, rowvar=False, ddof=1))
            deviations = np.sqrt(np.maximum(np.diag(covariance), 0))
            denominators = np.outer(deviations, deviations)
            correlations = np.divide(covariance, denominators, out=np.zeros_like(covariance),
                                     where=denominators > 1e-20)
            mean_correlation = float(correlations[upper].mean()) if len(upper[0]) else 0.
            breadth = float(np.mean(close[t] > moving_mean))
            drawdown = 1 - basket[t] / basket[t-119:t+1].max()
            short_vol = basket_returns[t-19:t+1].std(ddof=1)
            long_vol = basket_returns[t-59:t+1].std(ddof=1)
            vol_ratio = np.clip(short_vol / max(long_vol, .0001) - 1, -5, 5)
            self._extra_market[t] = np.r_[
                np.stack([volume_surprise, downside, relative], axis=1).reshape(-1),
                mean_correlation, breadth, drawdown, vol_ratio,
            ]

    def _validate_index(self, t):
        if (isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer))
                or not 120 <= t < len(self._base_market)):
            raise ValueError("Market feature index must be an integer with 120 days of history")
        return int(t)

    def market_features(self, t, enhanced=True):
        """Return a copy of account-independent features known at close[t]."""
        t = self._validate_index(t)
        if enhanced:
            return np.r_[self._base_market[t], self._extra_market[t]]
        return self._base_market[t].copy()

    def market_feature_names(self, enhanced=True):
        names = [f"{ticker}:{name}" for ticker in self.panel.tickers for name in self.BASE_FEATURES]
        if enhanced:
            names += [f"{ticker}:{name}" for ticker in self.panel.tickers for name in self.EXTRA_ASSET_FEATURES]
            names += list(self.EXTRA_MARKET_FEATURES)
        return names

    def estimated_covariance(self, t, window):
        """Daily log-return covariance, with 25% fixed diagonal shrinkage."""
        t = self._validate_index(t)
        if window not in (20, 60):
            raise ValueError("Risk covariance window must be 20 or 60")
        return self._covariances[window][t].copy()

    def estimated_risk(self, risky_mix, t=None):
        """Annualized full-investment volatility with a 1% numerical floor."""
        q = np.asarray(risky_mix, dtype=float)
        if (q.shape != (self.n_assets,) or not np.isfinite(q).all() or (q < 0).any()
                or not np.isclose(q.sum(), 1, atol=1e-7, rtol=0)):
            raise ValueError("risky_mix must be a finite nonnegative simplex")
        t = self._validate_index(self.t if t is None else t)
        return max(.01, *(np.sqrt(max(252 * float(q @ self._covariances[h][t] @ q), 0.))
                           for h in (20, 60)))

    def risk_target(self, scores, budget=.10):
        """Pure target calculation; does not change account or diagnostic state."""
        if not np.isfinite(budget) or budget <= 0:
            raise ValueError("Annual risk budget must be finite and positive")
        q = capped_risky_weights(scores, 1., self.max_weight)
        sigma = self.estimated_risk(q)
        exposure = min(1., float(budget) / sigma)
        return np.r_[1 - exposure, exposure * q]

    def action_to_weights(self, action):
        if not self.risk_control:
            return super().action_to_weights(action)
        action = np.asarray(action, dtype=float)
        if (action.shape != (self.n_assets + 1,) or not np.isfinite(action).all()
                or (np.abs(action) > 1).any()):
            raise ValueError("action must be a finite cash-first vector in [-1, 1]")
        budget = .10 * np.exp(np.log(2) * action[0])
        return self.risk_target(np.exp(.75 * action[1:]), budget)

    def _observation(self):
        if not self.enhanced_features:
            return super()._observation()
        peak = max(self._peak_value, self.value)
        account = [1 - self.value / peak, self.cash / self.value, self.receivable / self.value]
        phase = self.rebalance_phase / max(self.rebalance_every - 1, 1)
        return np.r_[self.market_features(self.t), self.weights, phase, account].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        self._peak_value = 1.
        return super().reset(seed=seed, options=options)

    def step_weights(self, weights):
        decision_t = self.t
        observation, reward, terminal, truncated, info = super().step_weights(weights)
        self._peak_value = max(self._peak_value, self.value)
        if self.risk_control:
            diagnostics = {name: None for name in (
                "requested_exposure", "estimated_risky_vol", "requested_target_vol", "requested_risky_mix",
            )}
            if weights is not None:
                target = np.asarray(weights, dtype=float)
                exposure = float(target[1:].sum())
                q = target[1:] / exposure if exposure > 0 else np.zeros(self.n_assets)
                sigma = self.estimated_risk(q, decision_t) if exposure > 0 else .01
                diagnostics.update(requested_exposure=exposure, estimated_risky_vol=sigma,
                                   requested_target_vol=exposure*sigma, requested_risky_mix=q.tolist())
            info.update(diagnostics)
            self.history[-1].update(diagnostics)
        return observation, reward, terminal, truncated, info
