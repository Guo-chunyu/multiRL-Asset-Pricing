"""v5: a causal 10% volatility anchor with small, bounded policy residuals.

This is a research hypothesis, not an alpha or realized-risk guarantee. The
optional fixed 60/120-close trend gate cannot be overridden by the residual.
Accounting, next-open execution, and weekly cadence are inherited unchanged.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from .corporate_actions import CorporateActions
from .risk_env import InformationRiskEnv
from .robust_env import capped_risky_weights


def _gate(close, t):
    return float(np.mean([
        close[t] > close[t-h+1:t+1].mean(axis=0) for h in (60, 120)
    ]))


def _target(covariances, gate, action, assets):
    action = np.asarray(action, dtype=float)
    if (action.shape != (assets + 1,) or not np.isfinite(action).all()
            or (np.abs(action) > 1).any()):
        raise ValueError("action must be a finite cash-first vector in [-1, 1]")
    q = capped_risky_weights(np.exp(.10 * action[1:]), 1., .5)
    sigma = max(.01, *(np.sqrt(max(252 * float(q @ c @ q), 0.)) for c in covariances))
    budget = .10 * (1 + .25 * action[0])
    exposure = gate * min(1., budget / sigma)
    return np.r_[1 - exposure, exposure * q]


def trend_volatility_target(signal_close, t, trend_gate=True):
    """Standalone zero-residual rule from a causal total-return index.

    Reads only the prefix through decision close ``t``; callers must provide
    an ex-date total-return index, e.g. ``actions.signal_close(panel)``. Future
    columns/rows are never used to fit or normalize the decision. Gate-off is
    the v4 equal-weight 10% volatility rule. This function does not trade.
    """
    close = np.asarray(signal_close, dtype=float)
    if (close.ndim != 2 or close.shape[1] < 2
            or isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer))
            or not 120 <= t < len(close)):
        raise ValueError("Require a 2D total-return index, >=2 assets and integer t >=120")
    if not isinstance(trend_gate, (bool, np.bool_)):
        raise ValueError("trend_gate must be boolean")
    prefix = close[t-120:t+1]
    if not np.isfinite(prefix).all() or (prefix <= 0).any():
        raise ValueError("Total-return history must be finite and positive")
    returns = np.log(prefix[1:] / prefix[:-1])
    covariances = []
    for h in (20, 60):
        sample = np.atleast_2d(np.cov(returns[-h:], rowvar=False, ddof=1))
        covariances.append(.75 * sample + .25 * np.diag(np.diag(sample)))
    gate = _gate(close, t) if trend_gate else 1.
    return _target(covariances, gate, np.zeros(close.shape[1] + 1), close.shape[1])


class ResidualRiskEnv(InformationRiskEnv):
    """Bounded risk-budget/asset-tilt residual around an explicit rule anchor.

    a0 maps linearly to 7.5--12.5% annual risk budget. Asset scores exp(.10*ai)
    have pairwise ratio at most exp(.20), redistributed under the 50% risky-mix
    cap. Trend participation is the average of 2*N strict close-above-MA
    indicators. Exposure is g*min(1,b/sigma), not min(1,g*b/sigma).

    Observation order: existing 8*N market factors, N+1 current cash-first
    weights, normalized weekly phase, N+1 anchor weights, effective gate.
    Same 10*N+4 shape with/without gating; no fitted scaler or v4 extra inputs.
    Direct step_weights benchmarks retain the inherited capped-simplex rules;
    only learned actions are constrained to the residual family.
    """

    def __init__(self, panel, start, end, training=False, episode_length=126,
                 rebalance_every=5, cost_bps=5, risk_aversion=2, actions=None,
                 trend_gate=True):
        if not isinstance(trend_gate, (bool, np.bool_)):
            raise ValueError("trend_gate must be boolean")
        self.trend_gate = bool(trend_gate)
        actions = actions if actions is not None else CorporateActions.empty(panel)
        close = actions.signal_close(panel)
        self._trend_participation = np.zeros(len(close))
        for t in range(120, len(close)):
            self._trend_participation[t] = _gate(close, t)
        super().__init__(panel, start, end, training=training,
                         episode_length=episode_length, rebalance_every=rebalance_every,
                         cost_bps=cost_bps, risk_aversion=risk_aversion, actions=actions,
                         enhanced_features=False, risk_control=True)
        self.observation_space = spaces.Box(
            -np.finfo(np.float32).max, np.finfo(np.float32).max,
            shape=(10 * self.n_assets + 4,), dtype=np.float32,
        )

    def trend_participation(self, t=None):
        """Effective gate; disabled mode returns 1, even when trend is bearish."""
        t = self._validate_index(self.t if t is None else t)
        return float(self._trend_participation[t]) if self.trend_gate else 1.

    def anchor_target(self, t=None):
        """Pure zero-residual target; no account or diagnostic mutation."""
        t = self._validate_index(self.t if t is None else t)
        return _target([self._covariances[h][t] for h in (20, 60)],
                       self.trend_participation(t), np.zeros(self.n_assets + 1), self.n_assets)

    def action_to_weights(self, action):
        return _target([self._covariances[h][self.t] for h in (20, 60)],
                       self.trend_participation(), action, self.n_assets)

    def _observation(self):
        phase = self.rebalance_phase / max(self.rebalance_every - 1, 1)
        return np.r_[self.market_features(self.t, enhanced=False), self.weights, phase,
                     self.anchor_target(), self.trend_participation()].astype(np.float32)

    def step_weights(self, weights):
        gate, anchor = self.trend_participation(), self.anchor_target()
        observation, reward, terminal, truncated, info = super().step_weights(weights)
        diagnostics = dict(anchor_target_weights=anchor.tolist(), trend_participation=gate,
                           trend_gate_enabled=self.trend_gate)
        info.update(diagnostics)
        self.history[-1].update(diagnostics)
        return observation, reward, terminal, truncated, info
