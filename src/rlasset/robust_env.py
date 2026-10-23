"""Causal multi-scale allocation with an explicit 80--100% equity mandate.

The bounded logits tilt an equal-weight allocation across risky assets, rather than sending
asset-cap overflow into cash. Targets are executable only every fifth trading
day; other days preserve quantities. This changes the investment mandate, so
research must compare equally constrained, equally scheduled benchmarks.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .corporate_actions import CorporateActionPortfolioEnv


def capped_risky_weights(scores, exposure, cap=.5):
    """Allocate a fixed risky budget proportionally, redistributing cap excess.

    Scores are strictly positive preferences, not portfolio weights. Requiring
    enough aggregate capacity avoids silently converting equity into cash.
    """
    scores = np.asarray(scores, dtype=float)
    if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
            or np.any(scores <= 0)):
        raise ValueError("scores must be a nonempty finite positive vector")
    if (not np.isfinite(exposure) or not 0 <= exposure <= 1
            or not np.isfinite(cap) or not 0 < cap <= 1
            or len(scores) * cap < exposure - 1e-12):
        raise ValueError("exposure and cap must admit a feasible nonnegative allocation")
    scores = scores / scores.max()
    allocation = np.zeros(len(scores))
    active = np.ones(len(scores), dtype=bool)
    remaining = float(exposure)
    while active.any():
        active_indices = np.flatnonzero(active)
        proposed = remaining * scores[active] / scores[active].sum()
        saturated = proposed > cap
        if not saturated.any():
            allocation[active_indices] = proposed
            break
        capped_indices = active_indices[saturated]
        allocation[capped_indices] = cap
        active[capped_indices] = False
        remaining = exposure - allocation.sum()
    return allocation


class RobustPortfolioEnv(CorporateActionPortfolioEnv):
    """Daily accounting, a causal factor vector, and scheduled allocation actions.

    Per-asset features (asset-major) are 1/5/20/60/120-day log momentum scaled
    by ``10 / sqrt(horizon)``, 20/60-day sample log-return volatility scaled
    by 100, and deviation from the trailing 60-close mean scaled by 10.
    Current cash-first holdings and the normalized rebalance phase follow.
    No full-sample fitted statistics enter this observation.

    Training samples contiguous ``episode_length``-transition fragments inside
    the supplied range. Their artificial endings are Gymnasium truncations;
    evaluation always runs the complete supplied range with true termination.
    The first decision after each reset is a rebalance date. Direct targets may
    represent cash benchmarks; the 20% cash maximum applies to learned actions,
    not to drifted holdings or explicitly supplied benchmark targets.
    """

    MOMENTUM_HORIZONS = (1, 5, 20, 60, 120)

    def __init__(
        self, panel, start, end, training=False, episode_length=126,
        rebalance_every=5, cost_bps=5, risk_aversion=2, actions=None,
    ):
        for name, value in (("episode_length", episode_length),
                            ("rebalance_every", rebalance_every)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(training, (bool, np.bool_)):
            raise ValueError("training must be boolean")
        if not all(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                   for value in (start, end)):
            raise ValueError("start and end must be integers")
        if not 120 <= start < end < len(panel.dates):
            raise ValueError("Require 120 <= start < end < number of market dates")
        if len(panel.tickers) < 2:
            raise ValueError("At least two assets are required for a 0.5 cap and full equity exposure")
        if training and end - start < episode_length:
            raise ValueError("Training range must contain a complete episode_length fragment")
        self.training = bool(training)
        self.episode_length = int(episode_length)
        self.rebalance_every = int(rebalance_every)
        self._range_start = int(start)
        self._range_end = int(end)
        super().__init__(panel, start=start, end=end, lookback=120,
                         cost_bps=cost_bps, risk_aversion=risk_aversion,
                         max_weight=.5, reward_scale=100, actions=actions)
        self.observation_space = spaces.Box(
            -np.finfo(np.float32).max, np.finfo(np.float32).max,
            shape=(9 * self.n_assets + 2,), dtype=np.float32,
        )

    @property
    def rebalance_phase(self):
        return (self.t - self.start) % self.rebalance_every

    def _observation(self):
        closes = self.signal_close
        current = closes[self.t]
        features = [10 * np.log(current / closes[self.t - horizon]) / np.sqrt(horizon)
                    for horizon in self.MOMENTUM_HORIZONS]
        for horizon in (20, 60):
            history = closes[self.t - horizon:self.t + 1]
            returns = np.log(history[1:] / history[:-1])
            features.append(100 * returns.std(axis=0, ddof=1))
        moving_average = closes[self.t - 59:self.t + 1].mean(axis=0)
        features.append(10 * (current / moving_average - 1))
        factor_vector = np.stack(features, axis=1).reshape(-1)
        phase = self.rebalance_phase / max(self.rebalance_every - 1, 1)
        return np.r_[factor_vector, self.weights, phase].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if self.training:
            self.start = int(self.np_random.integers(
                self._range_start, self._range_end - self.episode_length + 1,
            ))
            self.end = self.start + self.episode_length
        else:
            self.start, self.end = self._range_start, self._range_end
        observation, info = super().reset(seed=None, options=options)
        info.update(episode_start_index=self.start, episode_end_index=self.end,
                    rebalance_phase=0)
        self.history[-1].update(info)
        return observation, info

    def action_to_weights(self, action):
        action = np.asarray(action, dtype=float)
        if (action.shape != (self.n_assets + 1,) or not np.isfinite(action).all()
                or np.any(action < -1) or np.any(action > 1)):
            raise ValueError("action must be a finite cash-first vector in [-1, 1]")
        cash = .1 * (action[0] + 1)
        logits = .75 * action[1:]
        scores = np.exp(logits - logits.max())
        allocation = capped_risky_weights(scores, 1 - cash, self.max_weight)
        return np.r_[cash, allocation]

    def step(self, action):
        return self.step_weights(self.action_to_weights(action))

    def step_weights(self, weights):
        if self._terminated:
            raise RuntimeError("Episode is terminated; call reset before another step")
        # Validate requests even on non-rebalance days; silently discarding an
        # invalid request makes the public API depend on an incidental phase.
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if (weights.shape != (self.n_assets + 1,) or not np.isfinite(weights).all()
                    or np.any(weights < 0) or not np.isclose(weights.sum(), 1, atol=1e-7, rtol=0)
                    or np.any(weights[1:] > self.max_weight + 1e-7)):
                raise ValueError("weights must form a finite nonnegative capped simplex")
        phase = self.rebalance_phase
        execute = phase == 0 and weights is not None
        observation, reward, done, _, info = super().step_weights(weights if execute else None)
        info.update(rebalance_phase=self.rebalance_phase, decision_phase=phase,
                    rebalance_executed=execute, episode_start_index=self.start,
                    episode_end_index=self.end)
        self.history[-1].update(info)
        return observation, reward, bool(done and not self.training), bool(done and self.training), info

    def hold(self):
        return self.step_weights(None)
