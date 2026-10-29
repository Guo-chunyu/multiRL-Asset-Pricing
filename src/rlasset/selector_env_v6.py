"""v6 frozen-expert selection with independent, fully funded raw-price ledgers.

No realized volatility/drawdown ceiling is enforced. This is a structural
research hypothesis, not an alpha guarantee. Earlier environments are unchanged.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .residual_env import trend_volatility_target
from .robust_env import RobustPortfolioEnv

EXPERT_NAMES = ("cash", "equal50", "vol10", "trend10")


class FixedExpertSelectorEnv(gym.Env):
    """Daily discrete requests; next-open execution only every fifth decision.

    ``equal50`` means 50% cash and 50% aggregate equity spread equally
    across assets, independently of the existing 50%-per-asset target cap. Four independent shadows always follow their
    own fixed rule, paying their own costs; actual switching pays actual costs.

    Flat observation order: existing base market/weights/phase, four cash-first
    current targets, each shadow's trailing 20 net-return mean/std (population)
    and cumulative log return, available count (0..20), previous executed expert
    one-hot (all zero before execution or for a direct target), current drawdown,
    episode running maximum drawdown, liquid cash/NAV, receivable/NAV. Performance
    features are unscaled fractions/log units. No hidden performance warmup.
    """
    metadata = {"render_modes": []}
    expert_names = EXPERT_NAMES
    performance_window = 20

    def __init__(self, panel, start, end, training=False, episode_length=126,
                 rebalance_every=5, cost_bps=5, actions=None,
                 drawdown_penalty=True):
        super().__init__()
        if not isinstance(drawdown_penalty, (bool, np.bool_)):
            raise ValueError("drawdown_penalty must be boolean")
        if rebalance_every != 5:
            raise ValueError("v6 protocol requires rebalance_every=5")
        self.drawdown_penalty = bool(drawdown_penalty)
        self.objective = ("net_log_return_minus_half_max_drawdown" if self.drawdown_penalty
                          else "net_log_return")
        kwargs = dict(training=training, episode_length=episode_length,
                      rebalance_every=rebalance_every, cost_bps=cost_bps,
                      risk_aversion=0, actions=actions)
        self.base = RobustPortfolioEnv(panel, start, end, **kwargs)
        self.shadows = [RobustPortfolioEnv(panel, start, end, **kwargs) for _ in EXPERT_NAMES]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-np.finfo(np.float32).max,
                                           np.finfo(np.float32).max,
                                           shape=(13 * self.base.n_assets + 27,),
                                           dtype=np.float32)
        self.reset()

    @property
    def t(self):
        return self.base.t

    @property
    def history(self):
        return self.base.history

    @property
    def value(self):
        return self.base.value

    @property
    def rebalance_phase(self):
        return self.base.rebalance_phase

    def fixed_expert_action(self, expert):
        """Resolve an expert name/index without trading or changing state."""
        if isinstance(expert, str):
            if expert not in EXPERT_NAMES:
                raise ValueError("Unknown fixed expert")
            return EXPERT_NAMES.index(expert)
        array = np.asarray(expert)
        if (array.size != 1 or array.dtype.kind not in "iu" or array.dtype.kind == "b"
                or not self.action_space.contains(int(array.item()))):
            raise ValueError("Expert action must be an integer in [0, 3]")
        return int(array.item())

    def expert_targets(self, t=None):
        """Pure causal targets at decision close t; returned arrays are copies."""
        t = self.t if t is None else t
        n = self.base.n_assets
        close = self.base.signal_close
        return np.stack([np.r_[1., np.zeros(n)], np.r_[.5, np.full(n, .5/n)],
                         trend_volatility_target(close, t, False),
                         trend_volatility_target(close, t, True)])

    def equal_target_mix(self):
        """Average targets, NOT average shadow returns; trade via step_weights."""
        return self.expert_targets().mean(axis=0)

    def trailing_argmax(self):
        """Trailing net log-return winner; exact ties favor fixed expert order."""
        scores = [np.log1p(h[-20:]).sum() for h in self.performance_history]
        return int(np.argmax(scores))

    def _assert_calendar(self):
        actual = (self.base.t, self.base.start, self.base.end, self.base.rebalance_phase,
                  self.base.training, self.base._terminated)
        for shadow in self.shadows:
            if (shadow.t, shadow.start, shadow.end, shadow.rebalance_phase,
                    shadow.training, shadow._terminated) != actual:
                raise RuntimeError("Shadow/actual calendar mismatch")

    def _observation(self):
        performance = []
        for history in self.performance_history:
            recent = np.asarray(history[-20:])
            performance.extend([recent.mean(), recent.std(ddof=0), np.log1p(recent).sum()]
                               if len(recent) else [0., 0., 0.])
        previous = np.zeros(4)
        if self.previous_executed_expert is not None:
            previous[self.previous_executed_expert] = 1
        return np.r_[self.base._observation(), self.expert_targets().ravel(), performance,
                     min(len(self.performance_history[0]), 20), previous,
                     self.current_drawdown, self.max_drawdown,
                     self.base.cash/self.value, self.base.receivable/self.value].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        _, info = self.base.reset(seed=seed, options=options)
        # Pin every shadow to the sampled actual fragment, even for seed=None.
        # Keep training=True so artificial endings remain truncations on both.
        for shadow in self.shadows:
            shadow._range_start, shadow._range_end = self.base.start, self.base.end
            shadow.reset(seed=seed, options=options)
        self.performance_history = [[] for _ in EXPERT_NAMES]
        self.previous_executed_expert = None
        self.peak_value = 1.
        self.current_drawdown = self.max_drawdown = 0.
        self._assert_calendar()
        info.update(objective=self.objective, drawdown_penalty_enabled=self.drawdown_penalty,
                    previous_executed_expert=None, current_drawdown=0., max_drawdown=0.)
        self.history[-1].update(info)
        return self._observation(), info

    def step(self, action):
        index = self.fixed_expert_action(action)
        return self._step(self.expert_targets()[index], index)

    def step_weights(self, target):
        """Benchmark hook: execute an explicit target in the ACTUAL account."""
        return self._step(target, None)

    def _step(self, target, requested_index):
        self._assert_calendar()
        targets = self.expert_targets()
        old_max = self.max_drawdown
        _, _, terminated, truncated, info = self.base.step_weights(target)
        for index, shadow in enumerate(self.shadows):
            _, _, shadow_term, shadow_trunc, shadow_info = shadow.step_weights(targets[index])
            if (shadow_term, shadow_trunc) != (terminated, truncated):
                raise RuntimeError("Shadow/actual terminated or truncated mismatch")
            self.performance_history[index].append(shadow_info["net_return"])
        self._assert_calendar()
        executed = info["rebalance_executed"]
        if executed:
            self.previous_executed_expert = requested_index
        self.peak_value = max(self.peak_value, self.value)
        self.current_drawdown = max(0., 1 - self.value/self.peak_value)
        self.max_drawdown = max(old_max, self.current_drawdown)
        increment = self.max_drawdown - old_max
        log_return = float(np.log1p(info["net_return"]))
        penalty = .5 * increment if self.drawdown_penalty else 0.
        reward = 100 * (log_return - penalty)
        diagnostics = dict(
            requested_expert_index=requested_index,
            requested_expert=EXPERT_NAMES[requested_index] if requested_index is not None else None,
            executed_expert_index=requested_index if executed else None,
            executed_expert=EXPERT_NAMES[requested_index] if executed and requested_index is not None else None,
            action_ignored=bool(not executed), previous_executed_expert=self.previous_executed_expert,
            current_drawdown=self.current_drawdown, max_drawdown=self.max_drawdown,
            max_drawdown_increment=increment, drawdown_penalty=penalty,
            drawdown_penalty_enabled=self.drawdown_penalty, objective=self.objective,
            log_net_return=log_return, reward=reward,
        )
        info.update(diagnostics)
        self.history[-1].update(diagnostics)
        return self._observation(), float(reward), terminated, truncated, info


# Explicit versioned spelling for runner imports.
SelectorEnvV6 = FixedExpertSelectorEnv
