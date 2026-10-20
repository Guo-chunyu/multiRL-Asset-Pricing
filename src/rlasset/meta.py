"""Frozen strategy pool and causal, discrete MetaTrader-inspired selection."""
from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class Candidate:
    name: str
    predictor: Callable

    def weights(self, env, observation):
        return self.predictor(env, observation)


def model_candidate(name, model):
    def predict(env, observation):
        action, _ = model.predict(observation, deterministic=True)
        return env.action_to_weights(action)
    return Candidate(name, predict)


def capped_weights(scores, cap):
    scores = np.asarray(scores, dtype=float)
    risky = np.minimum(scores / scores.sum(), cap)
    # A theoretically unit risky sum can round to 1 + 2e-16. Scale only
    # downward (preserving the cap) and clamp the residual to keep cash long-only.
    risky /= max(1.0, float(risky.sum()))
    return np.r_[max(0.0, 1.0 - float(risky.sum())), risky]


def baseline_candidates():
    def equal(env, obs):
        return capped_weights(np.ones(len(env.panel.tickers)), env.max_weight)

    def trend(env, obs, direction):
        history = np.asarray(obs["prices"])[0]
        momentum = history[:, -1] / history[:, 0] - 1
        scores = np.exp(np.clip(direction * momentum * 10, -10, 10))
        return capped_weights(scores, env.max_weight)

    return {
        "cash": Candidate("cash", lambda env, obs: np.r_[1., np.zeros(len(env.panel.tickers))]),
        "equal_weight": Candidate("equal_weight", equal),
        "buy_hold": Candidate("buy_hold", lambda env, obs: equal(env, obs) if env.t == env.start else None),
        "momentum": Candidate("momentum", lambda env, obs: trend(env, obs, 1)),
        "reversal": Candidate("reversal", lambda env, obs: trend(env, obs, -1)),
    }


def mixture_candidate(candidates):
    return Candidate("equal_mixture", lambda env, obs: np.mean([x.weights(env, obs) for x in candidates], axis=0))


class SelectorEnv(gym.Env):
    """One DQN action selects one frozen policy. Shadow ledgers earn their own returns.

    Each shadow is evaluated using its own current weights, while the selected
    policy receives the actual portfolio's weights when proposing the real action.
    The three performance features per candidate are trailing mean return,
    volatility and cumulative log return; all are realized at or before t.
    """
    metadata = {"render_modes": []}

    def __init__(self, env_factory, candidates, performance_window=20):
        super().__init__()
        if not candidates or performance_window < 1:
            raise ValueError("Nonempty policy pool and positive window required")
        self.candidates = candidates
        self.base = env_factory()
        self.shadows = [env_factory() for _ in candidates]
        self.performance_window = performance_window
        self.performance_history = [[] for _ in candidates]
        self.action_space = spaces.Discrete(len(candidates))
        self.observation_space = spaces.Dict({**self.base.observation_space.spaces, "performance": spaces.Box(-np.inf, np.inf, shape=(3 * len(candidates),), dtype=np.float32)})

    def _augment(self, observation):
        features = []
        for history in self.performance_history:
            recent = np.asarray(history[-self.performance_window:], dtype=float)
            features.extend([recent.mean() * 100, recent.std() * 100, np.log1p(recent).sum() * 10] if len(recent) else [0., 0., 0.])
        return {**observation, "performance": np.asarray(features, dtype=np.float32)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.performance_history = [[] for _ in self.candidates]
        self.current_obs, info = self.base.reset(seed=seed)
        self.shadow_obs = [shadow.reset(seed=seed)[0] for shadow in self.shadows]
        return self._augment(self.current_obs), info

    def step(self, action):
        array = np.asarray(action)
        if array.size != 1 or not np.isfinite(array).all() or float(array.item()) != int(array.item()):
            raise ValueError("Selector action must be an integer")
        index = int(array.item())
        if not self.action_space.contains(index):
            raise ValueError("Selector action outside candidate pool")
        candidate = self.candidates[index]
        target = candidate.weights(self.base, self.current_obs)
        self.current_obs, reward, terminated, truncated, info = self.base.step_weights(target)
        for i, (policy, shadow) in enumerate(zip(self.candidates, self.shadows)):
            shadow_target = policy.weights(shadow, self.shadow_obs[i])
            self.shadow_obs[i], _, shadow_done, _, shadow_info = shadow.step_weights(shadow_target)
            if shadow_done != terminated:
                raise RuntimeError("Shadow/actual calendar mismatch")
            self.performance_history[i].append(shadow_info["net_return"])
        info = {**info, "selected_policy": candidate.name, "selected_index": index}
        self.base.history[-1].update({"selected_policy": candidate.name, "selected_index": index})
        return self._augment(self.current_obs), reward, terminated, truncated, info
