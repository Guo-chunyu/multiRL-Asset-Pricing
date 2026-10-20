"""Auditable one-asset quadratic reference for DDPO, not a market-return model.

The state is (current position, observable AR(1) predictor). A target position
earns predictor * target minus quadratic trade cost and quadratic position risk.
See docs/research/ddpo.md for the derivation and its scope.
"""

from dataclasses import asdict, dataclass
from math import isfinite, sqrt

import numpy as np


@dataclass(frozen=True)
class QuadraticFeedback:
    """Stationary unconstrained target-position coefficients."""

    position_coef: float
    signal_coef: float
    rho: float
    cost: float
    risk: float
    discount: float

    def target(self, position, signal):
        """Target holding; supports scalars and NumPy arrays."""
        return self.position_coef * position + self.signal_coef * signal

    def trade(self, position, signal):
        """Increment in holding, which is the action in the DDPO paper."""
        return self.target(position, signal) - position


def quadratic_feedback(*, rho=0.9, cost=1.0, risk=0.3, discount=1.0):
    """Solve the stationary scalar Riccati equation in a numerically stable form.

    Assumptions: abs(rho) < 1, cost >= 0, risk > 0, 0 <= discount <= 1;
    zero-mean finite-variance independent innovations; unrestricted positions.
    discount < 1 solves the discounted infinite-horizon problem. discount == 1
    solves the stationary average-reward problem, not a finite-horizon problem.
    """
    values = (rho, cost, risk, discount)
    if not all(isfinite(v) for v in values):
        raise ValueError("Parameters must be finite")
    if not -1 < rho < 1:
        raise ValueError("Stationary predictor requires abs(rho) < 1")
    if cost < 0 or risk <= 0:
        raise ValueError("cost must be nonnegative and risk must be positive")
    if not 0 <= discount <= 1:
        raise ValueError("discount must be in [0, 1]")
    # V(x,p) = -A*x**2 + B*x*p + C*p**2 + constant. Rationalizing the
    # positive root avoids cancellation and also handles discount == 0.
    b = risk + cost * (1 - discount)
    a = 2 * cost * risk / (b + sqrt(b * b + 4 * discount * cost * risk))
    h = cost + risk + discount * a
    position_coef = cost / h
    signal_coef = 1 / (2 * h * (1 - discount * rho * position_coef))
    return QuadraticFeedback(position_coef, signal_coef, rho, cost, risk, discount)


def evaluate_quadratic_benchmark(*, seed=42, horizon=5000, episodes=16,
                                 rho=0.9, cost=1.0, risk=0.3, discount=1.0,
                                 return_noise_std=0.0):
    """Compare analytic, one-step myopic, and zero policies on paired paths.

    Predictor variance is exactly one in stationarity and p_0 is stationary.
    Return noise is unavailable to all policies. Scores are finite-sample
    diagnostics; no sample-path ranking or trained-RL optimality is guaranteed.
    Discounted scores truncate the infinite horizon at the supplied horizon.
    """
    analytic = quadratic_feedback(rho=rho, cost=cost, risk=risk, discount=discount)
    if (not isinstance(horizon, int) or not isinstance(episodes, int)
            or horizon <= 0 or episodes <= 0):
        raise ValueError("horizon and episodes must be positive integers")
    if not isfinite(return_noise_std) or return_noise_std < 0:
        raise ValueError("return_noise_std must be finite and nonnegative")
    signal_seed, return_seed = np.random.SeedSequence(seed).spawn(2)
    signal_rng = np.random.default_rng(signal_seed)
    predictors = np.empty((horizon, episodes))
    predictors[0] = signal_rng.standard_normal(episodes)
    for t in range(1, horizon):
        predictors[t] = rho * predictors[t - 1] + sqrt(1 - rho**2) * signal_rng.standard_normal(episodes)
    realized = predictors + return_noise_std * np.random.default_rng(return_seed).standard_normal(predictors.shape)
    policies = {
        "analytic": analytic,
        "myopic": quadratic_feedback(rho=rho, cost=cost, risk=risk, discount=0),
        "zero": None,
    }
    scores = {}
    weights = discount ** np.arange(horizon, dtype=float)
    for name, policy in policies.items():
        positions = np.zeros(episodes)
        rewards = np.empty_like(predictors)
        pnls = np.empty_like(predictors)
        square_positions = np.empty_like(predictors)
        square_trades = np.empty_like(predictors)
        for t in range(horizon):
            target = np.zeros(episodes) if policy is None else policy.target(positions, predictors[t])
            square_trades[t] = (target - positions)**2
            square_positions[t] = target**2
            pnls[t] = target * realized[t] - cost * square_trades[t]
            rewards[t] = pnls[t] - risk * square_positions[t]
            positions = target
        episode_means = rewards.mean(axis=0)
        scores[name] = {
            "mean_reward": float(rewards.mean()),
            "mean_pnl": float(pnls.mean()),
            "mean_squared_position": float(square_positions.mean()),
            "mean_squared_trade": float(square_trades.mean()),
            "mean_discounted_return": float((rewards * weights[:, None]).sum(axis=0).mean()),
            "episode_mean_rewards": episode_means.tolist(),
            "standard_error_mean_reward": float(episode_means.std(ddof=1) / sqrt(episodes)) if episodes > 1 else None,
        }
    return {
        "objective": "stationary_average_reward" if discount == 1 else "discounted_infinite_horizon",
        "parameters": {"seed": seed, "horizon": horizon, "episodes": episodes,
                       "rho": rho, "cost": cost, "risk": risk, "discount": discount,
                       "return_noise_std": return_noise_std},
        "analytic_feedback": asdict(analytic),
        "policies": scores,
        "scope": "One synthetic asset, unrestricted signed positions, exogenous observable AR(1) predictor; no learned agent in this comparison.",
    }
