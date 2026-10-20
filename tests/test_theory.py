"""Independent checks of the stationary quadratic-control reference."""

import importlib

import numpy as np
import pytest
from scipy.linalg import solve_discrete_are


def theory():
    return importlib.import_module("rlasset.theory")


@pytest.mark.parametrize("discount,rho", [(0.0, 0.9), (0.6, 0.9), (0.99, 0.9), (1.0, 0.9), (1.0, -0.6)])
def test_feedback_maximizes_independent_matrix_bellman_objective(discount, rho):
    # A wrong discount/rho placement in the scalar solution breaks this check.
    feedback = theory().quadratic_feedback(rho=rho, cost=1.0, risk=0.3, discount=discount)
    transition = np.diag([1.0, rho])
    control = np.array([[1.0], [0.0]])
    state_cost = np.array([[0.3, -0.5], [-0.5, 0.0]])
    action_cost = np.array([[1.3]])
    cross_cost = np.array([[0.3], [-0.5]])
    value = solve_discrete_are(np.sqrt(discount) * transition,
                               np.sqrt(discount) * control,
                               state_cost, action_cost, s=cross_cost)

    def bellman_cost(state, trade):
        next_state = transition @ state + control[:, 0] * trade
        return (state @ state_cost @ state + 1.3 * trade**2
                + 2 * trade * float(state @ cross_cost[:, 0])
                + discount * next_state @ value @ next_state)

    for state in (np.array([0.7, -1.1]), np.array([-1.2, 0.9]), np.array([0.0, 2.0])):
        trade = feedback.trade(position=state[0], signal=state[1])
        step = 1e-4
        slope = (bellman_cost(state, trade + step) - bellman_cost(state, trade - step)) / (2 * step)
        assert abs(slope) < 1e-8
        assert bellman_cost(state, trade) < bellman_cost(state, trade + 0.1)
        assert bellman_cost(state, trade) == pytest.approx(float(state @ value @ state), abs=1e-9)


def test_no_cost_is_markowitz_and_zero_discount_is_myopic():
    # These limiting cases catch incorrect friction or risk normalization.
    free = theory().quadratic_feedback(cost=0, risk=0.25)
    assert free.target(position=20, signal=0.5) == pytest.approx(1.0)
    myopic = theory().quadratic_feedback(cost=1, risk=0.5, discount=0)
    assert myopic.target(position=2, signal=2) == pytest.approx(2.0)


@pytest.mark.parametrize("kwargs", [
    {"rho": 1.0}, {"rho": -1.0}, {"rho": np.nan},
    {"cost": -1}, {"risk": 0}, {"discount": 1.01}, {"cost": np.inf},
])
def test_feedback_rejects_parameters_outside_stationary_model(kwargs):
    with pytest.raises(ValueError):
        theory().quadratic_feedback(**kwargs)


def test_paired_seeded_reference_beats_myopic_in_persistent_signal_model():
    # This fails if the evaluator looks ahead, reverses costs, or swaps policies.
    report = theory().evaluate_quadratic_benchmark(seed=7, horizon=5000, episodes=12)
    scores = report["policies"]
    assert scores["analytic"]["mean_reward"] > scores["myopic"]["mean_reward"] + 0.03
    assert scores["myopic"]["mean_reward"] > scores["zero"]["mean_reward"]
    assert scores["zero"]["mean_reward"] == 0
    assert report == theory().evaluate_quadratic_benchmark(seed=7, horizon=5000, episodes=12)
    assert report["objective"] == "stationary_average_reward"


def test_discount_zero_scores_only_first_step_and_includes_transaction_cost():
    report = theory().evaluate_quadratic_benchmark(seed=7, horizon=10, episodes=4, discount=0)
    # With gamma=0 the optimal and myopic policies are identical.
    assert report["policies"]["analytic"] == report["policies"]["myopic"]
    assert report["objective"] == "discounted_infinite_horizon"
    for score in report["policies"].values():
        assert score["mean_reward"] == pytest.approx(
            score["mean_pnl"] - 0.3 * score["mean_squared_position"])


@pytest.mark.parametrize("kwargs", [{"episodes": 0}, {"horizon": 0}, {"return_noise_std": -1}])
def test_benchmark_rejects_invalid_sampling_settings(kwargs):
    with pytest.raises(ValueError):
        theory().evaluate_quadratic_benchmark(**kwargs)
