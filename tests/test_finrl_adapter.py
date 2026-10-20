"""Integration checks for the real, pinned FinRL training boundary."""
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces


class TinyPortfolioEnv(gym.Env):
    """Deterministic trainable fixture; no market-data or network dependency."""

    def __init__(self):
        self.observation_space = spaces.Dict({
            "prices": spaces.Box(-5, 5, shape=(3, 2, 8), dtype=np.float32),
            "weights": spaces.Box(0, 1, shape=(3,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        self.t = 0

    def observation(self):
        return {
            "prices": np.full((3, 2, 8), 0.1 + self.t / 100, dtype=np.float32),
            "weights": np.array([1, 0, 0], dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self.observation(), {}

    def step(self, action):
        self.t += 1
        reward = float(-np.square(action - np.array([0.2, -0.1, 0.6])).sum())
        return self.observation(), reward, self.t >= 12, False, {}


@pytest.mark.parametrize("algorithm", ["td3", "ddpg"])
def test_real_finrl_model_trains_and_roundtrips(tmp_path, algorithm):
    # Catches broken FinRL loading, disconnected features, and unusable models.
    from rlasset.finrl_adapter import make_agent
    from rlasset.extractors import SharedTemporalExtractor

    torch.set_num_threads(1)
    env = TinyPortfolioEnv()
    model = make_agent(env, algorithm=algorithm, seed=7, total_timesteps=24,
                       policy_kwargs={"features_extractor_class": SharedTemporalExtractor})
    assert model.rlasset_backend == "finrl"
    assert Path(model.rlasset_provenance["models_path"]).is_file()
    before = [p.detach().clone() for p in model.actor.parameters()]
    model.learn(total_timesteps=24)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.actor.parameters()))
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (3,)
    assert np.isfinite(action).all()
    model.save(tmp_path / algorithm)
    loaded = type(model).load(tmp_path / algorithm)
    restored, _ = loaded.predict(obs, deterministic=True)
    np.testing.assert_allclose(action, restored, atol=1e-7)


def test_shared_encoder_is_equivariant_and_keeps_current_holdings():
    # Catches asset-specific encoders or missing portfolio state.
    from rlasset.extractors import SharedTemporalExtractor

    env = TinyPortfolioEnv()
    extractor = SharedTemporalExtractor(env.observation_space, channels=6)
    prices = torch.randn(2, 3, 2, 8)
    weights = torch.tensor([[0.1, 0.3, 0.6], [0.7, 0.2, 0.1]])
    out = extractor({"prices": prices, "weights": weights})
    swapped = extractor({"prices": prices[:, :, [1, 0], :],
                         "weights": weights[:, [0, 2, 1]]})
    assert out.shape == (2, extractor.features_dim)
    torch.testing.assert_close(out[:, :12].reshape(2, 2, 6)[:, [1, 0]],
                               swapped[:, :12].reshape(2, 2, 6))
    torch.testing.assert_close(out[:, -3:], weights)
    out.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in extractor.parameters())


def test_sb3_backend_requires_an_explicit_name():
    from rlasset.finrl_adapter import make_agent

    model = make_agent(TinyPortfolioEnv(), backend="sb3", total_timesteps=10)
    assert model.rlasset_backend == "sb3"
    with pytest.raises(ValueError, match="backend"):
        make_agent(TinyPortfolioEnv(), backend="auto")
    with pytest.raises(ValueError, match="algorithm"):
        make_agent(TinyPortfolioEnv(), algorithm="unknown")


def test_extractor_rejects_inconsistent_portfolio_dimensions():
    from rlasset.extractors import SharedTemporalExtractor

    invalid_space = spaces.Dict({
        "prices": spaces.Box(-1, 1, shape=(3, 2, 8), dtype=np.float32),
        "weights": spaces.Box(0, 1, shape=(2,), dtype=np.float32),
    })
    with pytest.raises(ValueError, match="weights"):
        SharedTemporalExtractor(invalid_space)
