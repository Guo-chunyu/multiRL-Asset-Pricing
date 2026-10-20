"""Use the pinned, unmodified FinRL DRLAgent without optional broker imports.

FinRL's root __init__ eagerly imports live-trading and all data backends.
We register only its package search path, then import the actual source modules
normally. No agent, configuration, environment, or preprocessing implementation
is stubbed. Missing dependencies fail explicitly; SB3 is an opt-in backend.
"""
from __future__ import annotations

import hashlib
from contextlib import redirect_stdout
import importlib
import importlib.machinery
import io
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise


FINRL_COMMIT = "8511e7c15fbe1fbe5eb25e9f89932ac1ecd546a0"
FINRL_SOURCE_HASHES = {
    "finrl/agents/stablebaselines3/models.py": "e71707d0eaae0117fad36f9aa1be59da0186619db59111fe43372d92ad9510b0",
    "finrl/config.py": "48675567fd7b797d228b2d5729e7b00f824894a2303c8bb51ec62b9389c9b70e",
}
ALGORITHM_CLASSES = {"td3": TD3, "ddpg": DDPG, "sac": SAC, "ppo": PPO, "a2c": A2C}


def training_budget(algorithm: str, requested_steps: int, n_envs: int = 1) -> dict:
    """Describe expected SB3 environment steps, including on-policy rollouts.

    PPO uses the largest divisor <=100 (and >=2) of the requested per-env
    budget. Prime budgets >100 use 64-step rollouts, whose overshoot is explicit.
    A2C uses a divisor <=5. In the research runs 5,000/10,000 steps are exact.
    """
    if algorithm not in ALGORITHM_CLASSES:
        raise ValueError("algorithm must be td3, ddpg, sac, ppo, or a2c")
    if not isinstance(requested_steps, (int, np.integer)) or isinstance(requested_steps, bool) or requested_steps < 1:
        raise ValueError("training budget must be a positive integer")
    per_env = max(1, requested_steps // n_envs)
    rollout = None
    if algorithm == "ppo":
        if requested_steps < 2:
            raise ValueError("PPO training budget must contain at least two steps")
        rollout = next((n for n in range(min(100, per_env), 1, -1)
                        if per_env % n == 0), 64)
    elif algorithm == "a2c":
        rollout = next(n for n in range(min(5, per_env), 0, -1) if per_env % n == 0)
    quantum = n_envs * (rollout or 1)
    actual = math.ceil(requested_steps / quantum) * quantum
    return {"requested_steps": int(requested_steps), "expected_actual_steps": actual,
            "rollout_steps": rollout, "n_envs": n_envs}


def load_finrl_agent():
    """Return FinRL's actual DRLAgent and reproducible source provenance."""
    vendor = Path(__file__).resolve().parents[2] / "vendor" / "FinRL"
    package_dir = vendor / "finrl"
    if not (package_dir / "__init__.py").is_file():
        raise RuntimeError(
            "Pinned FinRL source is missing. Clone "
            "https://github.com/AI4Finance-Foundation/FinRL into vendor/FinRL "
            f"and checkout {FINRL_COMMIT}. backend='sb3' is an explicit alternative."
        )
    hashes = {}
    for relative, expected in FINRL_SOURCE_HASHES.items():
        # Canonical LF bytes make the check independent of git autocrlf.
        actual = hashlib.sha256((vendor / relative).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Pinned FinRL source hash mismatch: {relative}")
        hashes[relative] = actual
    try:
        commit = subprocess.run(
            ["git", "-C", str(vendor), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Cannot verify pinned FinRL checkout; git and its checkout are required") from exc
    if commit != FINRL_COMMIT:
        raise RuntimeError(f"Expected FinRL commit {FINRL_COMMIT}, found {commit}")

    if "finrl" in sys.modules:
        existing_paths = getattr(sys.modules["finrl"], "__path__", [])
        if package_dir.resolve() not in [Path(p).resolve() for p in existing_paths]:
            raise RuntimeError("Another FinRL package is already imported; start a fresh Python process")
    else:
        package = types.ModuleType("finrl")
        package.__path__ = [str(package_dir)]
        package.__package__ = "finrl"
        package.__spec__ = importlib.machinery.ModuleSpec("finrl", loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules["finrl"] = package
    try:
        module = importlib.import_module("finrl.agents.stablebaselines3.models")
    except ImportError as exc:
        raise RuntimeError(
            "FinRL's real SB3 interface could not load. Install the project dependencies "
            "including sb3-contrib==2.7.1, stockstats, and yfinance. "
            "No automatic SB3 fallback was performed."
        ) from exc
    expected_models = package_dir / "agents" / "stablebaselines3" / "models.py"
    if Path(module.__file__).resolve() != expected_models.resolve():
        raise RuntimeError("The loaded FinRL agent does not originate from the pinned checkout")
    provenance = {
        "repository": "https://github.com/AI4Finance-Foundation/FinRL",
        "commit": commit,
        "license": "MIT",
        "models_path": str(expected_models),
        "sha256_lf": hashes,
        "loader": "package path registration; root __init__ skipped; original modules executed",
    }
    return module.DRLAgent, provenance


def make_agent(
    env,
    algorithm: str = "td3",
    seed: int = 0,
    total_timesteps: int = 2000,
    learning_rate: float = 3e-4,
    policy: str = "MultiInputPolicy",
    policy_kwargs: dict | None = None,
    backend: str = "finrl",
):
    """Build an untrained continuous-control model through the chosen backend.

    The default executes ``FinRL.DRLAgent(env).get_model(...)``. ``total_timesteps``
    sizes warm-up or rollout lengths; the caller invokes ``learn`` separately.
    ``rlasset_backend`` and ``rlasset_provenance`` are retained in saved models.
    """
    if backend not in {"finrl", "sb3"}:
        raise ValueError("backend must be 'finrl' or explicitly 'sb3'")
    budget = training_budget(algorithm, total_timesteps, getattr(env, "num_envs", 1))
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    options = {"net_arch": [64, 64]}
    if policy_kwargs:
        options.update(policy_kwargs)
    model_kwargs = {"learning_rate": learning_rate, "device": "cpu"}
    if algorithm in {"td3", "ddpg", "sac"}:
        model_kwargs.update(learning_starts=min(100, total_timesteps // 5),
                            batch_size=64, buffer_size=50000, train_freq=1, gradient_steps=1)
    else:
        model_kwargs["n_steps"] = budget["rollout_steps"]
        if algorithm == "ppo":
            buffer_size = budget["rollout_steps"] * budget["n_envs"]
            model_kwargs["batch_size"] = next((n for n in range(min(64, buffer_size), 1, -1)
                                                 if buffer_size % n == 0), buffer_size)
    if backend == "finrl":
        agent_class, provenance = load_finrl_agent()
        if algorithm in {"td3", "ddpg"}:
            # FinRL resolves this string to NormalActionNoise(sigma=0.1).
            model_kwargs["action_noise"] = "normal"
        with redirect_stdout(io.StringIO()):
            model = agent_class(env).get_model(
                algorithm, policy=policy, policy_kwargs=options,
                model_kwargs=model_kwargs, seed=seed, verbose=0,
            )
    else:
        if algorithm in {"td3", "ddpg"}:
            n_actions = env.action_space.shape[-1]
            model_kwargs["action_noise"] = NormalActionNoise(
                mean=np.zeros(n_actions), sigma=np.full(n_actions, 0.1),
            )
        model_class = ALGORITHM_CLASSES[algorithm]
        model = model_class(
            policy, env, policy_kwargs=options, seed=seed, verbose=0, **model_kwargs,
        )
        provenance = {"backend": "stable_baselines3", "explicit_fallback": True}
    model.rlasset_backend = backend
    model.rlasset_provenance = provenance
    model.rlasset_training_budget = budget
    return model
