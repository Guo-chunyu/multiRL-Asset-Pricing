"""Real multi-algorithm learning, budget accounting and checkpoint compatibility."""
import json

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.mark.parametrize("algorithm", ["sac", "ppo", "a2c"])
def test_finrl_new_algorithms_learn_roundtrip_and_account_for_rollout_budget(tmp_path, algorithm):
    from rlasset.data import MarketPanel, synthetic_frame
    from rlasset.env import PortfolioEnv
    from rlasset.extractors import SharedTemporalExtractor
    from rlasset.finrl_adapter import make_agent

    torch.set_num_threads(1)
    panel = MarketPanel.from_frame(synthetic_frame(80, 2, 3))
    env = PortfolioEnv(panel, start=10, end=75, lookback=10)
    model = make_agent(env, algorithm=algorithm, total_timesteps=137, seed=3,
                       policy_kwargs={"features_extractor_class": SharedTemporalExtractor})
    before = [p.detach().clone() for p in model.policy.parameters()]
    model.learn(total_timesteps=137)
    assert model.rlasset_backend == "finrl"
    assert any(not torch.equal(a, b) for a, b in zip(before, model.policy.parameters()))
    assert model.rlasset_training_budget["requested_steps"] == 137
    assert model.rlasset_training_budget["expected_actual_steps"] == model.num_timesteps
    assert model.num_timesteps == (192 if algorithm == "ppo" else 137)
    observation, _ = env.reset()
    action, _ = model.predict(observation, deterministic=True)
    assert np.isfinite(action).all()
    model.save(tmp_path / algorithm)
    loaded = type(model).load(tmp_path / algorithm)
    np.testing.assert_allclose(loaded.predict(observation, deterministic=True)[0], action)


def test_all_algorithm_experiment_selects_rl_and_resumes_without_retraining(tmp_path):
    from rlasset.data import synthetic_frame
    from rlasset.experiment import ExperimentConfig, run_experiment, evaluate_saved

    data = tmp_path / "synthetic.csv"
    synthetic_frame(140, 2, 3).to_csv(data, index=False)
    config = ExperimentConfig(total_timesteps=24, seeds=(3,), lookback=10,
                              algorithms=("td3", "ddpg", "sac", "ppo", "a2c"),
                              selector_all_algorithms=True, risk_levels=(2.,))
    run = tmp_path / "run"
    result = run_experiment(data, run, config, data_kind="synthetic")
    seed_dir = run / "seed_3"
    details = json.loads((seed_dir / "models.json").read_text())
    assert list(details) == ["td3_risk_2", "ddpg", "sac", "ppo", "a2c"]
    for detail in details.values():
        assert detail["requested_steps"] == detail["actual_steps"] == 24
        assert detail["training_wall_seconds"] >= 0
    selection = json.loads((seed_dir / "selection.json").read_text())
    assert selection["pool_order"] == ["td3_risk_2", "ddpg", "sac", "ppo", "a2c", "momentum", "reversal", "cash"]
    validation = json.loads((seed_dir / "validation_metrics.json").read_text())
    expected = max(details, key=lambda name: validation[name]["total_return"] - .5 * validation[name]["max_drawdown"])
    assert selection["best_rl_on_validation"] == expected
    assert result["seeds"]["3"]["best_rl_on_validation"] == expected
    pd.testing.assert_frame_equal(pd.read_csv(seed_dir / "test_best_rl_on_validation.csv"),
                                  pd.read_csv(seed_dir / f"test_{expected}.csv"))
    reloaded = evaluate_saved(run, seed=3)
    for name, metrics in reloaded.items():
        assert np.isclose(metrics["total_return"], result["seeds"]["3"]["test_metrics"][name]["total_return"])
    times = {str(p): p.stat().st_mtime_ns for p in (seed_dir / "models").glob("*.zip")}
    resumed = run_experiment(data, run, config, data_kind="synthetic", resume=True)
    assert resumed == result
    assert times == {str(p): p.stat().st_mtime_ns for p in (seed_dir / "models").glob("*.zip")}
    selection["pool_order"].reverse()
    (seed_dir / "selection.json").write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="disagree"):
        evaluate_saved(run, seed=3)


def test_legacy_pool_order_is_not_expanded_implicitly():
    from rlasset.experiment import _pool

    models = {name: object() for name in ["td3_risk_0", "td3_risk_2", "td3_risk_8", "ddpg", "sac"]}
    legacy, _ = _pool(models)
    assert [x.name for x in legacy] == ["td3_risk_0", "td3_risk_2", "td3_risk_8", "momentum", "reversal", "cash"]
    order = ["sac", "ddpg", "td3_risk_8", "td3_risk_2", "td3_risk_0", "cash", "reversal", "momentum"]
    restored, _ = _pool(models, selector_all_algorithms=True, pool_order=order)
    assert [x.name for x in restored] == order
    with pytest.raises(ValueError, match="pool"):
        _pool(models, pool_order=order)


def test_multi_algorithm_cli_accepts_explicit_algorithms_and_risk_levels(tmp_path):
    from rlasset.cli import main

    # Invalid step budget stops before training, after the actual CLI config path.
    with pytest.raises(ValueError, match="budget"):
        main(["demo", "--output", str(tmp_path / "cli"), "--steps", "0",
              "--algorithms", "td3", "ddpg", "sac", "ppo", "a2c", "--risk-levels", "2"])


def test_ppo_prime_rollout_budget_has_a_valid_minibatch():
    from rlasset.data import MarketPanel, synthetic_frame
    from rlasset.env import PortfolioEnv
    from rlasset.finrl_adapter import make_agent

    env = PortfolioEnv(MarketPanel.from_frame(synthetic_frame(80, 2, 4)), start=10, end=75, lookback=10)
    model = make_agent(env, algorithm="ppo", total_timesteps=67)
    model.learn(total_timesteps=67)
    assert model.num_timesteps == 67


def test_legacy_manifest_without_new_fields_reloads_and_resumes(tmp_path):
    from stable_baselines3 import DQN
    from rlasset.data import synthetic_frame
    from rlasset.experiment import ExperimentConfig, run_experiment, evaluate_saved
    from rlasset.finrl_adapter import ALGORITHM_CLASSES

    data = tmp_path / "legacy.csv"
    synthetic_frame(140, 2, 5).to_csv(data, index=False)
    run = tmp_path / "legacy"
    config = ExperimentConfig(total_timesteps=24, seeds=(5,), lookback=10, risk_levels=(2.,))
    expected = run_experiment(data, run, config, data_kind="synthetic")
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["config"].pop("algorithms")
    manifest["config"].pop("selector_all_algorithms")
    (run / "manifest.json").write_text(json.dumps(manifest))
    seed_dir = run / "seed_5"
    details = json.loads((seed_dir / "models.json").read_text())
    for name, entry in details.items():
        path = seed_dir / "models" / name
        model = ALGORITHM_CLASSES[entry["algorithm"]].load(path)
        del model.rlasset_training_budget
        del model.rlasset_training_wall_seconds
        model.save(path)
    selector_path = seed_dir / "models/dqn_selector"
    selector = DQN.load(selector_path)
    del selector.rlasset_pool_order
    del selector.rlasset_training_wall_seconds
    selector.save(selector_path)
    actual = evaluate_saved(run, seed=5)
    assert actual == expected["seeds"]["5"]["test_metrics"]
    checkpoints = {str(p): p.stat().st_mtime_ns for p in (seed_dir / "models").glob("*.zip")}
    resumed = run_experiment(data, run, config, data_kind="synthetic", resume=True)
    assert resumed == expected
    assert checkpoints == {str(p): p.stat().st_mtime_ns for p in (seed_dir / "models").glob("*.zip")}
