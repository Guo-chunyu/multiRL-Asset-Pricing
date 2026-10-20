import json
import numpy as np
import pytest


def test_complete_experiment_trains_selects_and_reloads_without_test_training(tmp_path):
    from rlasset.data import synthetic_frame
    from rlasset.experiment import ExperimentConfig, run_experiment, evaluate_saved
    data_path = tmp_path / "synthetic.csv"
    synthetic_frame(140, 2, 3).to_csv(data_path, index=False)
    config = ExperimentConfig(total_timesteps=24, seeds=(3,), lookback=10)
    result = run_experiment(data_path, tmp_path / "run", config, data_kind="synthetic")
    assert result["data_kind"] == "synthetic"
    seed_dir = tmp_path / "run" / "seed_3"
    test = json.loads((seed_dir / "test_metrics.json").read_text())
    assert {"ddpg", "td3_risk_0", "equal_weight", "buy_hold", "dqn_selector"}.issubset(test)
    assert np.isfinite(test["dqn_selector"]["total_return"])
    selection = json.loads((seed_dir / "selection.json").read_text())
    assert selection["selected_on"] == "validation"
    before = json.loads((tmp_path / "run" / "manifest.json").read_text())
    reloaded = evaluate_saved(tmp_path / "run", seed=3)
    for name, metrics in reloaded.items():
        assert np.isclose(metrics["total_return"], test[name]["total_return"])
    assert before == json.loads((tmp_path / "run" / "manifest.json").read_text())
    stressed = evaluate_saved(tmp_path / "run", seed=3, cost_bps=50)
    assert stressed["equal_weight"]["total_return"] < reloaded["equal_weight"]["total_return"]
    assert stressed["cash"]["total_return"] == 0
    assert test == json.loads((seed_dir / "test_metrics.json").read_text())
    checkpoint_times = {str(p): p.stat().st_mtime_ns for p in (seed_dir / "models").glob("*.zip")}
    resumed = run_experiment(data_path, tmp_path / "run", config, data_kind="synthetic", resume=True)
    assert resumed == result
    assert checkpoint_times == {str(p): p.stat().st_mtime_ns for p in (seed_dir / "models").glob("*.zip")}
    with pytest.raises(ValueError, match="match"):
        run_experiment(data_path, tmp_path / "run", ExperimentConfig(total_timesteps=25, seeds=(3,), lookback=10), resume=True)
