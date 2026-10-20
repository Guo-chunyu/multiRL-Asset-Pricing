"""Future-price perturbations must not change learning or validation selection."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
from stable_baselines3.common.save_util import load_from_zip_file

from rlasset.data import synthetic_frame
from rlasset.experiment import ExperimentConfig, run_experiment


def _assert_exact_tree(left, right, path):
    """Compare all saved state_dicts, optimizers and extra PyTorch variables."""
    assert type(left) is type(right), path
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert torch.equal(left, right), f"Future prices changed checkpoint tensor: {path}"
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_exact_tree(left[key], right[key], f"{path}/{key}")
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_exact_tree(a, b, f"{path}/{index}")
    else:
        assert left == right, path


@pytest.mark.parametrize("steps", [24, 48])
def test_future_test_prices_cannot_change_five_rl_models_selector_or_validation(tmp_path, steps):
    # Catches future-dependent normalization/rewards, wrong training boundaries,
    # and test-dependent strategy selection. No model or training path is mocked.
    original = synthetic_frame(periods=140, assets=2, seed=3)
    changed = original.copy(deep=True)
    dates = pd.DatetimeIndex(sorted(original.date.unique()))
    # Hand-checked default partitions for 140 dates and lookback 10:
    # train 10..74, meta 75..100, validation 101..119, test 120..139.
    test_start = dates[120]
    future = changed.date >= test_start
    day = changed.loc[future, "date"].map({date: i for i, date in enumerate(dates[120:])}).to_numpy()
    direction = np.where(changed.loc[future, "tic"].eq("SYN000"), 1., -1.)
    multiplier = np.exp(direction * (.15 + .035 * day))
    changed.loc[future, ["open", "high", "low", "close"]] *= multiplier[:, None]
    pd.testing.assert_frame_equal(original.loc[~future], changed.loc[~future], check_exact=True)
    assert not original.loc[future, "close"].equals(changed.loc[future, "close"])

    config = ExperimentConfig(
        total_timesteps=steps, seeds=(3,), lookback=10, risk_levels=(2.,),
        algorithms=("td3", "ddpg", "sac", "ppo", "a2c"),
        selector_all_algorithms=True,
    )
    runs = []
    for label, frame in (("original", original), ("future_changed", changed)):
        data_path = tmp_path / f"{label}.csv"
        frame.to_csv(data_path, index=False)
        directory = tmp_path / label
        run_experiment(data_path, directory, config, data_kind="synthetic")
        runs.append(directory / "seed_3")

    left, right = runs
    checkpoints = {path.name for path in (left / "models").glob("*.zip")}
    assert checkpoints == {"td3_risk_2.zip", "ddpg.zip", "sac.zip", "ppo.zip", "a2c.zip", "dqn_selector.zip"}
    assert checkpoints == {path.name for path in (right / "models").glob("*.zip")}
    for name in sorted(checkpoints):
        _, left_parameters, left_variables = load_from_zip_file(left / "models" / name, load_data=False, device="cpu")
        _, right_parameters, right_variables = load_from_zip_file(right / "models" / name, load_data=False, device="cpu")
        assert left_parameters and "policy" in left_parameters
        _assert_exact_tree(left_parameters, right_parameters, name)
        _assert_exact_tree(left_variables, right_variables, f"{name}/extra_variables")

    for name in ("validation_metrics.json", "selection.json"):
        assert json.loads((left / name).read_text()) == json.loads((right / name).read_text()), name
    for ledger in left.glob("validation_*.csv"):
        pd.testing.assert_frame_equal(
            pd.read_csv(ledger), pd.read_csv(right / ledger.name), check_exact=True,
        )
    # Positive control: the shock really changes an independently valued
    # portfolio's held-out returns; this is not an ineffective constant rescale.
    left_test = json.loads((left / "test_metrics.json").read_text())
    right_test = json.loads((right / "test_metrics.json").read_text())
    assert left_test["equal_weight"]["total_return"] != right_test["equal_weight"]["total_return"]
