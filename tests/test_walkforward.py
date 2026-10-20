import json

import numpy as np
import pytest


def test_expanding_splits_have_expected_chronological_boundaries():
    from rlasset.walkforward import expanding_splits

    folds = expanding_splits(1000, lookback=20, folds=3)
    assert folds == [
        {"train": (20, 449), "meta": (450, 599), "validation": (600, 699), "test": (700, 799)},
        {"train": (20, 549), "meta": (550, 699), "validation": (700, 799), "test": (800, 899)},
        {"train": (20, 649), "meta": (650, 799), "validation": (800, 899), "test": (900, 999)},
    ]


@pytest.mark.parametrize("n,lookback,fold_count", [(503, 20, 3), (999, 10, 5), (140, 10, 3)])
def test_each_fold_is_causal_and_held_out_returns_never_overlap(n, lookback, fold_count):
    from rlasset.walkforward import expanding_splits

    plans = expanding_splits(n, lookback, fold_count)
    test_states, test_returns, train_ends, lengths = [], [], [], []
    for plan in plans:
        previous_end = lookback - 1
        for stage in ("train", "meta", "validation", "test"):
            start, end = plan[stage]
            assert start == previous_end + 1
            assert end - start + 1 >= 2
            assert 0 <= start < end < n
            previous_end = end
        start, end = plan["test"]
        test_states.extend(range(start, end + 1))
        test_returns.extend(range(start + 1, end + 1))
        train_ends.append(plan["train"][1])
        lengths.append(end - start + 1)
        assert plan["train"][0] == lookback
        assert plan["meta"][1] - plan["meta"][0] + 1 == int(n * .15)
        assert plan["validation"][1] - plan["validation"][0] + 1 == int(n * .10)
    assert len(test_returns) == len(set(test_returns))
    assert test_states == list(range(int(n * .70), n))
    assert np.all(np.diff(train_ends) > 0)
    assert max(lengths) - min(lengths) <= 1


@pytest.mark.parametrize("kwargs", [
    {"n": 30}, {"n": 100, "lookback": 50}, {"n": 100, "folds": 20},
    {"n": 100, "folds": 0}, {"n": 100, "lookback": 1},
    {"n": 100.5}, {"n": 100, "folds": 2.5}, {"n": 100, "lookback": 2.5},
])
def test_insufficient_or_nonintegral_walkforward_inputs_are_rejected(kwargs):
    from rlasset.walkforward import expanding_splits

    with pytest.raises(ValueError):
        expanding_splits(**kwargs)


def test_walkforward_summary_preserves_seed_identity_across_folds(tmp_path, monkeypatch):
    # Training is expensive and independently integrated; this checks orchestration
    # and JSON summaries with controlled fold outcomes, never the mock's call count.
    from rlasset.data import synthetic_frame
    from rlasset.experiment import ExperimentConfig
    from rlasset import walkforward

    data = tmp_path / "synthetic.csv"
    synthetic_frame(140, 2, 3).to_csv(data, index=False)
    config = ExperimentConfig(total_timesteps=24, seeds=(7, 17), lookback=10)
    def train_fold(data_path, output, config, *, split_indices):
        # Test blocks begin at 98,112,126; hand-defined fold returns.
        fold_index = {98: 0, 112: 1, 126: 2}[split_indices["test"][0]]
        return {
            "data_kind": "synthetic",
            "seeds": {
                "7": {"selected_strategy": "cash", "selected_test_metrics": {"total_return": [.01, .02, .03][fold_index]}},
                "17": {"selected_strategy": "momentum", "selected_test_metrics": {"total_return": [.11, .12, .13][fold_index]}},
            },
        }
    monkeypatch.setattr(walkforward, "run_experiment", train_fold)
    summary = walkforward.run_walk_forward(data, tmp_path / "walk", config, folds=3)
    stored = json.loads((tmp_path / "walk" / "walkforward_summary.json").read_text(encoding="utf-8"))
    assert summary == stored
    assert stored["test_returns_overlap"] is False
    assert len(stored["folds"]) == 3
    assert stored["seed_aggregates"]["7"]["fold_returns"] == [.01, .02, .03]
    assert stored["seed_aggregates"]["7"]["mean_fold_return"] == pytest.approx(.02)
    assert stored["seed_aggregates"]["7"]["std_fold_return"] == pytest.approx(.01)
    assert stored["seed_aggregates"]["17"]["mean_fold_return"] == pytest.approx(.12)
    assert stored["folds"][0]["splits"]["test"] == [98, 111]


def test_walkforward_refuses_a_nonempty_output_before_training(tmp_path):
    from rlasset.data import synthetic_frame
    from rlasset.experiment import ExperimentConfig
    from rlasset.walkforward import run_walk_forward

    data = tmp_path / "synthetic.csv"
    synthetic_frame(140, 2, 3).to_csv(data, index=False)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "prior_result.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_walk_forward(data, output, ExperimentConfig(lookback=10), folds=3)
    assert sentinel.read_text(encoding="utf-8") == "preserve"
