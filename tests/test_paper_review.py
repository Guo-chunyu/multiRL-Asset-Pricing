"""Independent audit checks for execution inputs and selector chronology."""
import hashlib
import json

import numpy as np
import pytest

from rlasset.data import MarketPanel, synthetic_frame
from rlasset.execution import ExecutionConfig
from rlasset.meta import Candidate, baseline_candidates
from rlasset.paper import PaperPolicyDriver, simulate_policy, validate_execution_snapshot


def _driver(target, panel=None):
    panel = panel or MarketPanel.from_frame(synthetic_frame(60, 2, 7))
    return PaperPolicyDriver(panel, 20, 21, 10, .5, 5,
                             candidate=Candidate("test", lambda env, obs: np.asarray(target)))


@pytest.mark.parametrize("target", [[.1, .8, .1], [.2, .4, .400005]])
def test_paper_rejects_cap_violation_and_non_simplex_target(target):
    driver = _driver(target)
    with pytest.raises(ValueError, match="target weights|max_weight"):
        simulate_policy(driver.panel, driver)


def test_accepted_simplex_roundoff_is_normalized_before_broker_sizing():
    driver = _driver([0., .50000004, .5])
    result = simulate_policy(driver.panel, driver)
    target = result["ledger"][1]["target_weights"]
    assert sum(target.values()) == pytest.approx(1., abs=1e-14)


@pytest.mark.parametrize("unit", ["hands", None])
def test_raw_metadata_requires_explicit_share_volume_units(tmp_path, unit):
    path = tmp_path / "raw.csv"
    path.write_text("raw data", encoding="utf-8")
    metadata = {"adjust": "none", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "volume_unit": unit}
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="volume"):
        validate_execution_snapshot(path)


class _RecordingSelector:
    def __init__(self):
        self.observations = []

    def predict(self, observation, deterministic):
        assert deterministic
        self.observations.append({key: value.copy() for key, value in observation.items()})
        return 0, None


def test_selector_has_only_realized_shadow_returns_and_actual_paper_weights():
    original = synthetic_frame(60, 2, 7)
    changed = original.copy()
    future = changed["date"] > sorted(changed["date"].unique())[22]
    changed.loc[future, ["open", "close", "high", "low"]] *= 3
    recorded = []
    actual = np.array([.3, .6, .1])  # Actual drift may exceed the target cap.
    for frame in (original, changed):
        panel = MarketPanel.from_frame(frame)
        selector = _RecordingSelector()
        driver = PaperPolicyDriver(panel, 20, 40, 10, .5, 5,
                                   selector=selector, pool=[baseline_candidates()["equal_weight"]])
        for t in range(20, 23):
            assert all(shadow.t == t for shadow in driver.context.shadows)
            assert all(len(history) == t - 20 for history in driver.context.performance_history)
            driver.decide(t, actual)
            np.testing.assert_allclose(selector.observations[-1]["weights"], actual)
            if t == 20:
                np.testing.assert_array_equal(selector.observations[-1]["performance"], 0)
            if t < 22:
                driver.advance_shadows()
        recorded.append(selector.observations)
    for first, second in zip(*recorded):
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])


def test_signal_scaling_never_changes_share_execution_prices():
    frame = synthetic_frame(60, 2, 7)
    raw = MarketPanel.from_frame(frame)
    scaled = frame.copy()
    scaled[["open", "close", "high", "low"]] *= 5
    signal = MarketPanel.from_frame(scaled)
    first = simulate_policy(raw, _driver([.2, .4, .4], raw),
                            config=ExecutionConfig(slippage_bps=0))
    second = simulate_policy(raw, _driver([.2, .4, .4], signal),
                             config=ExecutionConfig(slippage_bps=0))
    assert first == second
    first_fill = first["orders"][0]
    asset = raw.tickers.index(first_fill["ticker"])
    assert first_fill["filled_price"] == raw.open[21, asset]
