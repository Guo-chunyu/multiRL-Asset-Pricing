import json
import numpy as np
import pytest

from rlasset.data import MarketPanel, synthetic_frame
from rlasset.meta import baseline_candidates
from rlasset.paper import PaperPolicyDriver, simulate_policy, validate_execution_snapshot
from rlasset.execution import ExecutionConfig


def test_driver_observes_actual_holdings_and_is_causal():
    panel = MarketPanel.from_frame(synthetic_frame(60, 2, 7))
    driver = PaperPolicyDriver(panel, 20, 40, 10, .5, 5, baseline_candidates()["equal_weight"])
    seen = []
    from rlasset.meta import Candidate
    driver.candidate = Candidate("probe", lambda e, o: seen.append(o) or np.array([.2, .4, .4]))
    weights = np.array([.3, .6, .1])
    driver.decide(20, weights)
    np.testing.assert_allclose(seen[0]["weights"], weights)
    expected = panel.close[11:21].T / panel.close[20, :, None]
    np.testing.assert_allclose(seen[0]["prices"][0], expected, rtol=1e-6)


def test_realistic_replay_reconciles_cash_positions_orders_and_is_deterministic():
    panel = MarketPanel.from_frame(synthetic_frame(60, 2, 7))
    def run():
        driver = PaperPolicyDriver(panel, 20, 40, 10, .5, 5, baseline_candidates()["equal_weight"])
        return simulate_policy(panel, driver, initial_cash=100000,
                               config=ExecutionConfig(slippage_bps=2, min_commission=5))
    a, b = run(), run()
    assert a == b
    assert len(a["ledger"]) == 21
    assert a["orders"]
    for row in a["ledger"]:
        assert row["cash"] >= 0
        nav = row["cash"] + sum(row["positions"].get(t, 0) * row["mark_prices"][t] for t in panel.tickers)
        assert np.isclose(nav, row["nav"])
        assert sum(row["weights"].values()) == pytest.approx(1)
    for order in a["orders"]:
        assert order["requested_quantity"] % 100 == 0


def test_raw_snapshot_refuses_adjusted_or_tampered_data(tmp_path):
    path = tmp_path / "raw.csv"
    synthetic_frame(60, 2).to_csv(path, index=False)
    path.with_suffix(".metadata.json").write_text(json.dumps({"adjust": "hfq"}))
    with pytest.raises(ValueError, match="unadjusted"):
        validate_execution_snapshot(path)
    path.with_suffix(".metadata.json").write_text(json.dumps({"adjust": "none", "sha256": "wrong"}))
    with pytest.raises(ValueError, match="hash"):
        validate_execution_snapshot(path)


def test_consensus_selection_needs_only_validation_and_never_picks_a_seed(tmp_path):
    from rlasset.paper import choose_consensus_rl
    (tmp_path / "manifest.json").write_text(json.dumps({"config": {"seeds": [7, 17]}}))
    for seed, returns in [(7, {"ppo": .1, "sac": .4}), (17, {"ppo": .2, "sac": -.2})]:
        directory = tmp_path / f"seed_{seed}"
        directory.mkdir()
        (directory / "models.json").write_text(json.dumps({"ppo": {}, "sac": {}}))
        (directory / "validation_metrics.json").write_text(json.dumps({
            name: {"total_return": value, "max_drawdown": .1} for name, value in returns.items()
        }))
    # There are deliberately no test files or model performance ranks.
    assert choose_consensus_rl(tmp_path) == "ppo"
