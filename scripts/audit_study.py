"""Independently reconstruct saved paper ledgers and fingerprint completed study."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_paper(folder):
    summary = json.loads((folder / "summary.json").read_text())
    ledger = json.loads((folder / "ledger.json").read_text())
    import pandas as pd
    orders_path = folder / "orders.csv"
    try:
        orders = pd.read_csv(orders_path, dtype={"ticker": str}).to_dict("records")
    except pd.errors.EmptyDataError:
        orders = []
    assert len({o["client_order_id"] for o in orders}) == len(orders)
    by_date = defaultdict(list)
    for order in orders:
        by_date[order["date"]].append(order)
    cash, positions = summary["initial_cash"], defaultdict(int)
    assert len(ledger) == 171
    for row in ledger:
        sold_today = defaultdict(int)
        settled = positions.copy()
        for order in by_date[row["date"]]:
            tic, qty = order["ticker"], int(order["filled_quantity"])
            assert qty >= 0
            assert qty + order["cancelled_quantity"] == order["requested_quantity"]
            if qty:
                assert np.isclose(order["notional"], qty * order["filled_price"], atol=1e-7, rtol=1e-10)
            sign = 1 if order["side"] == "buy" else -1
            if sign == -1:
                sold_today[tic] += qty
                assert sold_today[tic] <= settled.get(tic, 0)  # no sale of new shares
            else:
                assert qty % 100 == 0
            cash -= sign * order["notional"] + order["commission"]
            positions[tic] += sign * qty
            assert positions[tic] >= 0 and cash >= -1e-7
            assert np.isclose(cash, order["cash_after"], rtol=0, atol=1e-6)
            assert positions[tic] == order["position_after"]
        assert np.isclose(cash, row["cash"], rtol=0, atol=1e-6)
        for tic in row["mark_prices"]:
            assert positions[tic] == row["positions"].get(tic, 0)
        nav = cash + sum(q * row["mark_prices"][tic] for tic, q in positions.items())
        assert np.isclose(nav, row["nav"], rtol=0, atol=1e-6)
    assert np.isclose(ledger[-1]["nav"], summary["final_nav"])
    return {"folder": str(folder), "transitions": len(ledger) - 1,
            "orders_reconciled": len(orders), "cash_shares_nav_T1": "passed"}


def main():
    root = Path(__file__).resolve().parents[1]
    runs = [root / "runs/comparison_v2", *(root / "runs/walkforward_v2").glob("fold_*"),
            root / "runs/risk0_v2", root / "runs/risk8_v2"]
    checkpoints = {}
    for run in runs:
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "complete", run
        assert digest(Path(manifest["data_path"])) == manifest["data_sha256"]
        for seed in (7, 17, 29):
            models = run / f"seed_{seed}" / "models"
            for model in models.glob("*.zip"):
                checkpoints[str(model.relative_to(root))] = digest(model)
    paper_folders = sorted((root / "runs/paper_v2").glob("seed_*/*/*"))
    assert len(paper_folders) == 30
    audits = [audit_paper(folder) for folder in paper_folders]
    for folder in paper_folders:
        summary = json.loads((folder / "summary.json").read_text())
        for name, expected in summary["model_hashes"].items():
            model = root / "runs/comparison_v2" / f"seed_{summary['seed']}" / "models" / name
            assert digest(model) == expected
    suites = ET.parse(root / "runs/pytest_v2.xml").getroot()
    tests = sum(int(s.get("tests", 0)) for s in suites.iter("testsuite"))
    failures = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites.iter("testsuite"))
    assert failures == 0
    result = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "pytest": {"tests": tests, "failures_or_errors": failures},
        "complete_runs": [str(p.relative_to(root)) for p in runs],
        "checkpoint_count": len(checkpoints), "checkpoints_sha256": checkpoints,
        "paper_audits": audits,
        "source_sha256": {str(p.relative_to(root)): digest(p) for p in (root / "src/rlasset").glob("*.py")},
        "scope": "Computational and accounting verification; retrospective evidence, not a profitability or live-order certification.",
    }
    target = root / "runs/verification_v2.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"tests": tests, "checkpoints": len(checkpoints), "paper_ledgers": len(audits),
                      "orders_reconciled": sum(a["orders_reconciled"] for a in audits)}))


if __name__ == "__main__":
    main()
