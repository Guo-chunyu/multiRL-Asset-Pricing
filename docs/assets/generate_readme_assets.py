"""Build self-contained README SVGs from frozen evidence; never run experiments.

Standard library only. Run from any directory; --check verifies rather than writes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "reports/runs/research_v9_signal"
INK, MUTED = "#14283f", "#53677c"
BLUE, TEAL, ORANGE = "#285a9b", "#087f82", "#ab4e28"
FONT = "Inter,DejaVu Sans,Arial,sans-serif"


def text(x, y, value, size=18, fill=INK, weight=400, anchor="start"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>')


def rect(x, y, width, height, fill, radius=12, stroke="none"):
    return (f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{radius}" fill="{fill}" stroke="{stroke}"/>')


def line(x1, y1, x2, y2, color="#c8d5df", width=2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{width}"/>')


def svg(name, height, title, description, content, background="#f7f9fc"):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" '
            f'viewBox="0 0 1200 {height}" role="img" aria-labelledby="{name}-title {name}-desc">\n'
            f'<title id="{name}-title">{escape(title)}</title>\n'
            f'<desc id="{name}-desc">{escape(description)}</desc>\n'
            f'<g font-family="{FONT}">\n{rect(0, 0, 1200, height, background, 20)}\n'
            + "\n".join(content) + "\n</g>\n</svg>\n")


def hero():
    c = [rect(36, 34, 5, 290, "#4bd2c4", 2),
         text(64, 66, "REPRODUCIBLE PORTFOLIO RESEARCH", 15, "#9db3cb", 600),
         text(60, 146, "multiRL", 74, "#ffffff", 700),
         text(64, 207, "Asset Pricing", 49, "#ffffff", 600),
         text(66, 249, "A-share ETFs  /  Deep reinforcement learning", 20, "#b6c7da"),
         line(66, 278, 637, 278, "#2a4058", 1)]
    for x, label in [(66, "CAUSAL DECISIONS"), (265, "AUDITABLE ACCOUNTS"), (492, "FROZEN PROTOCOLS")]:
        c.append(text(x, 307, label, 13, "#72dcd1", 600))
    c += [rect(742, 34, 420, 292, "#142b44", 18, "#2b455f"),
          text(772, 70, "LATEST STUDY", 14, "#9db3cb", 600),
          text(1128, 73, "v9", 27, "#72dcd1", 700, "end"),
          text(772, 114, "Relative-allocation signal", 24, "#ffffff", 600),
          text(772, 147, "3 ETFs  ·  4 years  ·  3 cost levels", 17, "#b6c7da")]
    for i, label in enumerate(["20-session rank", "50% equity", "24 ledgers"]):
        x = 772 + i * 121
        c += [rect(x, 173, 112, 44, "#213e59", 8), text(x+56, 200, label, 13, "#d2e6f5", 500, "middle")]
    c += [rect(772, 240, 360, 57, "#43382e", 9),
          text(791, 264, "GATE NOT PASSED", 16, "#ffd3a6", 700),
          text(791, 286, "No new RL training under this plan", 15, "#f1d7bc")]
    return svg("hero", 360, "multiRL Asset Pricing",
               "A-share ETF research with causal decisions, auditable accounts and frozen protocols. "
               "Latest study v9 failed its prespecified signal gate; no new RL was trained.", c, "#0b1d31")


def workflow():
    c = [text(44, 49, "FROM MARKET DATA TO REVIEWABLE EVIDENCE", 15, TEAL, 700),
         text(44, 87, "Separate the decision, the account and the claim.", 29, INK, 700)]
    cards = [
        ("01", "Verified inputs", ["Raw OHLCV", "Cash dividends + splits", "Snapshot identities"], BLUE),
        ("02", "Causal state", ["Information through close t", "Positions + receivables", "Past returns / risk"], TEAL),
        ("03", "Portfolio decision", ["RL, selectors or fixed rules", "Version-specific constraints", "No future observations"], BLUE),
        ("04", "Account + audit", ["Next raw open execution", "Fees + corporate actions", "Independent reconstruction"], TEAL),
    ]
    for i, (number, title, body, color) in enumerate(cards):
        x = 44 + i * 284
        c += [rect(x, 119, 260, 181, "#ffffff", 12, "#d8e2ec"),
              rect(x+16, 135, 36, 28, color, 7), text(x+34, 155, number, 14, "#ffffff", 700, "middle"),
              text(x+17, 193, title, 22, INK, 600)]
        c += [text(x+17, 224+j*25, value, 14, MUTED) for j, value in enumerate(body)]
        if i < 3:
            c += [line(x+265, 208, x+279, 208, "#8298ad"),
                  f'<path d="M{x+275} 203 l5 5 -5 5" fill="none" stroke="#8298ad" stroke-width="2"/>']
    c += [text(44, 340, "TWO DISTINCT RESEARCH PATHS", 14, MUTED, 700),
          rect(44, 359, 1112, 83, "#eaf0f9", 12),
          text(65, 391, "v3–v8", 19, BLUE, 700),
          text(65, 417, "Learned policies", 14, BLUE)]
    for x, title, subtitle in [(250, "Train", "Historical train windows"), (510, "Select + freeze", "Validation choices + hashes"), (826, "Evaluate + audit", "Fixed test paths / 5, 10, 20 bps")]:
        c += [text(x, 391, title, 20, INK, 600), text(x, 416, subtitle, 14, MUTED)]
    c += [line(445, 398, 485, 398, "#8298ad"), line(756, 398, 800, 398, "#8298ad"),
          rect(44, 457, 1112, 83, "#e8f5f3", 12),
          text(65, 489, "v9", 19, TEAL, 700), text(65, 515, "No RL training", 14, TEAL)]
    for x, title, subtitle in [(250, "Specify + review", "One rule / frozen gate"), (510, "Validate + audit", "24 annual accounts"), (826, "Gate not passed", "Do not train a new selector")]:
        c += [text(x, 489, title, 20, INK if x != 826 else ORANGE, 600), text(x, 514, subtitle, 14, MUTED)]
    c += [line(445, 496, 485, 496, "#8298ad"), line(756, 496, 800, 496, "#8298ad"),
          text(44, 578, "Previously inspected historical data. Engineering verification is not evidence of investment efficacy.", 15, MUTED)]
    return svg("workflow", 606, "Auditable portfolio research workflow",
               "Raw data and corporate actions feed causal state, version-specific portfolio decisions, next-open accounting and independent audits. "
               "v3 to v8 train, freeze validation selections and evaluate. v9 instead reviews one fixed rule, audits 24 validation paths and stops at a failed gate.", c)


def validated_gate():
    decision_path = EVIDENCE / "gate_decision.json"
    gate = json.loads(decision_path.read_text())
    with (EVIDENCE / "all_metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = [(int(r["cost_bps"]), int(r["year"]), r["rule"]) for r in rows]
    expected = {(c, y, r) for c in (5, 10, 20) for y in (2018, 2020, 2022, 2024)
                for r in ("equal50", "relative_momentum50")}
    if len(rows) != 24 or set(keys) != expected:
        raise ValueError("The complete frozen 24-row comparison is required")
    lookup = dict(zip(keys, rows))
    comparisons = {(r["cost_bps"], r["year"]): r for r in gate["comparisons"]}
    if len(comparisons) != 12:
        raise ValueError("Expected 12 paired comparisons")
    for cost in (5, 10, 20):
        wins, dd_ok = 0, True
        for year in (2018, 2020, 2022, 2024):
            base, new = [lookup[(cost, year, r)] for r in ("equal50", "relative_momentum50")]
            ret = float(new["total_return"]) - float(base["total_return"])
            dd = float(new["max_drawdown"]) - float(base["max_drawdown"])
            score = ret - 0.5 * dd
            for field, value in [("score_delta", score), ("total_return_delta", ret), ("max_drawdown_delta", dd)]:
                saved = comparisons[(cost, year)][field]
                if not (math.isfinite(saved) and math.isfinite(value)) or abs(saved - value) > 1e-12:
                    raise ValueError(f"CSV / gate mismatch: {cost}, {year}, {field}")
            wins += score > 0
            dd_ok &= dd <= 0.02 + 1e-12
        row = next(r for r in gate["cost_gates"] if r["cost_bps"] == cost)
        if (row["strict_score_wins"], row["drawdown_gate_passed"], row["passed"]) != (wins, dd_ok, wins >= 3 and dd_ok):
            raise ValueError("Derived cost gate differs from saved decision")
    if gate["passed"] or gate["decision"] != "do_not_train_new_selector":
        raise ValueError("These v9 illustrations require the frozen failed-gate evidence")
    return gate, hashlib.sha256(decision_path.read_bytes()).hexdigest()


def result_chart():
    gate, source_hash = validated_gate()
    c = [f'<!-- Source: reports/runs/research_v9_signal/gate_decision.json; SHA-256: {source_hash} -->',
         text(44, 48, "v9 / ALL REGISTERED VALIDATION COMPARISONS", 15, TEAL, 700),
         text(44, 86, "The relative-allocation rule did not clear the gate.", 29, INK, 700),
         text(44, 119, "Score difference vs equal50, in percentage points. Positive values favor the momentum rule.", 17, MUTED),
         text(44, 173, "ONE-WAY COST", 13, MUTED, 700),
         text(1054, 173, "SCORE WINS", 13, MUTED, 700, "middle")]
    years = (2018, 2020, 2022, 2024)
    for i, year in enumerate(years):
        c.append(text(280+i*192, 174, str(year), 19, INK, 600, "middle"))
    for row_index, cost in enumerate((5, 10, 20)):
        y = 194+row_index*83
        c.append(text(44, y+41, f"{cost} bps", 22, INK, 600))
        for i, year in enumerate(years):
            value = next(r["score_delta"]*100 for r in gate["comparisons"] if r["cost_bps"] == cost and r["year"] == year)
            pos = value > 0
            c += [rect(190+i*192, y, 180, 66, "#dfeef9" if pos else "#f8e7dd", 10),
                  text(280+i*192, y+43, f"{value:+.2f}", 29, BLUE if pos else ORANGE, 600, "middle")]
        wins = next(r["strict_score_wins"] for r in gate["cost_gates"] if r["cost_bps"] == cost)
        c += [text(1054, y+30, f"{wins} / 4", 29, INK, 700, "middle"),
              text(1054, y+55, "Need at least 3 / 4", 13, MUTED, 400, "middle")]
    c += [line(44, 465, 1156, 465, "#d8e2ec", 1),
          rect(44, 485, 538, 70, "#e8f5f3", 10), text(63, 513, "DRAWDOWN CONSTRAINT: PASSED", 15, TEAL, 700),
          text(63, 539, "No deterioration above 2 pp in any year / cost.", 16, MUTED),
          rect(600, 485, 556, 70, "#fff1e6", 10), text(619, 513, "OVERALL GATE: NOT PASSED", 15, ORANGE, 700),
          text(619, 539, "No new RL selector was trained under this plan.", 16, MUTED),
          text(44, 590, "Score = total net return − 0.5 × maximum drawdown. Each annual account starts independently.", 15, MUTED),
          text(44, 615, "Historical development only. A compute-budget hurdle, not a statistical test or an alpha estimate.", 15, MUTED)]
    description = "All 12 score differences in percentage points for relative_momentum50 minus equal50. "
    for cost in (5, 10, 20):
        values = [next(r["score_delta"]*100 for r in gate["comparisons"]
                       if r["cost_bps"] == cost and r["year"] == year) for year in years]
        description += f"{cost} bps: " + ", ".join(f"{value:+.2f}" for value in values) + ". "
    return svg("gate", 642, "v9 signal gate: two, two and zero winning years",
               description + "Years: 2018, 2020, 2022, 2024. "
               "The required three of four wins at every cost failed; all drawdown constraints passed.", c)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify SVGs and CSV consistency without writing")
    args = parser.parse_args()
    assets = {"readme-hero.svg": hero(), "research-workflow.svg": workflow(), "v9-signal-gate.svg": result_chart()}
    for name, value in assets.items():
        path = OUT / name
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != value:
                raise SystemExit(f"Stale or missing asset: {path}; rerun without --check")
        else:
            path.write_text(value, encoding="utf-8")
        print(f"{'Verified' if args.check else 'Generated'} {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
