# v8 results: count-only normalization

**The encoding fix improved neural responsiveness, but did not establish a reliably better trading strategy. No default strategy is promoted.**

The only intervention was dividing the bounded history-count observation by 20. Across 24 matched checkpoints on common mature validation states, mean first-layer tanh saturation fell from 70.5845% to 5.4123%; mean maximum per-action probability SD rose from 0.002545 to 0.035864. This is a mechanism result, not evidence of alpha.

## Complete 5-bps candidate results

These are means of three individual seeded runs, not a separately traded ensemble. Return is cumulative over each fold; CAGR uses 252 market sessions/year. Drawdown is a positive loss magnitude.

| Fold | Candidate | Total net return | Annualized net return | Maximum drawdown | Executed switches |
|---|---|---:|---:|---:|---:|
| fold_01 | drawdown_raw | 34.32% | 16.53% | 12.01% | 0.67 |
| fold_01 | drawdown_scaled | 37.08% | 17.77% | 11.85% | 2.33 |
| fold_01 | lognet_raw | 34.98% | 16.83% | 12.01% | 0.33 |
| fold_01 | lognet_scaled | 35.64% | 17.12% | 12.01% | 0.00 |
| fold_02 | drawdown_raw | -13.63% | -7.37% | 19.48% | 0.67 |
| fold_02 | drawdown_scaled | -13.61% | -7.35% | 19.43% | 0.67 |
| fold_02 | lognet_raw | -13.67% | -7.38% | 19.48% | 0.33 |
| fold_02 | lognet_scaled | -13.18% | -7.11% | 19.48% | 0.00 |
| fold_03 | drawdown_raw | -5.09% | -2.69% | 20.91% | 0.67 |
| fold_03 | drawdown_scaled | -4.23% | -2.23% | 20.91% | 0.00 |
| fold_03 | lognet_raw | -5.09% | -2.69% | 20.91% | 0.67 |
| fold_03 | lognet_scaled | -4.23% | -2.23% | 20.91% | 0.00 |
| fold_04 | drawdown_raw | 4.46% | 2.52% | 3.19% | 0.67 |
| fold_04 | drawdown_scaled | 3.40% | 1.94% | 3.53% | 3.33 |
| fold_04 | lognet_raw | 6.87% | 3.90% | 6.24% | 0.33 |
| fold_04 | lognet_scaled | 3.36% | 1.93% | 7.98% | 11.33 |

Historical test folds: 2019–2020; 2021–2022; 2023–2024; 2025–2026-10-22. Each starts independently at NAV 1; do not concatenate them into a live equity curve.

## Interpretation

- Small improvements in earlier folds do not establish durable superiority. In the latest fold, lognet scaling reduced mean cumulative return from 6.87% to 3.36% and worsened drawdown from 6.24% to 7.98%; its extra switching raises cost sensitivity. At 20 bps the paired cumulative-return difference worsens from −3.51 to −4.22 percentage points.
- Validation-only learned choices were drawdown_scaled, drawdown_scaled, lognet_raw, lognet_scaled. Including fixed comparators chose cash, equal50, cash, lognet_scaled. No test winner replaced these choices.
- Paired 10-day block intervals (2000 draws) are exploratory annualized arithmetic excess intervals, not confidence intervals for CAGR or cumulative differences. No multiplicity correction; seeds are optimizer randomness, not independent market histories.
- All historical periods had been inspected in prior studies. These results are retrospective development evidence, not untouched out-of-sample performance.

## Verification and compute

- 24 new fits × exactly 20,000 daily steps = 480,000 retained new transitions. 24 original v7 phase models were reused byte-for-byte and replay-verified, not retrained.
- Two execution interruptions were recovered with unchanged sources/configuration/seeds. Six, then nine, completed records were verified and retained. Causes and extra discarded partial compute are unknown and disclosed in execution_recovery.json; retained transition counts exclude that unknown overhead.
- Full suite passed 483 tests. Independent final review recomputed five financial metrics for all 216 test rows and all 24 paired contrasts, checked 48 model/validation records and 288 accounting-audit sidecars.
- Raw three-ETF price/dividend/split snapshots remain identical to prior restoration. Previously disclosed fresh cross-source verification limitations remain; no new data certification is claimed.
- No live trading, paid resources, new asset universe, remote push or publication. All prior v5–v7 frozen files remain unchanged.

## Artifacts

The complete run is runs/research_v8; compact evidence is mirrored in reports/runs/research_v8. REPORT.md, all_metrics.csv and selected_metrics.csv retain every fold, seed and 5/10/20-bps cost. validation_diagnostics.json separates common-state and own-policy diagnostics. The frozen protocol is docs/research-protocol-v8.md. Full models and daily ledgers are retained separately from the compact source/results package.
