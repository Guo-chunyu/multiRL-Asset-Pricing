# v6 results: fixed-expert PPO selector

## Bottom line

**No consistent incremental advantage over transparent deterministic comparators was established.** Validation deployment selection rejected PPO in all four folds (cash / equal50 / cash / vol10). Test behavior was overwhelmingly fixed-expert selection rather than dynamic rotation. Higher first-fold PPO return came with greater drawdown; the second fold was worse on both return and drawdown than the equal-target mixture.

These are retrospective exploratory results on already-inspected historical periods, not fresh out-of-sample confirmation, convergence evidence, or a guarantee of future returns. No family, lookback, seed, training budget, or validation choice changed after the preview.

## Completed execution

- 24 final PPO checkpoints; seeds 7/17/29; two objectives; four existing folds; exactly 20,000 steps each (480,000 total). Real pinned FinRL DRLAgent, 64×64 MLP, learning rate 1e-4.
- All checkpoints, validation records and selections frozen at 2026-10-26T16:40:29.865985+00:00 before v6 test evaluation.
- 5/10/20bps fixed-checkpoint replay completed; all six deterministic comparators use actual single-account costs and corporate actions. No rule is treated as three independent seed observations.
- 26 focused v6 tests and 107 inherited regressions passed independent review; final summary-only change additionally passed nine study tests. Synthetic Discrete PPO smoke passed.
- 216 actual-account ledger audits passed (24 preview, 24 PPO validation, 24 rule validation, 144 test). Maximum raw-account reconstruction discrepancy: 3.10×10⁻¹⁴.
- Data exactly matches certified historical hashes; fresh Eastmoney verification failed HTTP503. Archived historical cross-provider proof and restoration evidence are preserved; this is not fresh independent data verification.

## Primary 5bps comparison

Each PPO cell is the arithmetic mean of three separate trained paths, **not a tradable ensemble**. Figures are full-fold total return / maximum drawdown, not annualized. Rules are single deterministic paths.

| Test fold | Validation PPO choice | PPO return / DD | Equal-target mix return / DD | Trailing20 return / DD | Validation-fixed expert return / DD |
|---|---|---:|---:|---:|---:|
| fold_01 | lognet_ppo | 28.61% / 12.36% | 20.04% / 7.75% | 16.60% / 14.27% | cash: 0.00% / -0.00% |
| fold_02 | lognet_ppo | -16.39% / 23.08% | -8.23% / 12.93% | -5.07% / 10.99% | equal50: -9.00% / 16.22% |
| fold_03 | drawdown_ppo | -1.41% / 6.97% | -1.86% / 11.31% | -0.61% / 6.21% | cash: 0.00% / -0.00% |
| fold_04 | lognet_ppo | 4.23% / 3.19% | 7.90% / 5.91% | 7.64% / 6.64% | vol10: 12.66% / 9.58% |

## Paired uncertainty

95% overlapping moving-block percentile intervals, 10-session blocks, 2000 draws. Quantities below are annualized arithmetic paired excess in percentage points, **not CAGR differences**. Seed-mean daily-return series is used for the contrast; its compounded return need not equal the average of seed terminal returns. No multiple-comparison correction.

| Fold | Comparator | Excess pp/year | 95% interval |
|---|---|---:|---:|
| fold_01 | equal_target_mix | 3.75 | [-2.84, 9.61] |
| fold_01 | trailing20 | 4.94 | [-1.50, 9.70] |
| fold_01 | cash | 13.55 | [-6.49, 31.44] |
| fold_02 | equal_target_mix | -4.45 | [-11.40, 1.61] |
| fold_02 | trailing20 | -6.26 | [-16.35, 2.20] |
| fold_02 | equal50 | -4.24 | [-8.22, -0.53] |
| fold_03 | equal_target_mix | 0.25 | [-3.97, 4.10] |
| fold_03 | trailing20 | -0.45 | [-7.79, 6.28] |
| fold_03 | cash | -0.53 | [-6.54, 6.68] |
| fold_04 | equal_target_mix | -2.15 | [-7.03, 2.01] |
| fold_04 | trailing20 | -2.15 | [-11.27, 7.03] |
| fold_04 | vol10 | -4.99 | [-16.55, 3.77] |

All eight selected-PPO contrasts against the mixture/trailing selector span zero. The negative fold-2 interval versus validation-fixed equal50 is descriptive exploratory evidence, not a corrected confirmatory test.

## What the policies actually did

- Fold 1: seeds 7/17 selected vol10 at all 98 executions; seed29 selected trend10 at all98.
- Fold 2: seed7 selected vol10 at all97 executions; seeds17/29 used vol10 at96 executions and cash once.
- Fold 3: seeds7/17 selected cash at all97 executions; seed29 selected vol10 at all97.
- Fold 4: seeds7/29 selected cash at all87 executions; seed17 used vol10 at86 and equal50 once.
These counts refer to the validation-selected PPO objective and actual rebalance executions. Ignored daily proposals are separately recorded in raw ledgers and `selection_counts.csv`. Cash-heavy results are reduced exposure, not automatic success. Training-seed variation must not be interpreted as independent market evidence.

## Cost stress (same checkpoints and selections)

| One-way bps | Fold1 PPO mean return | Fold2 | Fold3 | Fold4 |
|---:|---:|---:|---:|---:|
| 5 | 28.61% | -16.39% | -1.41% | 4.23% |
| 10 | 28.22% | -16.60% | -1.49% | 4.14% |
| 20 | 27.44% | -17.01% | -1.65% | 3.98% |

## Artifacts and boundaries

`runs/research_v6/` contains all checkpoints, signed records, validation previews, test ledgers, audit sidecars, source/data manifest, frozen choices, complete metric CSV, selection counts and all72 prespecified contrasts. `reports/runs/research_v6/` is the compact versioned report mirror with a full run inventory. `REPORT.md` and `all_metrics.csv` report both objectives and every deterministic comparator, not only validation selections.

No live trading, paid-resource use, repository push or expansion of the asset pool occurred. The unresolved user risk/return acceptance threshold is not invented. The narrow conclusion is that this finite v6 experiment did not demonstrate useful learned dynamic expert selection; it does not prove the broader idea impossible or justify post-hoc budget/parameter changes.
