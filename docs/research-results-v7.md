# v7 results: causal action-mask controlled study

## Result first

**Masking changed some returns, but did not reliably improve the selector or its risk control.** At 5 bps, mean paired total-return differences (phase minus all-true) were +6.37/+2.72/−3.17/+2.65 percentage points for lognet and +5.16/0.00/−3.68/+0.23 points for drawdown across folds 1–4. Fold 1 gains depend strongly on seed 29. In fold 3, masked mean drawdown rises from about 6.97% to 20.91%; mean equity exposure rises from about 23% to 70%. Executed expert switches remain only 0–0.67 on average per entire test fold: this is not evidence of successful dynamic expert selection.

Fold 1 paired block intervals exclude zero for both objectives at all three costs; other intervals include zero (one contrast is exactly identical). These are conditional exploratory intervals on annualized arithmetic daily excess, not intervals on the total-return differences below, and have no multiplicity correction. Repeatedly inspected historical data and seed dependence prohibit a confirmatory efficacy claim.

Validation-with-rules selection chose a fixed rule in every fold: cash, equal50, cash, vol10. Their 5-bps test returns were 0.00%, −9.00%, 0.00%, +12.66%, respectively. No test winner was used for selection.

## Primary paired effects across costs

Each entry is the mean of three paired-seed total-return differences, in percentage points. Folds are kept separate; overlapping market histories are not combined into an independent aggregate.

| Cost (one-way bps) | Fold | Objective | Phase − all-true return (pp) | Annualized arithmetic excess 95% block interval |
|---:|---|---|---:|---:|
| 5 | fold_01 | lognet | +6.37 | [+0.87%, +4.36%] |
| 5 | fold_01 | drawdown | +5.16 | [+0.73%, +4.28%] |
| 5 | fold_02 | lognet | +2.72 | [-2.01%, +5.02%] |
| 5 | fold_02 | drawdown | +0.00 | [+0.00%, +0.00%] |
| 5 | fold_03 | lognet | -3.17 | [-13.09%, +13.32%] |
| 5 | fold_03 | drawdown | -3.68 | [-13.19%, +13.30%] |
| 5 | fold_04 | lognet | +2.65 | [-1.89%, +5.71%] |
| 5 | fold_04 | drawdown | +0.23 | [-0.05%, +0.12%] |
| 10 | fold_01 | lognet | +6.45 | [+0.92%, +4.43%] |
| 10 | fold_01 | drawdown | +5.27 | [+0.78%, +4.34%] |
| 10 | fold_02 | lognet | +2.63 | [-2.04%, +4.99%] |
| 10 | fold_02 | drawdown | +0.00 | [+0.00%, +0.00%] |
| 10 | fold_03 | lognet | -3.33 | [-13.16%, +13.23%] |
| 10 | fold_03 | drawdown | -3.84 | [-13.26%, +13.21%] |
| 10 | fold_04 | lognet | +2.63 | [-1.90%, +5.70%] |
| 10 | fold_04 | drawdown | +0.23 | [-0.06%, +0.12%] |
| 20 | fold_01 | lognet | +6.62 | [+1.02%, +4.56%] |
| 20 | fold_01 | drawdown | +5.47 | [+0.86%, +4.45%] |
| 20 | fold_02 | lognet | +2.46 | [-2.12%, +4.95%] |
| 20 | fold_02 | drawdown | +0.00 | [+0.00%, +0.00%] |
| 20 | fold_03 | lognet | -3.65 | [-13.32%, +13.05%] |
| 20 | fold_03 | drawdown | -4.16 | [-13.41%, +13.02%] |
| 20 | fold_04 | lognet | +2.59 | [-1.91%, +5.69%] |
| 20 | fold_04 | drawdown | +0.23 | [-0.08%, +0.12%] |

## All learned candidates at 5 bps

Seed means; every individual seed is retained in `all_metrics.csv`.

| Fold | Candidate | Net return | Max drawdown | Equity exposure | Executed switches |
|---|---|---:|---:|---:|---:|
| fold_01 | lognet_alltrue | 28.61% | 12.36% | 48.81% | 0.00 |
| fold_01 | lognet_phase | 34.98% | 12.01% | 53.55% | 0.33 |
| fold_01 | drawdown_alltrue | 29.16% | 12.36% | 48.96% | 0.33 |
| fold_01 | drawdown_phase | 34.32% | 12.01% | 53.41% | 0.67 |
| fold_02 | lognet_alltrue | -16.39% | 23.08% | 61.39% | 0.67 |
| fold_02 | lognet_phase | -13.67% | 19.48% | 49.06% | 0.33 |
| fold_02 | drawdown_alltrue | -13.63% | 19.48% | 49.06% | 0.67 |
| fold_02 | drawdown_phase | -13.63% | 19.48% | 49.06% | 0.67 |
| fold_03 | lognet_alltrue | -1.93% | 6.97% | 23.30% | 0.33 |
| fold_03 | lognet_phase | -5.09% | 20.91% | 70.16% | 0.67 |
| fold_03 | drawdown_alltrue | -1.41% | 6.97% | 23.50% | 0.00 |
| fold_03 | drawdown_phase | -5.09% | 20.91% | 70.16% | 0.67 |
| fold_04 | lognet_alltrue | 4.23% | 3.19% | 20.50% | 0.33 |
| fold_04 | lognet_phase | 6.87% | 6.24% | 37.09% | 0.33 |
| fold_04 | drawdown_alltrue | 4.23% | 3.19% | 20.50% | 0.33 |
| fold_04 | drawdown_phase | 4.46% | 3.19% | 20.29% | 0.67 |

## Frozen selection results at 5 bps

Rules appear once per fold, not three artificial seed replicates. Learned-only and with-rules selections remain separate.

| Fold | Selection | Chosen strategy | Net return | Max drawdown | Equity exposure |
|---|---|---|---:|---:|---:|
| fold_01 | selected_on_validation | lognet_alltrue | 28.61% | 12.36% | 48.81% |
| fold_01 | selected_with_rules | cash | 0.00% | 0.00% | 0.00% |
| fold_02 | selected_on_validation | lognet_alltrue | -16.39% | 23.08% | 61.39% |
| fold_02 | selected_with_rules | equal50 | -9.00% | 16.22% | 50.01% |
| fold_03 | selected_on_validation | drawdown_alltrue | -1.41% | 6.97% | 23.50% |
| fold_03 | selected_with_rules | cash | 0.00% | 0.00% | 0.00% |
| fold_04 | selected_on_validation | lognet_phase | 6.87% | 6.24% | 37.09% |
| fold_04 | selected_with_rules | vol10 | 12.66% | 9.58% | 61.53% |

## Design and verification

- Same official sb3-contrib 2.7.1 MaskablePPO backend in both arms; only current pre-step masks differ within each objective. This is not a FinRL training-entry run.
- Effective frozen v6 configuration was verified by instantiation: entropy coefficient is **0.0**, not nominal FinRL 0.01 (defaults are not merged when model kwargs are supplied). Other PPO settings, objectives, reward scale, daily transitions and 126-transition episodes match v6.
- 48 retained fits × exactly 20,000 daily transitions = 960,000 retained-fit transitions. Every seed/fold group has identical recorded 159-fragment start/end schedules across all four arms.
- One execution interruption left six verified complete fits. Those were reused; unfinished fits restarted under the identical plan/seed. Unknown discarded partial compute is additional to the 960,000 retained-fit total. No cause was established and no test had been opened.
- All sources, raw data, plan, models and validation records froze before test access. Independent pretraining and pretest integrity reviews passed; no evaluation directories existed at the pretest gate. No source changes after preparation.
- 216 test ledgers plus 96 preview/validation ledgers were independently reconstructed from raw opens and corporate actions. Maximum absolute error across test ledgers: 3.10e−14. Six rules were evaluated once per fold/cost.
- Full repository suite passed 444 tests before training and again after reporting (444 passed in 63.13 seconds); focused v7/v6 suite passed 54. Final regression evidence is in `tmp/pytest-v7-final.log`.
- All frozen v5/v6 source hashes remain unchanged. No new assets, live/paid actions, pushes, or publication. Data remain the exact restored three-ETF snapshots; fresh independent price-verification limitations remain disclosed.
- Previously inspected history makes this exploratory development, not untouched out-of-sample evidence. Masking is not established as the cause or cure of prior behavior.

## Artifacts

- `runs/research_v7/REPORT.md`: all candidate/comparator metrics, paired effects, executed selection frequencies and costs.
- `runs/research_v7/all_metrics.csv`, `selected_metrics.csv`, `selection_counts.csv`.
- `runs/research_v7/paired_mask_contrasts.json` and `primary_contrasts.json`.
- `runs/research_v7/manifest.json`, `frozen_selection.json`, signed per-model records/checkpoints, actual-account ledgers and auditor sidecars.
- `runs/research_v7/execution_recovery.json` and `pretraining_review.json`.
- `docs/research-protocol-v7.md`: prespecified design, provenance, selection and reporting rules.
