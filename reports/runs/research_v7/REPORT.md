# v7 causal action-mask controlled experiment

Retrospective exploratory historical development, not untouched OOS or live efficacy.
Official sb3-contrib 2.7.1 MaskablePPO for both all-true and phase-mask arms; not a FinRL training entry.
48 fits × 20,000 DAILY market transitions = 960,000; no macrostepping.
Effective v6 ent_coef=0.0: nominal FinRL .01 is not merged when adapter supplies kwargs.
All 48 checkpoints and all validation selections were frozen before any v7 test evaluation.
Validation-only baseline previews were observed before training; the registered family was not changed.
Costs are one-way bps. Seeds describe training randomness, not independent market histories.
Deterministic rules have one path per fold/cost, not three independent seed replicates.

| Cost | Fold | Strategy | Return mean (seed range) | Max drawdown mean | Mean equity exposure | Daily turnover |
|---:|---|---|---:|---:|---:|---:|
| 5 | fold_01 | cash | 0.00% (0.00%–0.00%) | 0.00% | 0.00% | 0.0000 |
| 5 | fold_01 | equal50 | 31.70% (31.70%–31.70%) | 7.63% | 50.10% | 0.0024 |
| 5 | fold_01 | vol10 | 35.64% (35.64%–35.64%) | 12.01% | 53.70% | 0.0094 |
| 5 | fold_01 | trend10 | 14.56% (14.56%–14.56%) | 13.05% | 39.04% | 0.0200 |
| 5 | fold_01 | equal_target_mix | 20.04% (20.04%–20.04%) | 7.75% | 35.74% | 0.0071 |
| 5 | fold_01 | trailing20 | 16.60% (16.60%–16.60%) | 14.27% | 40.21% | 0.0304 |
| 5 | fold_01 | lognet_alltrue | 28.61% (14.56%–35.64%) | 12.36% | 48.81% | 0.0130 |
| 5 | fold_01 | lognet_phase | 34.98% (33.66%–35.64%) | 12.01% | 53.55% | 0.0094 |
| 5 | fold_01 | drawdown_alltrue | 29.16% (16.20%–35.64%) | 12.36% | 48.96% | 0.0135 |
| 5 | fold_01 | drawdown_phase | 34.32% (33.66%–35.64%) | 12.01% | 53.41% | 0.0094 |
| 5 | fold_02 | cash | 0.00% (0.00%–0.00%) | 0.00% | 0.00% | 0.0000 |
| 5 | fold_02 | equal50 | -9.00% (-9.00%–-9.00%) | 16.22% | 50.01% | 0.0022 |
| 5 | fold_02 | vol10 | -15.50% (-15.50%–-15.50%) | 23.08% | 61.89% | 0.0102 |
| 5 | fold_02 | trend10 | -8.56% (-8.56%–-8.56%) | 12.26% | 24.14% | 0.0215 |
| 5 | fold_02 | equal_target_mix | -8.23% (-8.23%–-8.23%) | 12.93% | 34.01% | 0.0071 |
| 5 | fold_02 | trailing20 | -5.07% (-5.07%–-5.07%) | 10.99% | 28.66% | 0.0344 |
| 5 | fold_02 | lognet_alltrue | -16.39% (-16.84%–-15.50%) | 23.08% | 61.39% | 0.0102 |
| 5 | fold_02 | lognet_phase | -13.67% (-15.50%–-10.02%) | 19.48% | 49.06% | 0.0140 |
| 5 | fold_02 | drawdown_alltrue | -13.63% (-16.84%–-8.56%) | 19.48% | 49.06% | 0.0140 |
| 5 | fold_02 | drawdown_phase | -13.63% (-16.84%–-8.56%) | 19.48% | 49.06% | 0.0140 |
| 5 | fold_03 | cash | 0.00% (0.00%–0.00%) | 0.00% | 0.00% | 0.0000 |
| 5 | fold_03 | equal50 | 2.90% (2.90%–2.90%) | 11.97% | 50.01% | 0.0022 |
| 5 | fold_03 | vol10 | -4.23% (-4.23%–-4.23%) | 20.91% | 70.50% | 0.0105 |
| 5 | fold_03 | trend10 | -7.12% (-7.12%–-7.12%) | 12.30% | 26.84% | 0.0205 |
| 5 | fold_03 | equal_target_mix | -1.86% (-1.86%–-1.86%) | 11.31% | 36.84% | 0.0068 |
| 5 | fold_03 | trailing20 | -0.61% (-0.61%–-0.61%) | 6.21% | 25.82% | 0.0329 |
| 5 | fold_03 | lognet_alltrue | -1.93% (-5.78%–0.00%) | 6.97% | 23.30% | 0.0035 |
| 5 | fold_03 | lognet_phase | -5.09% (-5.78%–-4.23%) | 20.91% | 70.16% | 0.0106 |
| 5 | fold_03 | drawdown_alltrue | -1.41% (-4.23%–0.00%) | 6.97% | 23.50% | 0.0035 |
| 5 | fold_03 | drawdown_phase | -5.09% (-5.78%–-4.23%) | 20.91% | 70.16% | 0.0106 |
| 5 | fold_04 | cash | 0.00% (0.00%–0.00%) | 0.00% | 0.00% | 0.0000 |
| 5 | fold_04 | equal50 | 7.63% (7.63%–7.63%) | 9.14% | 50.01% | 0.0023 |
| 5 | fold_04 | vol10 | 12.66% (12.66%–12.66%) | 9.58% | 61.53% | 0.0101 |
| 5 | fold_04 | trend10 | 10.98% (10.98%–10.98%) | 7.08% | 45.70% | 0.0215 |
| 5 | fold_04 | equal_target_mix | 7.90% (7.90%–7.90%) | 5.91% | 39.33% | 0.0076 |
| 5 | fold_04 | trailing20 | 7.64% (7.64%–7.64%) | 6.64% | 39.21% | 0.0311 |
| 5 | fold_04 | lognet_alltrue | 4.23% (0.00%–12.68%) | 3.19% | 20.50% | 0.0034 |
| 5 | fold_04 | lognet_phase | 6.87% (0.00%–12.66%) | 6.24% | 37.09% | 0.0041 |
| 5 | fold_04 | drawdown_alltrue | 4.23% (0.00%–12.68%) | 3.19% | 20.50% | 0.0034 |
| 5 | fold_04 | drawdown_phase | 4.46% (0.00%–13.37%) | 3.19% | 20.29% | 0.0034 |
| 10 | fold_01 | cash | 0.00% (0.00%–0.00%) | 0.00% | 0.00% | 0.0000 |
| 10 | fold_01 | equal50 | 31.63% (31.63%–31.63%) | 7.63% | 50.10% | 0.0024 |
| 10 | fold_01 | vol10 | 35.33% (35.33%–35.33%) | 12.05% | 53.70% | 0.0094 |
| 10 | fold_01 | trend10 | 14.00% (14.00%–14.00%) | 13.32% | 39.04% | 0.0200 |
| 10 | fold_01 | equal_target_mix | 19.83% (19.83%–19.83%) | 7.77% | 35.74% | 0.0070 |
| 10 | fold_01 | trailing20 | 15.74% (15.74%–15.74%) | 14.34% | 40.21% | 0.0304 |
| 10 | fold_01 | lognet_alltrue | 28.22% (14.00%–35.33%) | 12.47% | 48.81% | 0.0130 |
| 10 | fold_01 | lognet_phase | 34.67% (33.36%–35.33%) | 12.05% | 53.55% | 0.0094 |
| 10 | fold_01 | drawdown_alltrue | 28.75% (15.59%–35.33%) | 12.47% | 48.96% | 0.0135 |
| 10 | fold_01 | drawdown_phase | 34.01% (33.36%–35.33%) | 12.05% | 53.41% | 0.0094 |
| 10 | fold_02 | cash | 0.00% (0.00%–0.00%) | 0.00% | 0.00% | 0.0000 |
| 10 | fold_02 | equal50 | -9.05% (-9.05%–-9.05%) | 16.24% | 50.01% | 0.0022 |
| 10 | fold_02 | vol10 | -15.70% (-15.70%–-15.70%) | 23.22% | 61.89% | 0.0102 |
| 10 | fold_02 | trend10 | -9.04% (-9.04%–-9.04%) | 12.68% | 24.14% | 0.0215 |
| 10 | fold_02 | equal_target_mix | -8.38% (-8.38%–-8.38%) | 13.05% | 34.01% | 0.0071 |
| 10 | fold_02 | trailing20 | -5.86% (-5.86%–-5.86%) | 11.54% | 28.66% | 0.0344 |
| 10 | fold_02 | lognet_alltrue | -16.60% (-17.05%–-15.70%) | 23.22% | 61.39% | 0.0102 |
| 10 | fold_02 | lognet_phase | -13.96% (-15.70%–-10.49%) | 19.71% | 49.06% | 0.0140 |
| 10 | fold_02 | drawdown_alltrue | -13.93% (-17.05%–-9.04%) | 19.71% | 49.06% | 0.0140 |
| 10 | fold_02 | drawdown_phase | -13.93% (-17.05%–-9.04%) | 19.71% | 49.06% | 0.0140 |
| 10 | fold_03 | cash | 0.00% (0.00%–0.00%) | 0.00% | 0.00% | 0.0000 |
| 10 | fold_03 | equal50 | 2.85% (2.85%–2.85%) | 11.98% | 50.01% | 0.0022 |
| 10 | fold_03 | vol10 | -4.48% (-4.48%–-4.48%) | 21.03% | 70.50% | 0.0105 |
| 10 | fold_03 | trend10 | -7.58% (-7.58%–-7.58%) | 12.67% | 26.84% | 0.0205 |
| 10 | fold_03 | equal_target_mix | -2.02% (-2.02%–-2.02%) | 11.43% | 36.84% | 0.0068 |
| 10 | fold_03 | trailing20 | -1.40% (-1.40%–-1.40%) | 6.57% | 25.82% | 0.0329 |
| 10 | fold_03 | lognet_alltrue | -2.01% (-6.02%–0.00%) | 7.01% | 23.30% | 0.0035 |
| 10 | fold_03 | lognet_phase | -5.33% (-6.02%–-4.48%) | 21.03% | 70.16% | 0.0106 |
| 10 | fold_03 | drawdown_alltrue | -1.49% (-4.48%–0.00%) | 7.01% | 23.50% | 0.0035 |
| 10 | fold_03 | drawdown_phase | -5.33% (-6.02%–-4.48%) | 21.03% | 70.16% | 0.0106 |
| 10 | fold_04 | cash | 0.00% (0.00%–0.00%) | 0.00% | 0.00% | 0.0000 |
| 10 | fold_04 | equal50 | 7.58% (7.58%–7.58%) | 9.14% | 50.01% | 0.0023 |
| 10 | fold_04 | vol10 | 12.41% (12.41%–12.41%) | 9.65% | 61.53% | 0.0101 |
| 10 | fold_04 | trend10 | 10.47% (10.47%–10.47%) | 7.15% | 45.70% | 0.0215 |
| 10 | fold_04 | equal_target_mix | 7.72% (7.72%–7.72%) | 5.97% | 39.33% | 0.0076 |
| 10 | fold_04 | trailing20 | 5.85% (5.85%–5.85%) | 7.68% | 39.35% | 0.0317 |
| 10 | fold_04 | lognet_alltrue | 4.14% (0.00%–12.43%) | 3.22% | 20.50% | 0.0034 |
| 10 | fold_04 | lognet_phase | 6.77% (0.00%–12.41%) | 6.26% | 37.09% | 0.0041 |
| 10 | fold_04 | drawdown_alltrue | 4.14% (0.00%–12.43%) | 3.22% | 20.50% | 0.0034 |
| 10 | fold_04 | drawdown_phase | 4.38% (0.00%–13.13%) | 3.22% | 20.29% | 0.0034 |
| 20 | fold_01 | cash | 0.00% (0.00%–0.00%) | 0.00% | 0.00% | 0.0000 |
| 20 | fold_01 | equal50 | 31.48% (31.48%–31.48%) | 7.64% | 50.10% | 0.0024 |
| 20 | fold_01 | vol10 | 34.71% (34.71%–34.71%) | 12.13% | 53.70% | 0.0094 |
| 20 | fold_01 | trend10 | 12.90% (12.90%–12.90%) | 13.85% | 39.04% | 0.0200 |
| 20 | fold_01 | equal_target_mix | 19.42% (19.42%–19.42%) | 7.82% | 35.74% | 0.0070 |
| 20 | fold_01 | trailing20 | 14.04% (14.04%–14.04%) | 14.46% | 40.21% | 0.0304 |
| 20 | fold_01 | lognet_alltrue | 27.44% (12.90%–34.71%) | 12.70% | 48.81% | 0.0130 |
| 20 | fold_01 | lognet_phase | 34.05% (32.75%–34.71%) | 12.13% | 53.55% | 0.0094 |
| 20 | fold_01 | drawdown_alltrue | 27.93% (14.37%–34.71%) | 12.70% | 48.96% | 0.0135 |
| 20 | fold_01 | drawdown_phase | 33.40% (32.75%–34.71%) | 12.13% | 53.41% | 0.0094 |
| 20 | fold_02 | cash | 0.00% (0.00%–0.00%) | 0.00% | 0.00% | 0.0000 |
| 20 | fold_02 | equal50 | -9.14% (-9.14%–-9.14%) | 16.28% | 50.01% | 0.0022 |
| 20 | fold_02 | vol10 | -16.12% (-16.12%–-16.12%) | 23.50% | 61.89% | 0.0102 |
| 20 | fold_02 | trend10 | -9.98% (-9.98%–-9.98%) | 13.50% | 24.14% | 0.0215 |
| 20 | fold_02 | equal_target_mix | -8.70% (-8.70%–-8.70%) | 13.27% | 34.01% | 0.0071 |
| 20 | fold_02 | trailing20 | -7.41% (-7.41%–-7.41%) | 12.64% | 28.66% | 0.0344 |
| 20 | fold_02 | lognet_alltrue | -17.01% (-17.46%–-16.12%) | 23.50% | 61.39% | 0.0102 |
| 20 | fold_02 | lognet_phase | -14.55% (-16.12%–-11.41%) | 20.17% | 49.06% | 0.0140 |
| 20 | fold_02 | drawdown_alltrue | -14.52% (-17.46%–-9.98%) | 20.17% | 49.06% | 0.0140 |
| 20 | fold_02 | drawdown_phase | -14.52% (-17.46%–-9.98%) | 20.17% | 49.06% | 0.0140 |
| 20 | fold_03 | cash | 0.00% (0.00%–0.00%) | 0.00% | 0.00% | 0.0000 |
| 20 | fold_03 | equal50 | 2.74% (2.74%–2.74%) | 12.00% | 50.01% | 0.0022 |
| 20 | fold_03 | vol10 | -4.96% (-4.96%–-4.96%) | 21.31% | 70.50% | 0.0105 |
| 20 | fold_03 | trend10 | -8.49% (-8.49%–-8.49%) | 13.40% | 26.84% | 0.0205 |
| 20 | fold_03 | equal_target_mix | -2.34% (-2.34%–-2.34%) | 11.66% | 36.84% | 0.0068 |
| 20 | fold_03 | trailing20 | -2.95% (-2.95%–-2.95%) | 7.44% | 25.82% | 0.0328 |
| 20 | fold_03 | lognet_alltrue | -2.16% (-6.49%–0.00%) | 7.10% | 23.30% | 0.0035 |
| 20 | fold_03 | lognet_phase | -5.81% (-6.49%–-4.96%) | 21.31% | 70.16% | 0.0106 |
| 20 | fold_03 | drawdown_alltrue | -1.65% (-4.96%–0.00%) | 7.10% | 23.50% | 0.0035 |
| 20 | fold_03 | drawdown_phase | -5.81% (-6.49%–-4.96%) | 21.31% | 70.16% | 0.0106 |
| 20 | fold_04 | cash | 0.00% (0.00%–0.00%) | 0.00% | 0.00% | 0.0000 |
| 20 | fold_04 | equal50 | 7.47% (7.47%–7.47%) | 9.15% | 50.01% | 0.0023 |
| 20 | fold_04 | vol10 | 11.92% (11.92%–11.92%) | 9.79% | 61.53% | 0.0101 |
| 20 | fold_04 | trend10 | 9.44% (9.44%–9.44%) | 7.30% | 45.70% | 0.0215 |
| 20 | fold_04 | equal_target_mix | 7.37% (7.37%–7.37%) | 6.08% | 39.33% | 0.0076 |
| 20 | fold_04 | trailing20 | 5.49% (5.49%–5.49%) | 6.97% | 39.21% | 0.0311 |
| 20 | fold_04 | lognet_alltrue | 3.98% (0.00%–11.94%) | 3.26% | 20.50% | 0.0034 |
| 20 | fold_04 | lognet_phase | 6.57% (0.00%–11.92%) | 6.31% | 37.09% | 0.0041 |
| 20 | fold_04 | drawdown_alltrue | 3.98% (0.00%–11.94%) | 3.26% | 20.50% | 0.0034 |
| 20 | fold_04 | drawdown_phase | 4.21% (0.00%–12.63%) | 3.26% | 20.29% | 0.0034 |

## Frozen validation selections

| Fold | Learned-only choice | With-rules choice |
|---|---|---|
| fold_01 | lognet_alltrue | cash |
| fold_02 | lognet_alltrue | equal50 |
| fold_03 | drawdown_alltrue | cash |
| fold_04 | lognet_phase | vol10 |

## Paired masking effects (phase minus alltrue)

| Cost | Fold | Objective | Mean paired total-return difference | Annualized arithmetic excess (95% block interval) |
|---:|---|---|---:|---:|
| 5 | fold_01 | lognet | 6.37% | 2.71% (0.87%, 4.36%) |
| 5 | fold_01 | drawdown | 5.16% | 2.21% (0.73%, 4.28%) |
| 5 | fold_02 | lognet | 2.72% | 1.50% (-2.01%, 5.02%) |
| 5 | fold_02 | drawdown | 0.00% | 0.00% (0.00%, 0.00%) |
| 5 | fold_03 | lognet | -3.17% | -1.25% (-13.09%, 13.32%) |
| 5 | fold_03 | drawdown | -3.68% | -1.53% (-13.19%, 13.30%) |
| 5 | fold_04 | lognet | 2.65% | 1.60% (-1.89%, 5.71%) |
| 5 | fold_04 | drawdown | 0.23% | 0.12% (-0.05%, 0.12%) |
| 10 | fold_01 | lognet | 6.45% | 2.76% (0.92%, 4.43%) |
| 10 | fold_01 | drawdown | 5.27% | 2.26% (0.78%, 4.34%) |
| 10 | fold_02 | lognet | 2.63% | 1.46% (-2.04%, 4.99%) |
| 10 | fold_02 | drawdown | 0.00% | 0.00% (0.00%, 0.00%) |
| 10 | fold_03 | lognet | -3.33% | -1.34% (-13.16%, 13.23%) |
| 10 | fold_03 | drawdown | -3.84% | -1.62% (-13.26%, 13.21%) |
| 10 | fold_04 | lognet | 2.63% | 1.59% (-1.90%, 5.70%) |
| 10 | fold_04 | drawdown | 0.23% | 0.12% (-0.06%, 0.12%) |
| 20 | fold_01 | lognet | 6.62% | 2.85% (1.02%, 4.56%) |
| 20 | fold_01 | drawdown | 5.47% | 2.37% (0.86%, 4.45%) |
| 20 | fold_02 | lognet | 2.46% | 1.36% (-2.12%, 4.95%) |
| 20 | fold_02 | drawdown | 0.00% | 0.00% (0.00%, 0.00%) |
| 20 | fold_03 | lognet | -3.65% | -1.52% (-13.32%, 13.05%) |
| 20 | fold_03 | drawdown | -4.16% | -1.80% (-13.41%, 13.02%) |
| 20 | fold_04 | lognet | 2.59% | 1.57% (-1.91%, 5.69%) |
| 20 | fold_04 | drawdown | 0.23% | 0.12% (-0.08%, 0.12%) |

## Executed choices and costs

Frequencies below use executed selections only. Ignored dummy requests are never allocations.

| Cost | Fold | Candidate | Cash | Equal50 | Vol10 | Trend10 | Mean switches | Sum daily cost fractions |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 5 | fold_01 | lognet_alltrue | 0.0% | 0.0% | 66.7% | 33.3% | 0.0 | 0.315% |
| 5 | fold_01 | lognet_phase | 0.0% | 0.0% | 99.7% | 0.3% | 0.3 | 0.229% |
| 5 | fold_01 | drawdown_alltrue | 0.0% | 0.0% | 67.0% | 33.0% | 0.3 | 0.329% |
| 5 | fold_01 | drawdown_phase | 0.3% | 0.0% | 99.3% | 0.3% | 0.7 | 0.230% |
| 5 | fold_02 | lognet_alltrue | 0.7% | 0.0% | 99.3% | 0.0% | 0.7 | 0.247% |
| 5 | fold_02 | lognet_phase | 0.3% | 0.0% | 66.7% | 33.0% | 0.3 | 0.338% |
| 5 | fold_02 | drawdown_alltrue | 0.3% | 0.0% | 66.0% | 33.7% | 0.7 | 0.338% |
| 5 | fold_02 | drawdown_phase | 0.3% | 0.0% | 66.0% | 33.7% | 0.7 | 0.338% |
| 5 | fold_03 | lognet_alltrue | 67.0% | 0.0% | 33.0% | 0.0% | 0.3 | 0.085% |
| 5 | fold_03 | lognet_phase | 0.3% | 0.0% | 99.3% | 0.3% | 0.7 | 0.255% |
| 5 | fold_03 | drawdown_alltrue | 66.7% | 0.0% | 33.3% | 0.0% | 0.0 | 0.085% |
| 5 | fold_03 | drawdown_phase | 0.3% | 0.0% | 99.3% | 0.3% | 0.7 | 0.255% |
| 5 | fold_04 | lognet_alltrue | 66.7% | 0.4% | 33.0% | 0.0% | 0.3 | 0.073% |
| 5 | fold_04 | lognet_phase | 33.3% | 33.0% | 33.3% | 0.4% | 0.3 | 0.089% |
| 5 | fold_04 | drawdown_alltrue | 66.7% | 0.4% | 33.0% | 0.0% | 0.3 | 0.073% |
| 5 | fold_04 | drawdown_phase | 67.0% | 0.4% | 32.6% | 0.0% | 0.7 | 0.073% |
| 10 | fold_01 | lognet_alltrue | 0.0% | 0.0% | 66.7% | 33.3% | 0.0 | 0.630% |
| 10 | fold_01 | lognet_phase | 0.0% | 0.0% | 99.7% | 0.3% | 0.3 | 0.459% |
| 10 | fold_01 | drawdown_alltrue | 0.0% | 0.0% | 67.0% | 33.0% | 0.3 | 0.658% |
| 10 | fold_01 | drawdown_phase | 0.3% | 0.0% | 99.3% | 0.3% | 0.7 | 0.459% |
| 10 | fold_02 | lognet_alltrue | 0.7% | 0.0% | 99.3% | 0.0% | 0.7 | 0.494% |
| 10 | fold_02 | lognet_phase | 0.3% | 0.0% | 66.7% | 33.0% | 0.3 | 0.676% |
| 10 | fold_02 | drawdown_alltrue | 0.3% | 0.0% | 66.0% | 33.7% | 0.7 | 0.676% |
| 10 | fold_02 | drawdown_phase | 0.3% | 0.0% | 66.0% | 33.7% | 0.7 | 0.676% |
| 10 | fold_03 | lognet_alltrue | 67.0% | 0.0% | 33.0% | 0.0% | 0.3 | 0.170% |
| 10 | fold_03 | lognet_phase | 0.3% | 0.0% | 99.3% | 0.3% | 0.7 | 0.510% |
| 10 | fold_03 | drawdown_alltrue | 66.7% | 0.0% | 33.3% | 0.0% | 0.0 | 0.170% |
| 10 | fold_03 | drawdown_phase | 0.3% | 0.0% | 99.3% | 0.3% | 0.7 | 0.510% |
| 10 | fold_04 | lognet_alltrue | 66.7% | 0.4% | 33.0% | 0.0% | 0.3 | 0.146% |
| 10 | fold_04 | lognet_phase | 33.3% | 33.0% | 33.3% | 0.4% | 0.3 | 0.179% |
| 10 | fold_04 | drawdown_alltrue | 66.7% | 0.4% | 33.0% | 0.0% | 0.3 | 0.146% |
| 10 | fold_04 | drawdown_phase | 67.0% | 0.4% | 32.6% | 0.0% | 0.7 | 0.146% |
| 20 | fold_01 | lognet_alltrue | 0.0% | 0.0% | 66.7% | 33.3% | 0.0 | 1.260% |
| 20 | fold_01 | lognet_phase | 0.0% | 0.0% | 99.7% | 0.3% | 0.3 | 0.917% |
| 20 | fold_01 | drawdown_alltrue | 0.0% | 0.0% | 67.0% | 33.0% | 0.3 | 1.316% |
| 20 | fold_01 | drawdown_phase | 0.3% | 0.0% | 99.3% | 0.3% | 0.7 | 0.918% |
| 20 | fold_02 | lognet_alltrue | 0.7% | 0.0% | 99.3% | 0.0% | 0.7 | 0.987% |
| 20 | fold_02 | lognet_phase | 0.3% | 0.0% | 66.7% | 33.0% | 0.3 | 1.351% |
| 20 | fold_02 | drawdown_alltrue | 0.3% | 0.0% | 66.0% | 33.7% | 0.7 | 1.351% |
| 20 | fold_02 | drawdown_phase | 0.3% | 0.0% | 66.0% | 33.7% | 0.7 | 1.351% |
| 20 | fold_03 | lognet_alltrue | 67.0% | 0.0% | 33.0% | 0.0% | 0.3 | 0.340% |
| 20 | fold_03 | lognet_phase | 0.3% | 0.0% | 99.3% | 0.3% | 0.7 | 1.019% |
| 20 | fold_03 | drawdown_alltrue | 66.7% | 0.0% | 33.3% | 0.0% | 0.0 | 0.340% |
| 20 | fold_03 | drawdown_phase | 0.3% | 0.0% | 99.3% | 0.3% | 0.7 | 1.019% |
| 20 | fold_04 | lognet_alltrue | 66.7% | 0.4% | 33.0% | 0.0% | 0.3 | 0.293% |
| 20 | fold_04 | lognet_phase | 33.3% | 33.0% | 33.3% | 0.4% | 0.3 | 0.358% |
| 20 | fold_04 | drawdown_alltrue | 66.7% | 0.4% | 33.0% | 0.0% | 0.3 | 0.293% |
| 20 | fold_04 | drawdown_phase | 67.0% | 0.4% | 32.6% | 0.0% | 0.7 | 0.293% |

Primary controlled contrasts: phase minus alltrue separately for lognet and drawdown; every seed/fold/cost is retained.
Additional prespecified contrasts: equal-target mix, trailing-20 shadow selector, and the best validation-fixed expert.
Moving-block intervals use 10-day blocks / 2000 draws, concern annualized arithmetic excess (not CAGR), and are exploratory without multiple-testing correction.
All models and all six comparators are reported; no test winner is selected.
Every saved actual-account test ledger was independently reconstructed from raw corporate-action data.
