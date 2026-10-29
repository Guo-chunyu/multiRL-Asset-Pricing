# v8 count-only scaling controlled experiment

Exploratory repeatedly inspected historical data, not untouched OOS or live performance.
24 new fits × 20,000 daily transitions = 480,000, paired to 24 byte-identical reused v7 phase controls.
Only the history-count feature is divided by 20. No new features, entropy, rewards, schedules or assets.
All source/data/control identities and 48 candidate checkpoints/validation records freeze before v8 test replays.
Rules count once; seeds represent optimizer randomness, not independent market histories.
Independent raw-ledger audit is an engineering check, not evidence of alpha.

| Cost | Fold | Strategy | Net total return mean | Max drawdown mean | Exposure mean | Executed switches mean |
|---:|---|---|---:|---:|---:|---:|
| 5 | fold_01 | cash | 0.00% | 0.00% | 0.00% | 0.00 |
| 5 | fold_01 | equal50 | 31.70% | 7.63% | 50.10% | 0.00 |
| 5 | fold_01 | vol10 | 35.64% | 12.01% | 53.70% | 0.00 |
| 5 | fold_01 | trend10 | 14.56% | 13.05% | 39.04% | 0.00 |
| 5 | fold_01 | equal_target_mix | 20.04% | 7.75% | 35.74% | 0.00 |
| 5 | fold_01 | trailing20 | 16.60% | 14.27% | 40.21% | 31.00 |
| 5 | fold_01 | lognet_raw | 34.98% | 12.01% | 53.55% | 0.33 |
| 5 | fold_01 | lognet_scaled | 35.64% | 12.01% | 53.70% | 0.00 |
| 5 | fold_01 | drawdown_raw | 34.32% | 12.01% | 53.41% | 0.67 |
| 5 | fold_01 | drawdown_scaled | 37.08% | 11.85% | 54.70% | 2.33 |
| 5 | fold_02 | cash | 0.00% | 0.00% | 0.00% | 0.00 |
| 5 | fold_02 | equal50 | -9.00% | 16.22% | 50.01% | 0.00 |
| 5 | fold_02 | vol10 | -15.50% | 23.08% | 61.89% | 0.00 |
| 5 | fold_02 | trend10 | -8.56% | 12.26% | 24.14% | 0.00 |
| 5 | fold_02 | equal_target_mix | -8.23% | 12.93% | 34.01% | 0.00 |
| 5 | fold_02 | trailing20 | -5.07% | 10.99% | 28.66% | 32.00 |
| 5 | fold_02 | lognet_raw | -13.67% | 19.48% | 49.06% | 0.33 |
| 5 | fold_02 | lognet_scaled | -13.18% | 19.48% | 49.31% | 0.00 |
| 5 | fold_02 | drawdown_raw | -13.63% | 19.48% | 49.06% | 0.67 |
| 5 | fold_02 | drawdown_scaled | -13.61% | 19.43% | 48.37% | 0.67 |
| 5 | fold_03 | cash | 0.00% | 0.00% | 0.00% | 0.00 |
| 5 | fold_03 | equal50 | 2.90% | 11.97% | 50.01% | 0.00 |
| 5 | fold_03 | vol10 | -4.23% | 20.91% | 70.50% | 0.00 |
| 5 | fold_03 | trend10 | -7.12% | 12.30% | 26.84% | 0.00 |
| 5 | fold_03 | equal_target_mix | -1.86% | 11.31% | 36.84% | 0.00 |
| 5 | fold_03 | trailing20 | -0.61% | 6.21% | 25.82% | 23.00 |
| 5 | fold_03 | lognet_raw | -5.09% | 20.91% | 70.16% | 0.67 |
| 5 | fold_03 | lognet_scaled | -4.23% | 20.91% | 70.50% | 0.00 |
| 5 | fold_03 | drawdown_raw | -5.09% | 20.91% | 70.16% | 0.67 |
| 5 | fold_03 | drawdown_scaled | -4.23% | 20.91% | 70.50% | 0.00 |
| 5 | fold_04 | cash | 0.00% | 0.00% | 0.00% | 0.00 |
| 5 | fold_04 | equal50 | 7.63% | 9.14% | 50.01% | 0.00 |
| 5 | fold_04 | vol10 | 12.66% | 9.58% | 61.53% | 0.00 |
| 5 | fold_04 | trend10 | 10.98% | 7.08% | 45.70% | 0.00 |
| 5 | fold_04 | equal_target_mix | 7.90% | 5.91% | 39.33% | 0.00 |
| 5 | fold_04 | trailing20 | 7.64% | 6.64% | 39.21% | 31.00 |
| 5 | fold_04 | lognet_raw | 6.87% | 6.24% | 37.09% | 0.33 |
| 5 | fold_04 | lognet_scaled | 3.36% | 7.98% | 36.73% | 11.33 |
| 5 | fold_04 | drawdown_raw | 4.46% | 3.19% | 20.29% | 0.67 |
| 5 | fold_04 | drawdown_scaled | 3.40% | 3.53% | 18.78% | 3.33 |
| 10 | fold_01 | cash | 0.00% | 0.00% | 0.00% | 0.00 |
| 10 | fold_01 | equal50 | 31.63% | 7.63% | 50.10% | 0.00 |
| 10 | fold_01 | vol10 | 35.33% | 12.05% | 53.70% | 0.00 |
| 10 | fold_01 | trend10 | 14.00% | 13.32% | 39.04% | 0.00 |
| 10 | fold_01 | equal_target_mix | 19.83% | 7.77% | 35.74% | 0.00 |
| 10 | fold_01 | trailing20 | 15.74% | 14.34% | 40.21% | 31.00 |
| 10 | fold_01 | lognet_raw | 34.67% | 12.05% | 53.55% | 0.33 |
| 10 | fold_01 | lognet_scaled | 35.33% | 12.05% | 53.70% | 0.00 |
| 10 | fold_01 | drawdown_raw | 34.01% | 12.05% | 53.41% | 0.67 |
| 10 | fold_01 | drawdown_scaled | 36.77% | 11.88% | 54.70% | 2.33 |
| 10 | fold_02 | cash | 0.00% | 0.00% | 0.00% | 0.00 |
| 10 | fold_02 | equal50 | -9.05% | 16.24% | 50.01% | 0.00 |
| 10 | fold_02 | vol10 | -15.70% | 23.22% | 61.89% | 0.00 |
| 10 | fold_02 | trend10 | -9.04% | 12.68% | 24.14% | 0.00 |
| 10 | fold_02 | equal_target_mix | -8.38% | 13.05% | 34.01% | 0.00 |
| 10 | fold_02 | trailing20 | -5.86% | 11.54% | 28.66% | 32.00 |
| 10 | fold_02 | lognet_raw | -13.96% | 19.71% | 49.06% | 0.33 |
| 10 | fold_02 | lognet_scaled | -13.48% | 19.71% | 49.31% | 0.00 |
| 10 | fold_02 | drawdown_raw | -13.93% | 19.71% | 49.06% | 0.67 |
| 10 | fold_02 | drawdown_scaled | -13.89% | 19.66% | 48.37% | 0.67 |
| 10 | fold_03 | cash | 0.00% | 0.00% | 0.00% | 0.00 |
| 10 | fold_03 | equal50 | 2.85% | 11.98% | 50.01% | 0.00 |
| 10 | fold_03 | vol10 | -4.48% | 21.03% | 70.50% | 0.00 |
| 10 | fold_03 | trend10 | -7.58% | 12.67% | 26.84% | 0.00 |
| 10 | fold_03 | equal_target_mix | -2.02% | 11.43% | 36.84% | 0.00 |
| 10 | fold_03 | trailing20 | -1.40% | 6.57% | 25.82% | 23.00 |
| 10 | fold_03 | lognet_raw | -5.33% | 21.03% | 70.16% | 0.67 |
| 10 | fold_03 | lognet_scaled | -4.48% | 21.03% | 70.50% | 0.00 |
| 10 | fold_03 | drawdown_raw | -5.33% | 21.03% | 70.16% | 0.67 |
| 10 | fold_03 | drawdown_scaled | -4.48% | 21.03% | 70.50% | 0.00 |
| 10 | fold_04 | cash | 0.00% | 0.00% | 0.00% | 0.00 |
| 10 | fold_04 | equal50 | 7.58% | 9.14% | 50.01% | 0.00 |
| 10 | fold_04 | vol10 | 12.41% | 9.65% | 61.53% | 0.00 |
| 10 | fold_04 | trend10 | 10.47% | 7.15% | 45.70% | 0.00 |
| 10 | fold_04 | equal_target_mix | 7.72% | 5.97% | 39.33% | 0.00 |
| 10 | fold_04 | trailing20 | 5.85% | 7.68% | 39.35% | 28.00 |
| 10 | fold_04 | lognet_raw | 6.77% | 6.26% | 37.09% | 0.33 |
| 10 | fold_04 | lognet_scaled | 3.03% | 8.00% | 36.73% | 11.33 |
| 10 | fold_04 | drawdown_raw | 4.38% | 3.22% | 20.29% | 0.67 |
| 10 | fold_04 | drawdown_scaled | 3.25% | 3.58% | 18.78% | 3.33 |
| 20 | fold_01 | cash | 0.00% | 0.00% | 0.00% | 0.00 |
| 20 | fold_01 | equal50 | 31.48% | 7.64% | 50.10% | 0.00 |
| 20 | fold_01 | vol10 | 34.71% | 12.13% | 53.70% | 0.00 |
| 20 | fold_01 | trend10 | 12.90% | 13.85% | 39.04% | 0.00 |
| 20 | fold_01 | equal_target_mix | 19.42% | 7.82% | 35.74% | 0.00 |
| 20 | fold_01 | trailing20 | 14.04% | 14.46% | 40.21% | 31.00 |
| 20 | fold_01 | lognet_raw | 34.05% | 12.13% | 53.55% | 0.33 |
| 20 | fold_01 | lognet_scaled | 34.71% | 12.13% | 53.70% | 0.00 |
| 20 | fold_01 | drawdown_raw | 33.40% | 12.13% | 53.41% | 0.67 |
| 20 | fold_01 | drawdown_scaled | 36.14% | 11.95% | 54.70% | 2.33 |
| 20 | fold_02 | cash | 0.00% | 0.00% | 0.00% | 0.00 |
| 20 | fold_02 | equal50 | -9.14% | 16.28% | 50.01% | 0.00 |
| 20 | fold_02 | vol10 | -16.12% | 23.50% | 61.89% | 0.00 |
| 20 | fold_02 | trend10 | -9.98% | 13.50% | 24.14% | 0.00 |
| 20 | fold_02 | equal_target_mix | -8.70% | 13.27% | 34.01% | 0.00 |
| 20 | fold_02 | trailing20 | -7.41% | 12.64% | 28.66% | 32.00 |
| 20 | fold_02 | lognet_raw | -14.55% | 20.17% | 49.06% | 0.33 |
| 20 | fold_02 | lognet_scaled | -14.07% | 20.17% | 49.31% | 0.00 |
| 20 | fold_02 | drawdown_raw | -14.52% | 20.17% | 49.06% | 0.67 |
| 20 | fold_02 | drawdown_scaled | -14.46% | 20.11% | 48.37% | 0.67 |
| 20 | fold_03 | cash | 0.00% | 0.00% | 0.00% | 0.00 |
| 20 | fold_03 | equal50 | 2.74% | 12.00% | 50.01% | 0.00 |
| 20 | fold_03 | vol10 | -4.96% | 21.31% | 70.50% | 0.00 |
| 20 | fold_03 | trend10 | -8.49% | 13.40% | 26.84% | 0.00 |
| 20 | fold_03 | equal_target_mix | -2.34% | 11.66% | 36.84% | 0.00 |
| 20 | fold_03 | trailing20 | -2.95% | 7.44% | 25.82% | 23.00 |
| 20 | fold_03 | lognet_raw | -5.81% | 21.31% | 70.16% | 0.67 |
| 20 | fold_03 | lognet_scaled | -4.96% | 21.31% | 70.50% | 0.00 |
| 20 | fold_03 | drawdown_raw | -5.81% | 21.31% | 70.16% | 0.67 |
| 20 | fold_03 | drawdown_scaled | -4.96% | 21.31% | 70.50% | 0.00 |
| 20 | fold_04 | cash | 0.00% | 0.00% | 0.00% | 0.00 |
| 20 | fold_04 | equal50 | 7.47% | 9.15% | 50.01% | 0.00 |
| 20 | fold_04 | vol10 | 11.92% | 9.79% | 61.53% | 0.00 |
| 20 | fold_04 | trend10 | 9.44% | 7.30% | 45.70% | 0.00 |
| 20 | fold_04 | equal_target_mix | 7.37% | 6.08% | 39.33% | 0.00 |
| 20 | fold_04 | trailing20 | 5.49% | 6.97% | 39.21% | 32.00 |
| 20 | fold_04 | lognet_raw | 6.57% | 6.31% | 37.09% | 0.33 |
| 20 | fold_04 | lognet_scaled | 2.36% | 8.02% | 36.73% | 11.33 |
| 20 | fold_04 | drawdown_raw | 4.21% | 3.26% | 20.29% | 0.67 |
| 20 | fold_04 | drawdown_scaled | 2.95% | 3.69% | 18.78% | 3.33 |

## Frozen validation choices

| Fold | Learned-only | With rules |
|---|---|---|
| fold_01 | drawdown_scaled | cash |
| fold_02 | drawdown_scaled | equal50 |
| fold_03 | lognet_raw | cash |
| fold_04 | lognet_scaled | lognet_scaled |

## Paired scaling effects

| Cost | Fold | Objective | Total-return difference | Annualized arithmetic excess 95% block interval |
|---:|---|---|---:|---:|
| 5 | fold_01 | lognet | 0.66% | [-0.00%, 0.34%] |
| 5 | fold_01 | drawdown | 2.76% | [-0.17%, 1.71%] |
| 5 | fold_02 | lognet | 0.49% | [-0.19%, 0.28%] |
| 5 | fold_02 | drawdown | 0.02% | [-0.55%, 0.82%] |
| 5 | fold_03 | lognet | 0.86% | [-0.00%, 0.66%] |
| 5 | fold_03 | drawdown | 0.86% | [-0.00%, 0.66%] |
| 5 | fold_04 | lognet | -3.51% | [-5.67%, 2.01%] |
| 5 | fold_04 | drawdown | -1.05% | [-1.39%, 0.43%] |
| 10 | fold_01 | lognet | 0.66% | [0.00%, 0.35%] |
| 10 | fold_01 | drawdown | 2.76% | [-0.16%, 1.72%] |
| 10 | fold_02 | lognet | 0.48% | [-0.18%, 0.28%] |
| 10 | fold_02 | drawdown | 0.04% | [-0.54%, 0.82%] |
| 10 | fold_03 | lognet | 0.86% | [-0.00%, 0.67%] |
| 10 | fold_03 | drawdown | 0.86% | [-0.00%, 0.67%] |
| 10 | fold_04 | lognet | -3.75% | [-5.79%, 1.87%] |
| 10 | fold_04 | drawdown | -1.12% | [-1.44%, 0.39%] |
| 20 | fold_01 | lognet | 0.65% | [-0.00%, 0.36%] |
| 20 | fold_01 | drawdown | 2.74% | [-0.16%, 1.75%] |
| 20 | fold_02 | lognet | 0.48% | [-0.17%, 0.28%] |
| 20 | fold_02 | drawdown | 0.06% | [-0.51%, 0.82%] |
| 20 | fold_03 | lognet | 0.85% | [-0.00%, 0.71%] |
| 20 | fold_03 | drawdown | 0.85% | [-0.00%, 0.71%] |
| 20 | fold_04 | lognet | -4.22% | [-6.04%, 1.56%] |
| 20 | fold_04 | drawdown | -1.26% | [-1.53%, 0.33%] |

## Mechanistic validation diagnostics

Table uses common fixed-equal50 phase-zero validation states; mature means at least 20 completed daily returns. Own-policy diagnostics are additionally retained in JSON.
| Fold | Candidate | Mature tanh saturation mean | Probability SD maximum mean | Entropy mean |
|---|---|---:|---:|---:|
| fold_01 | lognet_raw | 76.68% | 0.0004 | 0.5144 |
| fold_01 | lognet_scaled | 5.67% | 0.0197 | 0.5259 |
| fold_01 | drawdown_raw | 68.00% | 0.0016 | 0.5676 |
| fold_01 | drawdown_scaled | 5.50% | 0.0268 | 0.6067 |
| fold_02 | lognet_raw | 71.69% | 0.0032 | 0.6734 |
| fold_02 | lognet_scaled | 7.35% | 0.0145 | 0.6458 |
| fold_02 | drawdown_raw | 64.04% | 0.0046 | 1.0098 |
| fold_02 | drawdown_scaled | 4.05% | 0.0405 | 1.0711 |
| fold_03 | lognet_raw | 72.19% | 0.0006 | 0.6518 |
| fold_03 | lognet_scaled | 6.30% | 0.0062 | 0.3390 |
| fold_03 | drawdown_raw | 68.26% | 0.0011 | 0.5445 |
| fold_03 | drawdown_scaled | 4.99% | 0.0066 | 0.4401 |
| fold_04 | lognet_raw | 72.09% | 0.0032 | 0.8946 |
| fold_04 | lognet_scaled | 3.48% | 0.1353 | 1.0667 |
| fold_04 | drawdown_raw | 71.72% | 0.0056 | 0.4539 |
| fold_04 | drawdown_scaled | 5.96% | 0.0373 | 0.6445 |

More switching or lower saturation alone is not financial improvement. Every candidate, seed and fold is retained.
Intervals use 10-day moving blocks and 2000 resamples for annualized arithmetic excess, not CAGR; no multiplicity correction.
No test-selected winner, live deployment, new assets, paid resources or external publication.
