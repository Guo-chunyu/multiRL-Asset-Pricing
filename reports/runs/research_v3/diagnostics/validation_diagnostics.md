# Validation allocation diagnostics

Status: **complete**. Completed 48/48; pending 0; invalid 0.

Validation only. Columns are descriptive; rows follow the registered order rather than performance rank.

| Fold | Seed | Candidate | Return | Mean equity | Max target std | Daily gross turnover | 5 bps annual fee proxy | Flags |
|---|---:|---|---:|---:|---:|---:|---:|---|
| fold_01 | 7 | legacy_ppo | -13.95% | 54.56% | 2.40% | 0.91% | 0.11% | - |
| fold_01 | 7 | robust_ppo | -23.58% | 90.70% | 2.15% | 0.78% | 0.10% | - |
| fold_01 | 7 | robust_sac | -23.39% | 89.75% | 0.94% | 0.74% | 0.09% | - |
| fold_01 | 7 | robust_td3 | -26.73% | 98.19% | 4.77% | 1.27% | 0.16% | - |
| fold_01 | 17 | legacy_ppo | -10.19% | 50.63% | 0.00% | 0.48% | 0.06% | near_constant_executed_targets |
| fold_01 | 17 | robust_ppo | -25.51% | 99.99% | 0.23% | 0.69% | 0.09% | near_constant_executed_targets |
| fold_01 | 17 | robust_sac | -23.60% | 91.33% | 1.40% | 0.97% | 0.12% | - |
| fold_01 | 17 | robust_td3 | -20.53% | 80.50% | 9.28% | 2.61% | 0.33% | - |
| fold_01 | 29 | legacy_ppo | -21.48% | 72.03% | 1.92% | 1.05% | 0.13% | - |
| fold_01 | 29 | robust_ppo | -24.51% | 99.98% | 0.82% | 0.67% | 0.08% | - |
| fold_01 | 29 | robust_sac | -23.43% | 89.34% | 2.21% | 1.12% | 0.14% | - |
| fold_01 | 29 | robust_td3 | -18.89% | 79.92% | 0.03% | 0.50% | 0.06% | near_constant_executed_targets |
| fold_02 | 7 | legacy_ppo | 10.05% | 50.37% | 0.01% | 0.47% | 0.06% | near_constant_executed_targets |
| fold_02 | 7 | robust_ppo | 21.59% | 93.17% | 2.78% | 0.94% | 0.12% | - |
| fold_02 | 7 | robust_sac | 20.79% | 89.42% | 3.11% | 1.16% | 0.15% | - |
| fold_02 | 7 | robust_td3 | 22.84% | 99.99% | 0.00% | 0.56% | 0.07% | near_constant_executed_targets |
| fold_02 | 17 | legacy_ppo | 10.15% | 50.77% | 0.07% | 0.48% | 0.06% | near_constant_executed_targets |
| fold_02 | 17 | robust_ppo | 19.32% | 85.41% | 2.21% | 0.73% | 0.09% | - |
| fold_02 | 17 | robust_sac | 21.34% | 89.92% | 1.90% | 1.17% | 0.15% | - |
| fold_02 | 17 | robust_td3 | 19.69% | 80.97% | 13.31% | 2.43% | 0.31% | - |
| fold_02 | 29 | legacy_ppo | 11.08% | 55.14% | 1.34% | 0.69% | 0.09% | - |
| fold_02 | 29 | robust_ppo | 22.15% | 93.51% | 1.66% | 0.83% | 0.11% | - |
| fold_02 | 29 | robust_sac | 22.21% | 91.40% | 6.05% | 1.12% | 0.14% | - |
| fold_02 | 29 | robust_td3 | 20.77% | 97.96% | 12.49% | 2.68% | 0.34% | - |
| fold_03 | 7 | legacy_ppo | -0.49% | 2.70% | 0.08% | 0.05% | 0.01% | near_constant_executed_targets, cash_heavy |
| fold_03 | 7 | robust_ppo | -17.31% | 95.77% | 1.03% | 0.71% | 0.09% | - |
| fold_03 | 7 | robust_sac | -17.28% | 90.97% | 2.74% | 0.96% | 0.12% | - |
| fold_03 | 7 | robust_td3 | -15.15% | 79.96% | 9.02% | 3.04% | 0.38% | - |
| fold_03 | 17 | legacy_ppo | -12.48% | 69.26% | 0.35% | 0.64% | 0.08% | near_constant_executed_targets |
| fold_03 | 17 | robust_ppo | -16.74% | 91.16% | 1.74% | 0.89% | 0.11% | - |
| fold_03 | 17 | robust_sac | -16.64% | 90.11% | 1.13% | 0.74% | 0.09% | - |
| fold_03 | 17 | robust_td3 | -17.83% | 90.45% | 8.56% | 3.27% | 0.41% | - |
| fold_03 | 29 | legacy_ppo | -9.22% | 50.01% | 0.00% | 0.46% | 0.06% | near_constant_executed_targets |
| fold_03 | 29 | robust_ppo | -15.97% | 86.69% | 2.77% | 1.01% | 0.13% | - |
| fold_03 | 29 | robust_sac | -16.67% | 90.86% | 1.96% | 0.92% | 0.12% | - |
| fold_03 | 29 | robust_td3 | -14.89% | 89.22% | 13.66% | 2.56% | 0.32% | - |
| fold_04 | 7 | legacy_ppo | 10.45% | 50.22% | 0.07% | 0.43% | 0.05% | near_constant_executed_targets |
| fold_04 | 7 | robust_ppo | 12.48% | 80.04% | 1.28% | 0.71% | 0.09% | - |
| fold_04 | 7 | robust_sac | 15.18% | 91.96% | 2.44% | 0.98% | 0.12% | - |
| fold_04 | 7 | robust_td3 | 16.29% | 85.93% | 9.00% | 2.58% | 0.33% | - |
| fold_04 | 17 | legacy_ppo | 10.57% | 50.76% | 0.02% | 0.43% | 0.05% | near_constant_executed_targets |
| fold_04 | 17 | robust_ppo | 14.57% | 96.87% | 2.40% | 0.76% | 0.10% | - |
| fold_04 | 17 | robust_sac | 14.69% | 89.79% | 2.63% | 0.99% | 0.12% | - |
| fold_04 | 17 | robust_td3 | 14.77% | 79.96% | 7.23% | 2.00% | 0.25% | - |
| fold_04 | 29 | legacy_ppo | 5.05% | 50.03% | 0.00% | 0.54% | 0.07% | near_constant_executed_targets |
| fold_04 | 29 | robust_ppo | 16.39% | 88.33% | 0.80% | 0.53% | 0.07% | - |
| fold_04 | 29 | robust_sac | 14.94% | 89.15% | 3.08% | 1.15% | 0.15% | - |
| fold_04 | 29 | robust_td3 | 15.34% | 99.97% | 6.22% | 1.95% | 0.25% | - |

- No rankings, new strategy selection, model changes, or retraining are performed.
- Allocation variability uses submitted targets on scheduled rebalance days only; passive drift is separate.
- Near-constant means maximum component target standard deviation <0.005 (0.5 percentage points).
- Cash-heavy means mean cash weight >=0.95; the cash slot includes unpaid dividend receivables.
- High gross turnover means mean daily two-sided risky turnover >0.20; this is a descriptive threshold.
- Annual fee proxies are 252 * observed mean turnover * fee rate, holding the observed path fixed.
- Fee proxies are first-order diagnostics, not retraded cost-stress backtests or exact cumulative losses.
- Low drawdown or high return may reflect exposure differences; these diagnostics do not establish alpha.
- Partial snapshots must not be treated as comparisons over all registered seeds or folds.
