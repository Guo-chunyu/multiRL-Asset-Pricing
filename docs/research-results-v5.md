# v5 controlled residual-risk results

2026-10-26. **The learned residuals did not establish a dependable improvement over their own simple rule anchors.** Do not promote them on this evidence. This is a completed finite experiment, not a new untouched out-of-sample test or profitability guarantee.

## Completed design and checks

- Three registered families × three seeds × four historical folds = **36 models**, all exactly 20,000 steps (720,000 total), real pinned FinRL PPO; no failed/dropped seeds.
- Validation-only rule preview ran before fitting. All checkpoints and validation selections froze before any test ledger opened. No parameter changes after seeing returns.
- 5/10/20bps one-way costs replayed the same checkpoints. Source/data/protocol identity and frozen selections rechecked successfully after evaluation.
- **390 tests passed**; final runner-specific 7 tests rerun passed. Independent raw-open accounting reconstruction passed on 36 learned validation, 8 selection-rule validation, 20 preview-rule and 168 test ledgers.
- Restored data exactly matches original v4 hashes (line endings restored), 2,800 sessions, three ETFs, 25 cash events, one split. Historical archived cross-provider audit applies to identical content; **fresh** Eastmoney verification failed HTTP503 and was not fabricated. Restoration audit records this distinction and current primary-source checks.

## Learned return increment versus its own anchor

Percentage points of cumulative test return, averaged over all three seeds, at 5bps. Each fold independently starts at NAV 1; rows must not be stitched into a fictitious continuous curve.

| Test interval | Volatility residual minus volatility rule | Trend residual minus trend rule |
|---|---:|---:|
| 2019–2020 | -0.753 | -2.197 |
| 2021–2022 | +0.572 | +0.218 |
| 2023–2024 | -0.519 | -0.125 |
| 2025–2026-10-22 | -0.127 | -0.225 |

Neither residual beats its own anchor in three of four folds. The sign pattern persists at 10/20bps. All 36 paired annualized-arithmetic-excess bootstrap intervals (three learned families × four folds × three costs) include zero. These are exploratory 10-day moving-block percentile intervals, 2,000 draws, conditional on realized models; they are not corrected multiple-comparison tests or probabilities of future success.

## Risk reduction is mostly the rule, not learned alpha

At 5bps, trend residual mean risky exposure ranges 23.24%–44.00%; volatility residual 52.01%–79.08%. Trend residual mean daily turnover ranges 0.0168–0.0213, compared with volatility residual 0.0091–0.0106. Lower exposure reduces drawdown but is not a free improvement:

- 2019–2020: trend residual return 12.36%, max drawdown 10.89%; volatility residual 34.89%, drawdown 11.33%.
- 2021–2022: trend residual -8.34%, drawdown 11.98%; volatility residual -14.92%, drawdown 22.30%.
- 2023–2024: trend residual -7.25%, drawdown 12.66%; volatility residual -4.75%, drawdown 23.11%.
- 2025–2026-10-22: trend residual 10.76%, drawdown 6.73%; volatility residual 12.53%, drawdown 10.17%.

Original v4-risk retraining also shows unstable anchor increments: +5.56,+0.52,-0.20,-2.10pp across folds, with all paired intervals including zero. Frozen validation choices alternate between trend residual and v4risk; allowing rule fallback chooses the trend rule in fold03. No historical winner is being reselected on tests.

## Artifacts and interpretation

Full local experiment: `runs/research_v5/` contains hashed manifest, protocol/source hashes, checkpoint zips, signed validation records, frozen selection, all seed/fold raw ledgers, independent `.audit.json` sidecars, logs, `REPORT.md`, `all_metrics.csv`, `anchor_contrasts.json`, `status.json`, and `artifact_inventory.json` SHA256 inventory. Small reproducible report artifacts are also copied to `reports/runs/research_v5/`.

Pragmatic conclusion: retain transparent audited rules as baselines and the engineering capability, but this small residual PPO formulation has not justified its added complexity. Do not tune this gate after reviewing the same four historical folds and call the result a fresh validation. A future hypothesis needs its own prospective protocol and genuinely untouched confirmation data; no additional search was run here.
