# v9 results: relative-allocation signal gate

**The prespecified budget gate failed. No new RL selector was trained.**

A single 20-session momentum ranking changed relative ETF weights while fixing total equity at 50%. The comparison used equal50 at identical next-open execution, five-session rebalance, raw prices/corporate actions, and three transaction-cost levels. No lookback, weight, cost or gate was selected after observing the outcomes.

The fixed score was total net return minus half maximum drawdown, not return alone. At 5 and 10 bps, the rule won in only 2/4 registered validation years; at 20 bps, 0/4. The no-more-than-2-percentage-point drawdown-deterioration constraint passed, but the required 3/4 score wins at every cost did not. This gate is an arbitrary declared compute-budget hurdle, not a statistical test or proof that all relative-allocation strategies fail.

All years were previously inspected development data; 2020/2022/2024 overlap historical test periods in older studies. This stage uses only its fixed validation paths and does not create new untouched out-of-sample evidence.

Engineering verification: 498 full-suite tests passed. Independent review re-ran all 24 raw accounting audits, recomputed five financial metrics for all 24 rows, and verified 1176 scheduled causal targets. Maximum raw-account discrepancy was approximately 1.51e-14. All v8 frozen input identities stayed unchanged.

One initial execution failed before saving its first ledger because an output directory was missing. The failed manifest/review/log were retained; only directory creation plus a synthetic integration regression was added. A new manifest and independent preflight were then prepared. Rules, data, calendars and gate were unchanged; no failed-run financial output was used to redesign the experiment.

The full run is runs/research_v9_signal; a compact mirror is reports/runs/research_v9_signal. No live trades, paid resources, new assets, model training or GitHub push occurred. The requested high-return/low-drawdown outcome has not been achieved.

# v9 relative-allocation signal gate

Validation-only historical development. No RL was trained and no test split was opened.
One 20-session rule, fixed 50% equity, no parameter search. Same total equity does not imply same volatility or beta.
Beta is deliberately not estimated; this study makes no risk-adjusted alpha claim. Each year starts independently; do not concatenate NAVs.

Budget gate: **do_not_train_new_selector**

| Cost bps | Year | Rule | Net total return | Annual return | MDD | Annual volatility | Mean turnover | Target L1 vs equal50 |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | 2018 | equal50 | -13.640% | -14.162% | 16.105% | 10.760% | 0.00354 | 0.00000 |
| 5 | 2018 | relative_momentum50 | -14.872% | -15.437% | 17.308% | 10.877% | 0.02614 | 0.16667 |
| 5 | 2020 | equal50 | 11.768% | 12.283% | 7.635% | 11.420% | 0.00353 | 0.00000 |
| 5 | 2020 | relative_momentum50 | 11.925% | 12.447% | 7.435% | 11.416% | 0.01941 | 0.16667 |
| 5 | 2022 | equal50 | -9.217% | -9.617% | 13.022% | 9.859% | 0.00338 | 0.00000 |
| 5 | 2022 | relative_momentum50 | -10.041% | -10.475% | 14.093% | 9.821% | 0.02077 | 0.16667 |
| 5 | 2024 | equal50 | 9.656% | 10.119% | 7.083% | 11.476% | 0.00366 | 0.00000 |
| 5 | 2024 | relative_momentum50 | 10.345% | 10.842% | 7.315% | 11.532% | 0.02734 | 0.16667 |
| 10 | 2018 | equal50 | -13.677% | -14.200% | 16.119% | 10.760% | 0.00354 | 0.00000 |
| 10 | 2018 | relative_momentum50 | -15.141% | -15.715% | 17.521% | 10.877% | 0.02614 | 0.16667 |
| 10 | 2020 | equal50 | 11.721% | 12.233% | 7.637% | 11.421% | 0.00353 | 0.00000 |
| 10 | 2020 | relative_momentum50 | 11.662% | 12.173% | 7.444% | 11.421% | 0.01941 | 0.16667 |
| 10 | 2022 | equal50 | -9.254% | -9.656% | 13.056% | 9.860% | 0.00338 | 0.00000 |
| 10 | 2022 | relative_momentum50 | -10.266% | -10.709% | 14.285% | 9.822% | 0.02077 | 0.16667 |
| 10 | 2024 | equal50 | 9.608% | 10.068% | 7.087% | 11.475% | 0.00366 | 0.00000 |
| 10 | 2024 | relative_momentum50 | 9.982% | 10.461% | 7.417% | 11.530% | 0.02734 | 0.16667 |
| 20 | 2018 | equal50 | -13.751% | -14.276% | 16.147% | 10.761% | 0.00354 | 0.00000 |
| 20 | 2018 | relative_momentum50 | -15.676% | -16.268% | 17.945% | 10.879% | 0.02613 | 0.16667 |
| 20 | 2020 | equal50 | 11.625% | 12.134% | 7.641% | 11.424% | 0.00353 | 0.00000 |
| 20 | 2020 | relative_momentum50 | 11.139% | 11.625% | 7.461% | 11.433% | 0.01940 | 0.16667 |
| 20 | 2022 | equal50 | -9.328% | -9.733% | 13.122% | 9.862% | 0.00338 | 0.00000 |
| 20 | 2022 | relative_momentum50 | -10.714% | -11.175% | 14.667% | 9.825% | 0.02077 | 0.16667 |
| 20 | 2024 | equal50 | 9.511% | 9.966% | 7.095% | 11.474% | 0.00366 | 0.00000 |
| 20 | 2024 | relative_momentum50 | 9.260% | 9.702% | 7.620% | 11.529% | 0.02734 | 0.16667 |

Gate requires at least 3/4 strictly positive total-return-minus-half-MDD differences at EACH cost, and no annual MDD deterioration above 2 percentage points at ANY cost.
This arbitrary prespecified research-budget hurdle is not a statistical test. A failure forbids new selector training under this plan; do not alter the rule after seeing results.
