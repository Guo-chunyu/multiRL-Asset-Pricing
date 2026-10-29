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
