# v9: prespecified validation-only relative-allocation signal gate

Registered before financial evaluation. No RL training, no new test-split replay, no asset expansion, paid resource, live action or external publication. Preserve all frozen v5–v8 files and runs. These are previously inspected historical development periods, not untouched out-of-sample evidence.

## One hypothesis and exact rule

Existing selector targets all lie on the cash/equal-risky-basket line. Test only whether a distinct relative-allocation rule justifies a later, separately reviewed RL experiment. The sole new rule is **relative_momentum50**: 50% cash; rank the original three ETFs (510050, 510300, 510500) by causal 20-trading-session total return through decision close. Assign risky proportions 1/2, 1/3, 1/6 to descending rank, giving actual account target equity weights 1/4, 1/6, 1/12. Exact score ties use ascending ticker code. Compare only **equal50** (50% cash, each ETF 1/6). No lookback, weight, tie, rule or gate search.

Use corporate-action-aware signal close for ranking, never adjusted execution prices. Read only the 21-close prefix ending at the decision date. The first decision follows the inherited annual range's first close; execution is at the next raw open. Five-session rebalancing, daily marked accounting, original dividend entitlements/receivables/payment and splits, all-cash NAV 1 reset. Between rebalances quantities are preserved. Same total equity exposure does not imply same beta or volatility. No beta/alpha estimate is made; volatility and actual exposure are reported.

## Data scope and frozen gates

Use only the four existing validation periods **2018, 2020, 2022, 2024** as independent annual paths, each from the first to last available close in that year. History strictly before a decision can warm price-derived factors. These years overlap some previously reported tests, so the restriction is to this experiment's registered validation split, not a claim that the years were never seen. Do not evaluate 2019–2020/2021–2022/2023–2024/2025–2026 test paths for this study.

Same restored certified three-ETF data and disclosed fresh cross-source verification limitations. Freeze raw input hashes, all relevant source/tests/protocol/auditor hashes and validation calendars in a manifest. The `evaluate` stage requires a separate passed `preflight_review.json` matching the manifest hash. No financial evaluation before root review. Any code/protocol/data change invalidates resume and requires review, never silently update a prepared experiment.

## Fixed compute-budget hurdle

Evaluate both rules at each of **5, 10, 20 one-way bps**: 24 annual ledgers in total, no random seeds or fake replicates. At **every** cost, relative_momentum50 must have strictly greater `total_return - 0.5 * max_drawdown` than equal50 in at least **3 of 4 years**. At **every year and every cost**, its maximum drawdown may be no more than **2 percentage points worse**. All conditions must pass. Equality of scores is not a win; a 2 pp drawdown difference passes (numerical tolerance 1e-12).

This arbitrary but prespecified hurdle only budgets future computation. It is not a statistical test, proof of alpha, or deployment permission. If any condition fails, **do not train a new selector under this plan**, and do not alter lookbacks/weights/gates in response. Passing only permits proposing a separately reviewed next study.

## Required checks and report

Test exact ranking/ties/causality, weights/caps/exposure, inherited next-open and corporate-action accounting, validation-only calendars and review gate, complete grid and failure handling. Reconstruct each saved account independently using the existing raw-price auditor. Report every year/cost/rule: net total and annual return, maximum drawdown, annual volatility, actual exposure, turnover/costs, requested-target L1 difference and distinct target count. Do not concatenate independent annual NAVs, choose a winning cost/year or describe a seed average as an investable portfolio. No bootstrap threshold is added.

Stages: tests → prepare → independent preflight review → evaluate → independent ledger/result review. Commands use `python -m rlasset.signal_gate_v9 --stage prepare` and `--stage evaluate`, default output `runs/research_v9_signal`. No RL model is produced.
