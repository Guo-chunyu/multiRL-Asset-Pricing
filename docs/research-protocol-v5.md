# v5: fixed controlled residual-risk experiment

Registered 2026-10-26 before any v5 performance was computed. Historical folds have already been explored in v3/v4; **this is retrospective development, not untouched out-of-sample evidence**. Restored inputs match the original v4 snapshot exactly after LF-to-CRLF canonicalization. Fresh Eastmoney cross-provider verification is unavailable (HTTP503); certification transparently uses historical snapshot identity and complete input checks, not a claim of fresh independent verification. Comparisons emphasize all policies rerun on this common dataset. No promising returns are guaranteed.

## Question and fixed family

Does a small learned residual improve on a transparent volatility/trend rule after costs, or is apparent improvement entirely the rule/exposure? Three candidates: `v4risk_ppo` (exact v4 risk environment, base observations); `residual_vol_ppo` (bounded residual without trend gate); `residual_trend_ppo` (same with fixed gate). The latter two have identical observation shape; original market features, current weights, weekly phase, explicit anchor weights and effective gate. The comparison with v4risk is a package comparison (action bounds and anchor observation), not isolated action-bounds attribution.

Anchor is equal risky mix with 10% annual volatility budget. As in v4, use 20/60-day trailing TR log-return sample covariance, each shrunk 25% toward diagonal. Risk estimate is maximum of their annualized portfolio volatilities and 1% floor. Risky mix capped at 50% per asset. The trend gate is the mean across all three ETFs and both windows of strict `TR_close[t] > mean(TR_close[t-h+1:t+1])`, h=60/120. Gate-off is g=1. Exposure is **g * min(1,b/sigma)**. This ordering is fixed.

Residual action is N+1 bounded [-1,1]: `b=.10*(1+.25*a0)` (7.5%–12.5%), risky preference `exp(.10*ai)`, then fixed risky-mix cap. Zero action exactly reproduces its anchor. Gate cannot be overridden. No new reward, gate threshold/window tuning, search expansion, or seed dropping after viewing baseline or learned historical performance.

## Data and accounting gate

Use only certified `data/v3/ashare_broad_raw.csv`, `cash_actions.csv`, `splits.csv`: 510050/510300/510500, original common calendar 2015-04-15 through 2026-10-22. No hfq execution, synthetic replacement, or empty action substitution. Preparation requires `restoration_verification.json` explicitly `certified_ready:true`, matching SHA256 values in `input_sha256`, and either completed independent verification or exact historical certified snapshot identity modulo line endings with archived prior cross-provider audit matching all three hashes, current primary-source corporate-action spot checks, and the fresh-provider limitation explicitly disclosed. This evidence route was agreed before any v5 performance was computed. Missing both evidence paths blocks training. Manifest hashes inputs, certification, all execution dependencies, independent accounting auditor, and this protocol. Changed code/data/protocol invalidates resume; use a new explicitly documented run for engineering corrections.

Keep inherited raw-price corporate-action accounting: decision at close, execution following open, ex-date dividend receivables paid only on actual pay date, splits adjust shares. Features use causal total-return index only. Long-only, no leverage, fractional shares, zero interest, tax-before cash distributions; weekly rebalance. Inherited reward/risk aversion and random 126-session training episodes unchanged. Independent auditor reconstructs raw-open cash flows and NAV rather than trusting environment results.

## Finite budget, freeze and baselines

Main budget **36 models**: 3 candidates x 3 seeds (7/17/29) x all 4 existing calendar folds, each PPO 20000 environment steps, real pinned FinRL DRLAgent/SB3, MlpPolicy 64x64, learning rate 1e-4. No convergence claim. Training ends at 2017/2019/2021/2023, validation 2018/2020/2022/2024, historical tests 2019–2020/2021–2022/2023–2024/2025–2026-10-22. All folds independently start NAV=1. Final checkpoint only; no best-test checkpoint selection.

First run fixed no-training rules on validation periods only: cash, equal 50%, equal 100%, zero-action volatility anchor, zero-action trend anchor. This preview is descriptive, cannot change candidates, and is hash-recorded. No test ledger is opened before all models and selections are frozen; historical test baseline metrics are computed only afterwards. Execute the registered learned family regardless of appealing/disappointing rule returns unless an engineering/data error blocks it.

Freeze **all** learned checkpoints, three seeds' validation ledgers and records for all folds before learned test evaluation. Per-fold RL choice maximizes mean-seed `(total_return - .5*max_drawdown)`; separately select among all three learned candidates plus both anchors with that same criterion. Rules count once, not as independent random seeds. Test performance never changes choice. Fixed tie order is candidate registry then anchor registry. Fit shards may resume only if signatures/checksums/calendar/metrics reconcile.

## Evaluation and decision discipline

Primary cost 5bps one-way; 10/20bps stress tests replay exact checkpoints without retraining. Report every seed/candidate/fold, return range, drawdown, annual volatility, Sharpe (undefined stays null), turnover and exposure. Compare each residual with its own zero-action anchor and v4risk with volatility anchor. A return reduction with drawdown reduction is a tradeoff, not automatic improvement. Report paired seed-mean daily-return increments via 10-day moving-block bootstrap, 2000 resamples, separately by fold; intervals describe annualized arithmetic mean excess, not CAGR or future outperformance probability. Seeds are optimization randomness, not additional market history. No multiple comparison correction or independent confirmation claim.

Commands from repository root with the verified runtime and PYTHONPATH=src:

```
python -m rlasset.study_v5 --stage status
python -m rlasset.study_v5 --stage prepare
python -m rlasset.study_v5 --stage baselines
python -m rlasset.study_v5 --stage fit --seed 7  # repeat 17,29; optional --fold shard
python -m rlasset.study_v5 --stage select
python -m rlasset.study_v5 --stage evaluate --cost-bps 5  # repeat10,20
python -m rlasset.study_v5 --stage report
```

Default output `runs/research_v5`, never overwrite v4. Keep logs, model zip files, signed records, frozen selections, raw ledgers, independent audit sidecars, metric CSV and anchor contrasts. Synthetic tests and tiny learning smoke tests verify mechanics only; they are never market efficacy results. A data blocker is an explicit incomplete study, not a negative or positive strategy result.
