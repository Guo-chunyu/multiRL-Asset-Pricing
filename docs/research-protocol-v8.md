# v8: single history-count unit ablation

Prespecified before any v8 real-data fit. This is retrospective exploratory development on historical periods repeatedly inspected in v3–v7, not pristine out-of-sample discovery, a performance guarantee, or live results. No new assets, paid data/compute, publication or live trading. Preserve all prior frozen sources, runs, and checksums.

## Hypothesis and falsification

Read-only v7 validation probes found low market-state sensitivity and substantial first-layer tanh saturation, with a large contribution from the unscaled shadow-history count (0–20). The intervention is **only division of that observation coordinate by 20**. It changes units, not information. All other feature values, shadow/account ledgers, rewards, masks, PPO parameters, targets and fragment schedules remain unchanged. No fitted scaler, clipping, entropy regularization, macrosteps, reward changes or expanded expert set.

Mechanism is not equivalent to alpha. If saturation falls but action probabilities remain state-insensitive, count scaling is not a sufficient explanation. If state sensitivity/switching increases without stable after-cost risk-adjusted gains, no financial advantage is demonstrated. Increasing switches alone is not a success criterion. No numerical return/drawdown goal has yet been supplied by the user.

## Candidates, budget and reused controls

Fixed tie/registry order: lognet_raw, lognet_scaled, drawdown_raw, drawdown_scaled. Both v6 objectives remain: 100 × daily log net return; and 100 × (daily log net return minus 0.5 × increment in running maximum drawdown). Both use the v7 causal phase mask, with index 0 an ignored dummy on non-rebalance days.

Raw controls are **all 24 v7 phase checkpoints**, selected by mechanism, not by historical test rank: two objectives × seeds 7/17/29 × all four folds. They are copied byte-for-byte, not refitted. New scaled fits use the same two objectives/seeds/folds: **24 × exactly 20,000 daily transitions = 480,000 new retained-fit transitions**. The 480,000 historical control transitions are reused, not counted as new compute. No checkpoint selection, budget search or early stopping. Final checkpoint only.

Official sb3-contrib MaskablePPO 2.7.1 on CPU. Exact effective v7 config: MlpPolicy [64,64], learning_rate .0001, n_steps 100, batch_size 50, n_epochs 10, gamma .99, gae_lambda .95, clip_range .2, clip_range_vf None, normalize_advantage True, ent_coef **0.0**, vf_coef .5, max_grad_norm .5, target_kl None. No hidden FinRL defaults. Random contiguous 126-daily-transition fragments. Actual 159-fragment schedules must match independent seeded NumPy schedules and paired raw controls.

## Data and execution

Same certified restored 510050/510300/510500 raw-price snapshots, cash dividend entitlement/receivables/payment and splits, 2015-04-15–2026-10-22. Exact historical hashes and disclosed fresh cross-source verification limitations remain inherited. Raw next-open execution, rebalance every five sessions, all-cash initial NAV 1, four independent fully funded shadow experts, no hidden warmup. Fixed experts: cash, equal50 (50% TOTAL equity), vol10, trend10. Six comparators add single-account equal-target mix and causal trailing20 shadow winner.

Four unchanged calendar folds: training through 2017/2019/2021/2023; validation 2018/2020/2022/2024; test 2019–2020/2021–2022/2023–2024/2025–2026-10-22. Overlap and repeated historical inspection prohibit treating folds as independent unseen discoveries.

## Reuse checks, identity and gates

Before training, validate original v7 manifest/source/data identities; all reused record seals, model/validation hashes, original effective PPO config, versions, budgets and schedules. Check model ZIP runtime package metadata against current Python, SB3, PyTorch, NumPy, cloudpickle and Gymnasium; require sb3-contrib 2.7.1. OS/kernel is not a numeric-package parity requirement. Replay every raw control on validation only through count_scale=1; verify all metrics and complete daily net-return vectors against originals. New source/tests/protocol, inherited identities, runtime and every reused artifact hash enter the immutable v8 manifest.

Stages: prepare → import-controls → baselines → independent pretraining review → fit (three seed shards) → select → independent pretest review → evaluate costs → report. No real fitting until tests/review pass. Original tests already exist but no v8 test replay opens until every candidate checkpoint, validation ledger and record is complete and frozen. Preparation/import/baselines use no historical test performance. Resume fails closed on changed sources, data, protocol, runtime, controls, model/validation bytes or schedule. Do not silently change a prepared protocol; disclose fixes and reprepare before fitting/testing as appropriate.

## Diagnostics fixed before fitting

For every raw and scaled model, record policy probabilities, mean categorical entropy, per-action probability standard deviation, distinct argmax actions, and first policy tanh saturation fraction (absolute activation > .95). Use current pre-step phase-zero states only; ignored singleton masks would otherwise distort the statistic. Record all decisions and mature decisions separately (at least 20 completed daily shadow returns).

Primary mechanistic comparison uses **common fixed-equal50 validation state trajectories**, transforming only count units for each model; this avoids confounding with account states generated by different policies. Also retain each model's own-policy validation trajectory diagnostics. No counterfactual count-zero deployment or recalibration of old checkpoints. Diagnostics are secondary to financial metrics and never used to choose a test winner.

## Selection and financial reporting

Per fold choose among all four learned candidates using three-seed mean validation total return minus 0.5 maximum drawdown. Separately choose among those four means and six scalar fixed comparators. Rules have one path, not three artificial seed copies. Exact ties follow the registry then comparator order. Also freeze best validation-fixed expert. Freeze all 48 candidate checkpoints/records/validation hashes and the baseline summary before any v8 test replay.

Primary cost 5 one-way bps; saved models and frozen selections replay at 10/20 bps without retraining/reselection. Report every candidate/seed/fold and all six rules: total return, CAGR, maximum drawdown, volatility/Sharpe, exposure, turnover, execution switches and costs. Selected learned-only and selected-with-rules paths remain distinct.

Primary controlled contrast: scaled minus raw, separately for each objective, paired by seed/fold/calendar/cost. Also compare to equal-target mix, trailing20 and validation-best fixed expert. Ten-day moving-block bootstrap, 2000 resamples, 95% interval on annualized arithmetic excess (not CAGR nor total-return difference), no multiplicity correction. Retain all effects; no historical test winner promotion. Independently reconstruct every actual-account validation/test ledger from raw opens/corporate actions using the existing separate accounting auditor. Audit agreement tests accounting, not efficacy.

Official implementation guidance: https://stable-baselines3.readthedocs.io/en/v2.2.1/guide/rl_tips.html recommends normalizing custom inputs. This motivates an engineering hypothesis, not a prediction of financial success; installed pinned 2.7.1 implementation governs execution.
