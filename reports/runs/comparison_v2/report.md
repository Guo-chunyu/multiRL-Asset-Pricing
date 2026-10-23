> 2026-10-23 correction: the market-performance interpretation of these Tencent hfq price-ratio results is withdrawn because additive adjustments distort returns. Original numerical records are retained for audit. See docs/data-audit-v3.md and the v3 raw-price/corporate-action study.

# 实验结果

数据：market；资产：159915, 510050, 510300, 510500；每个基础模型及选择器请求预算：10000 steps。

基础算法：td3, ddpg, sac, ppo, a2c；TD3（若启用）风险参数：(2.0,)；另训练DQN选择器。所有测试模型均冻结。各模型实际步数与训练秒数见 seed 目录 models.json / selector.json，上策略算法仅在完整 rollout 边界结束。协议：50/20/15/15 chronological holdout。各阶段从现金开始，边界跨期收益舍弃。

## 时间划分

| 阶段 | 首次决策 | 最后估值 | 收益期数 |
|---|---|---|---:|
| train | 2022-02-08 | 2024-06-11 | 567 |
| meta | 2024-06-12 | 2025-05-20 | 226 |
| validation | 2025-05-21 | 2026-01-27 | 170 |
| test | 2026-01-28 | 2026-10-16 | 170 |

## 测试收益（扣费）

| 策略 | 多种子平均累计收益 | 种子间标准差 |
|---|---:|---:|
| cash | 0.00% | 0.00% |
| equal_weight | -12.35% | 0.00% |
| buy_hold | -12.63% | 0.00% |
| momentum | -12.34% | 0.00% |
| reversal | -13.71% | 0.00% |
| td3_risk_2 | -14.02% | 4.08% |
| ddpg | -7.07% | 8.24% |
| sac | -5.37% | 5.21% |
| ppo | -6.76% | 4.08% |
| a2c | -4.31% | 7.46% |
| equal_mixture | -6.85% | 0.58% |
| dqn_selector | -3.92% | 5.62% |

## 验证集选出的策略

- seed 7：momentum，测试累计收益 -12.34%，最大回撤 19.97%。
  图：[净值与回撤](seed_7/test_equity.png)。
  仅在基础RL模型中按同一验证规则选择：a2c；固定后测试收益 -12.92%。
- seed 17：momentum，测试累计收益 -12.34%，最大回撤 19.97%。
  图：[净值与回撤](seed_17/test_equity.png)。
  仅在基础RL模型中按同一验证规则选择：ddpg；固定后测试收益 -16.37%。
- seed 29：momentum，测试累计收益 -12.34%，最大回撤 19.97%。
  图：[净值与回撤](seed_29/test_equity.png)。
  仅在基础RL模型中按同一验证规则选择：sac；固定后测试收益 -11.01%。

## 解释范围

本次是有限训练预算下的研究实验；能跑通、单段收益较高均不能证明样本外优势。不同种子共用市场历史，不是独立市场样本；测试表仅比较，不用于再挑参数。固定存续ETF与当前复权快照存在选择/时点限制。连续仓位没有整数手、最低佣金、涨跌停、停牌与订单成交限制，不能作为实盘结果。

详见项目 docs/research、docs/design.md 与 docs/limitations.md。manifest.json 保存数据哈希、软件版本、配置和划分；seed目录保存模型、验证选择、每日权重与净收益。