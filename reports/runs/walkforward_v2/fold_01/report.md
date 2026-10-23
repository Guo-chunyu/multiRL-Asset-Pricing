> 2026-10-23 correction: the market-performance interpretation of these Tencent hfq price-ratio results is withdrawn because additive adjustments distort returns. Original numerical records are retained for audit. See docs/data-audit-v3.md and the v3 raw-price/corporate-action study.

# 实验结果

数据：market；资产：159915, 510050, 510300, 510500；每个基础模型及选择器请求预算：5000 steps。

基础算法：td3, ddpg, sac, ppo, a2c；TD3（若启用）风险参数：(2.0,)；另训练DQN选择器。所有测试模型均冻结。各模型实际步数与训练秒数见 seed 目录 models.json / selector.json，上策略算法仅在完整 rollout 边界结束。协议：custom chronological splits。各阶段从现金开始，边界跨期收益舍弃。

## 时间划分

| 阶段 | 首次决策 | 最后估值 | 收益期数 |
|---|---|---|---:|
| train | 2022-02-08 | 2024-02-29 | 500 |
| meta | 2024-03-01 | 2024-11-15 | 172 |
| validation | 2024-11-18 | 2025-05-12 | 114 |
| test | 2025-05-13 | 2025-10-30 | 115 |

## 测试收益（扣费）

| 策略 | 多种子平均累计收益 | 种子间标准差 |
|---|---:|---:|
| cash | 0.00% | 0.00% |
| equal_weight | 29.07% | 0.00% |
| buy_hold | 29.96% | 0.00% |
| momentum | 32.57% | 0.00% |
| reversal | 26.48% | 0.00% |
| td3_risk_2 | 6.89% | 7.08% |
| ddpg | 17.67% | 14.55% |
| sac | 19.54% | 1.45% |
| ppo | 14.14% | 17.79% |
| a2c | 4.77% | 8.16% |
| equal_mixture | 13.91% | 0.44% |
| dqn_selector | 4.81% | 2.06% |

## 验证集选出的策略

- seed 7：dqn_selector，测试累计收益 2.50%，最大回撤 0.54%。
  图：[净值与回撤](seed_7/test_equity.png)。
  仅在基础RL模型中按同一验证规则选择：td3_risk_2；固定后测试收益 0.27%。
- seed 17：dqn_selector，测试累计收益 6.48%，最大回撤 3.02%。
  图：[净值与回撤](seed_17/test_equity.png)。
  仅在基础RL模型中按同一验证规则选择：a2c；固定后测试收益 0.09%。
- seed 29：dqn_selector，测试累计收益 5.43%，最大回撤 3.59%。
  图：[净值与回撤](seed_29/test_equity.png)。
  仅在基础RL模型中按同一验证规则选择：a2c；固定后测试收益 0.03%。

## 解释范围

本次是有限训练预算下的研究实验；能跑通、单段收益较高均不能证明样本外优势。不同种子共用市场历史，不是独立市场样本；测试表仅比较，不用于再挑参数。固定存续ETF与当前复权快照存在选择/时点限制。连续仓位没有整数手、最低佣金、涨跌停、停牌与订单成交限制，不能作为实盘结果。

详见项目 docs/research、docs/design.md 与 docs/limitations.md。manifest.json 保存数据哈希、软件版本、配置和划分；seed目录保存模型、验证选择、每日权重与净收益。