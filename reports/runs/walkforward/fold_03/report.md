> 2026-10-23 correction: the market-performance interpretation of these Tencent hfq price-ratio results is withdrawn because additive adjustments distort returns. Original numerical records are retained for audit. See docs/data-audit-v3.md and the v3 raw-price/corporate-action study.

# 实验结果

数据：market；资产：159915, 510050, 510300, 510500；每个基础模型及选择器预算：1500 steps。

每个种子含3个TD3风险偏好模型、DDPG及DQN；所有测试模型均冻结。协议：custom chronological splits。各阶段从现金开始，边界跨期收益舍弃。

## 时间划分

| 阶段 | 首次决策 | 最后估值 | 收益期数 |
|---|---|---|---:|
| train | 2022-02-08 | 2025-02-17 | 732 |
| meta | 2025-02-18 | 2025-10-31 | 172 |
| validation | 2025-11-03 | 2026-04-23 | 114 |
| test | 2026-04-24 | 2026-10-16 | 115 |

## 测试收益（扣费）

| 策略 | 多种子平均累计收益 | 种子间标准差 |
|---|---:|---:|
| cash | 0.00% | 不适用 |
| equal_weight | -13.51% | 不适用 |
| buy_hold | -13.49% | 不适用 |
| momentum | -12.37% | 不适用 |
| reversal | -15.42% | 不适用 |
| td3_risk_0 | -16.01% | 不适用 |
| td3_risk_2 | -14.42% | 不适用 |
| td3_risk_8 | -6.03% | 不适用 |
| ddpg | -4.08% | 不适用 |
| equal_mixture | -12.21% | 不适用 |
| dqn_selector | -16.90% | 不适用 |

## 验证集选出的策略

- seed 7：td3_risk_0，测试累计收益 -16.01%，最大回撤 23.92%。
  图：[净值与回撤](seed_7/test_equity.png)。

## 解释范围

本次是有限训练预算下的研究实验；能跑通、单段收益较高均不能证明样本外优势。不同种子共用市场历史，不是独立市场样本；测试表仅比较，不用于再挑参数。固定存续ETF与当前复权快照存在选择/时点限制。连续仓位没有整数手、最低佣金、涨跌停、停牌与订单成交限制，不能作为实盘结果。

详见项目 docs/research、docs/design.md 与 docs/limitations.md。manifest.json 保存数据哈希、软件版本、配置和划分；seed目录保存模型、验证选择、每日权重与净收益。