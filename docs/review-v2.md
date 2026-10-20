# 五算法扩展的因果性与重载审查

审查日期：2026-10-20。范围为 TD3、DDPG、SAC、PPO、A2C、冻结基础策略上的 DQN 选择器，以及验证集选择与模型重载流程。新增审计没有修改生产实现或主实验模型；原有 `docs/review.md` 保留。

## 未来价格扰动测试

可重复运行：

```powershell
.venv/Scripts/python.exe -m pytest tests/test_causality_v2.py -q --junitxml=runs/causality_v2_pytest.xml
```

实测结果：**2 passed in 21.59s**，机器可读记录为 `runs/causality_v2_pytest.xml`。

每项测试使用固定 140 日、2 资产的合成数据，数据与主种子均为 3，lookback 为 10，风险系数为 2。分别以每模型 24 步和 48 步预算，独立运行两份完整实验。时间索引为 train 10–74、meta 75–100、validation 101–119、test 120–139。

第二份数据只改 test 区间：一只资产的 OHLC 乘以 `exp(0.15 + 0.035 × test_day)`，另一只乘以其倒数。每个日期的四个价格使用相同因子，维持合法 OHLC 次序；测试前的所有行逐元素保持不变。不同日期使用不同因子，因此扰动会改变实际收益，不只是价格单位。

每对实验均确认：

- 五个基础模型与 DQN 的全部已保存 `state_dict` 张量精确相同，使用 `torch.equal`，没有浮点容差；同时比较优化器状态和额外 PyTorch 变量，包括 SAC 熵系数相关变量。
- 完整验证集指标字典与 `selection.json` 精确相同，涵盖总体选中策略、最佳基础 RL、验证指标和 DQN 策略池顺序。
- 每个验证集逐日账本精确相同，包括 DQN 选择与 `validation_best_rl_on_validation.csv`。
- 测试段等权策略的累计收益确实变化，证明价格冲击有效。

两项测试共独立训练 24 个模型文件，并进行 12 对模型状态比较。实际执行真实 FinRL/SB3 学习，没有替换模型或学习过程。

为确认该测试能够识别泄漏，在独立临时 Python 进程中，故意把环境观测乘以 `panel.close[-1] / panel.close[0]`，使训练观测使用最终日期价格。24 步测试如预期失败，首个差异为 `a2c.zip/policy/log_std`；该失败也伴随验证集选择改变。此变异仅存在于该进程的内存中，未写入生产文件。正常实现随后通过上述两项测试。

## 主实验 seed 7 的精确重载

审查对象：`runs/comparison_v2/seed_7`，五个基础模型与 DQN 均记录 10,000 个实际训练步骤。数据快照 SHA-256 为：

```text
64c84ef0588d2a8086e7dd56987d1ce75d6f5fe3d9b155e0cf21711012d9145c
```

执行 `evaluate_saved("runs/comparison_v2", seed=7)`，并与首次保存的 `test_metrics.json` 比较。**12 个策略 × 每策略 8 项指标，全部 96 项数值精确相同**，使用完整字典相等判断，没有数值容差。结果另存为该种子目录内的 `reloaded_test_metrics.json`。

重载前后 6 个模型文件的 SHA-256 与修改时间均未变化，`selection.json` 的字节内容也未变化。评估代码另对基础模型和 DQN 的 `num_timesteps` 做前后不变检查。

| 模型文件 | 重载前后相同的 SHA-256 |
|---|---|
| `td3_risk_2.zip` | `de83afd6ca4b4b9cf3d2662286608594c17979eaaf16847837a3b831d34e7f3a` |
| `ddpg.zip` | `35883b94d91f9f96f9244dac7e1da0dd3ef38f41da430e7c168ccdeae9980ef7` |
| `sac.zip` | `6ca194c7d402d3bf4eed9faec12ae2d68e3870c8963145fd4e04e403b5e1caba` |
| `ppo.zip` | `50c5154a8bc0fdf049ff9d5e2c8aaefa2dfa1e106c36cc8a2608ed5bd2c49445` |
| `a2c.zip` | `b2f811e54f46aa03f7b2a8066d920ab168c3b481c5b665fe74b8a91b38051d16` |
| `dqn_selector.zip` | `0cfb885d0d8ebbde852eebd1a6308d0af55826135e36dd66fffe9707034bd514` |

## 结论与适用边界

本次证据支持：在已测试的两组固定预算与数据扰动下，测试期未来价格没有影响基础模型学习、DQN 学习或验证选择；已完成主实验 seed 7 的保存模型可以精确重现首次评估。

这是一组可重复的隔离与复现测试，不能形式化证明所有输入、所有分段及所有未来代码修改均不存在泄漏。它也没有解决复权数据的时点可得性、固定存续 ETF 样本选择、人工反复查看测试结果后的研究选择偏差，或真实成交约束。小预算合成测试只用于软件审查，不构成策略收益或训练充分性的证据。主实验此次重载审计限于 seed 7，其他种子与滚动区间的结果由各自实验记录报告。
