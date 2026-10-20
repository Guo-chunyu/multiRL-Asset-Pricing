# FinRL 与 RLPortfolio 源码研究及接入决策

研究日期：2026-10-19。研究范围是首版 A 股场内 ETF 的深度强化学习组合优化；不包含 LLM。下面区分上游已有能力、从源码推导的问题、以及本项目的工程决策。

## 1. 固定研究版本与许可证

| 项目 | 本地位置 | 固定提交 | 上游提交日期 | 许可证 |
|---|---|---|---|---|
| [FinRL](https://github.com/AI4Finance-Foundation/FinRL) | `vendor/FinRL` | `8511e7c15fbe1fbe5eb25e9f89932ac1ecd546a0` | 2026-10-14，+08:00 | MIT，AI4Finance Foundation Inc. |
| [RLPortfolio](https://github.com/CaioSBC/RLPortfolio) | `vendor/RLPortfolio` | `071cf75da9ddf95ec62210627b971086eec6f18f` | 2025-03-12，-03:00 | MIT，Caio de Souza Barbosa Costa |

两者均为官方仓库的浅克隆，未修改上游源码。复制上游代码时应保留版权和 MIT 许可文本。FinRL README 另有商标说明；本项目不把自己的实现命名为官方 FinRL 产品。当前 FinRL README 将自身定位为经典教育与研究框架，另指向 FinRL-X。这里遵从用户指定，使用原 FinRL 仓库的实际接口，不替换成另一个项目。

运行时校验的 FinRL 关键文件 SHA-256（统一 CRLF 为 LF 后计算）：

| 文件 | SHA-256 |
|---|---|
| `finrl/agents/stablebaselines3/models.py` | `e71707d0eaae0117fad36f9aa1be59da0186619db59111fe43372d92ad9510b0` |
| `finrl/config.py` | `48675567fd7b797d228b2d5729e7b00f824894a2303c8bb51ec62b9389c9b70e` |

提交校验固定仓库版本，两个内容校验固定实际算法入口和默认配置。它们不是对整个 checkout 的签名。

## 2. FinRL：选择真实可用的框架边界

### 2.1 分层与实际算法接口

FinRL 的主要分层是 `meta` 数据与市场环境、`agents` 学习算法、`applications` 任务应用。首版采用这个边界：自有 A 股数据和成交环境通过 Gymnasium 连接 FinRL 的 SB3 `DRLAgent`，FinRL 再构造 SB3 的 TD3/DDPG 对象。它不是把直接调用 SB3 改名为“FinRL 风格”。

关键入口是 [stablebaselines3/models.py](https://github.com/AI4Finance-Foundation/FinRL/blob/8511e7c15fbe1fbe5eb25e9f89932ac1ecd546a0/finrl/agents/stablebaselines3/models.py)：

```python
agent = DRLAgent(env)
model = agent.get_model(
    "td3", policy="MultiInputPolicy", seed=0,
    model_kwargs={"learning_rate": 3e-4, "action_noise": "normal"},
    policy_kwargs={"net_arch": [64, 64]},
)
model.learn(total_timesteps=2000)
```

`get_model` 支持 `a2c/ddpg/td3/sac/ppo/tqc/crossq`。`action_noise="normal"` 会由 FinRL 转换成每一动作维度标准差 0.1 的 `NormalActionNoise`。注意该方法原地改写传入的 `model_kwargs`，所以适配器每次创建独立字典，避免复用被改写的配置。`train_model(model, tb_log_name, total_timesteps, callbacks)` 是 `learn` 的包装；`callbacks` 要传列表。

上游 `DRL_prediction` 假设环境存在 `get_sb_env`、`df`、`save_asset_memory`、`save_action_memory`，并采用特定终止处理。它不是任意 Gymnasium 环境的通用评估器。首版独立执行确定性评估循环，以统一记录每期净值、费用、目标仓位和漂移仓位，并验证时间边界。

### 2.2 为什么不是直接复制旧 portfolio_allocation 环境

FinRL 中有两个不同的组合环境，不能混为一谈。

| 环境 | 观察与动作 | 收益与成本 | 接入判断 |
|---|---|---|---|
| [`env_portfolio_allocation/env_portfolio.py`](https://github.com/AI4Finance-Foundation/FinRL/blob/8511e7c15fbe1fbe5eb25e9f89932ac1ecd546a0/finrl/meta/env_portfolio_allocation/env_portfolio.py) | 协方差矩阵与技术指标；softmax 资产权重 | `step` 中按 close 比率更新资产；虽然构造器存储 `transaction_cost_pct`，该更新分支没有扣费；奖励为新资产总值 | 不用于可信净收益基准 |
| [`env_portfolio_optimization/env_portfolio_optimization.py`](https://github.com/AI4Finance-Foundation/FinRL/blob/8511e7c15fbe1fbe5eb25e9f89932ac1ecd546a0/finrl/meta/env_portfolio_optimization/env_portfolio_optimization.py) | `(features, assets, window)` 历史张量；现金加资产权重；可附上次动作 | log 净收益、权重漂移、TRF/WVM 成本模型 | 值得借鉴会计与状态设计；旧式 `reset(self)` 和 API 切换参数仍需适配现代 Gymnasium |

第二个环境与 RLPortfolio 的 POE 同一研究方向。FinRL 同时包含 `agents/portfolio_optimization/{algorithms,architectures,utils}.py`，其中 `PolicyGradient`、EIIE 与 PVM 是另一条直接策略优化路径，不等同于 `agents/stablebaselines3/models.py` 的 DDPG/TD3。

首版采用自有环境的理由是：要明确上一日收盘可见特征、下一开盘成交、开盘到开盘收益、费用扣除顺序、现金、漂移和交易限制。仅替换价格列不自动解决时间对齐。

### 2.3 数据流水线中必须主动处理的问题

[DataProcessor 源码](https://github.com/AI4Finance-Foundation/FinRL/blob/8511e7c15fbe1fbe5eb25e9f89932ac1ecd546a0/finrl/meta/data_processor.py) 在这个提交中实际支持 `alpaca/wrds/yahoofinance/fxmacrodata`。README 的数据源表列出 Akshare、Baostock 等，不代表这个 `DataProcessor` 构造器可以直接接受这些名称。首版用专门的 A 股 ETF 下载与 CSV 接口，转换为 FinRL 常见的 `date/tic/OHLCV` 长表。

[`preprocessors.py`](https://github.com/AI4Finance-Foundation/FinRL/blob/8511e7c15fbe1fbe5eb25e9f89932ac1ecd546a0/finrl/meta/preprocessor/preprocessors.py) 的源码发现：

- `data_split(df,start,end)` 使用 `[start,end)` 半开区间，按 `date/tic` 排序，日期 factorize 为索引。边界语义值得复用。
- `FeatureEngineer.preprocess_data` 最后执行 `df.ffill().bfill()`。直接对完整长表使用这种填充可能跨资产传播数值，向后填充也可能使用后续才出现的数据；严格因果管线应按资产计算特征，不后向填充，不用测试数据拟合归一化器。
- `clean_data` 根据完整区间 pivot 后的 `dropna(axis=1)` 保留无缺失 ticker。推论：若根据整个训练与测试区间决定资产池，就会利用未来是否存在数据的信息；应预先定义资产池及缺失处理方案。
- `GroupByScaler` 提供按资产拟合的思想，但必须先切分再 `fit`，不能在全区间拟合。

这些是研究性数据纪律，不是对所有 FinRL 示例结果的统一否定。

### 2.4 最小依赖导入与本机兼容性

`finrl/__init__.py` 导入 `test/trade/train`；`trade` 又导入 Alpaca paper trading，`train` 又导入多数据源处理器。因此，普通 `from finrl.agents...` 会先要求与本任务无关的券商和数据源依赖。整个仓库的 `requirements.txt` 还包括 Ray、ElegantRL、TA-Lib、文档工具等。首版没有全量安装这些栈。

`src/rlasset/finrl_adapter.py` 做了一个明确、可审计的兼容边界：

1. 验证本地 FinRL checkout 提交和上述两个关键文件哈希。
2. 将真实 `vendor/FinRL/finrl` 路径注册为 Python `finrl` 包的搜索路径，只跳过根 `__init__`。
3. 正常导入并执行原始 `finrl.agents.stablebaselines3.models`、`config`、`env_stocktrading`、`preprocessors` 等模块。
4. 默认真实调用 `DRLAgent(env).get_model(...)`。没有算法 stub、源码改写或静默 fallback。
5. 缺失依赖时明确失败。只有显式 `backend="sb3"` 才直接调用 SB3，并在模型元数据中标记。

这不是对全部 FinRL API 的兼容承诺。已导入另一份 FinRL 时会拒绝混用；应使用新 Python 进程。`models.py` 现在导入 CrossQ/TQC，即便仅训练 TD3 也需要 `sb3-contrib`。`data_split` 的导入链还需要 `stockstats/yfinance`。

本机验证组合：Windows x64、Python 3.13.1、torch 2.11.0 CPU、numpy 2.2.6、pandas 2.3.3、Gymnasium 1.2.3、Stable-Baselines3 2.7.1、sb3-contrib 2.7.1、stockstats 0.6.9、yfinance 1.7.0。只验证上述适配路径；不声称 FinRL 的全部数据源/券商模块或全部 RLPortfolio 算法兼容此环境。

### 2.5 首次真实数据接入记录

东方财富接口在本机多次出现连接中断。下载器提供显式 `provider="eastmoney"` 与 `provider="tencent"`，不自动切换，不混合不同来源；缓存按 provider、ticker、请求区间和 `hfq` 分开保存。成功下载逐只验证并持久化，后续失败可以续传；失败重试有次数上限和退避。Windows 临时目录与项目缓存可能分别在 C: 和 W:，缓存通过目标盘暂存后替换，避免跨盘 rename 失败。

第一次可用实数数据使用 `AKShare / Tencent stock_zh_a_hist_tx`。完整 2018 起窗口中，159915 缺 2021-02-08，严格面板检查按预期拒绝；核查材料记载该 ETF 当日停牌、次日 10:30 复牌，因此也不能把 2 月 9 日视为正常开盘成交。研究期明确改为 2022-01-01 起，不填补缺价、不删除全市场某天。[信达证券 2021-02-09 交易提示](https://www.cindasc.com/osoa/views/a/20210209/42396.html)。

正式下载器最终生成 `data/processed/ashare_etf.csv` 和相邻元数据：159915、510050、510300、510500，各 1157 个完整交易日，2022-01-04 至 2026-10-16，共 4628 行。CSV SHA-256 为 `64c84ef0588d2a8086e7dd56987d1ce75d6f5fe3d9b155e0cf21711012d9145c`。元数据保留真实 provider、endpoint、每只 ETF 的下载时间、区间、行数及哈希。

另有 `data/raw_alternative/tencent_adjust_field_audit.json` 保存腾讯原响应字段核验；4 只 ETF、2018—2026 各年度响应均检查到 `hfqday`。这能证明接口返回字段口径，不能单独证明供应商复权算法等价于真实持仓现金分红再投资。本项目称它为后复权价格快照，不把它直接称为可成交总收益指数。

## 3. RLPortfolio：吸收什么，保留什么边界

### 3.1 环境记账、费用和漂移

[PortfolioOptimizationEnv](https://github.com/CaioSBC/RLPortfolio/blob/071cf75da9ddf95ec62210627b971086eec6f18f/rlportfolio/environment/portfolio_optimization_env.py) 是现代 Gymnasium 环境，`reset(seed,options)` 返回二元组，`step` 返回五元组。默认状态 `(F,N,L)`，动作长度 `N+1` 且第一维现金。动作已非负且和近似 1 时直接使用，否则 softmax。

记上一期价格变化后的实际权重为 `w_minus`，目标权重为 `w`，当期价格比率为 `y`，现金比率为 1。核心记账顺序为：

```text
V_after_fee = mu * V_before
holdings_end = V_after_fee * w * y
V_end = sum(holdings_end)
w_end = holdings_end / V_end
reward = log(V_end / V_before)
```

下一次手续费基准取 `_final_weights[-1]`，即实际漂移后的 `w_end`，而不是上期目标权重。这个区别会改变换手、费用和策略训练，是首版必须继承的原则。

上游提供三种费用模型：

- `trf`：迭代求 transaction remainder factor。对统一费率 `c`，更新式是 `mu = [1-c*w_minus[0]-(2c-c²)*sum(max(w_minus[i]-mu*w[i],0))]/[1-c*w[0]]`（求和只包括风险资产），直到变化小于 `1e-10`。
- `trf_approx`：`mu = 1-c*sum(abs(w[i]-w_minus[i]))`，风险资产双向交易金额近似；不要再额外乘一个不匹配的 `1/2`。
- `wvm`：按权重变动计算交易费并从现金扣除；若现金不足，可能保持原组合。它与 TRF 的经济假设不同，不能混用结果。

首版可借鉴漂移和自融资思想，并在自己的费用模型下分别定义佣金、滑点和换手口径。上游 TRF 没有表达 A 股整手、不同资产交易规则、买卖非对称费率、最小佣金、涨跌停不可成交等全部细节，不能据此宣称完整实盘撮合。

环境的 `return_last_action` 返回 `_actions_memory[-1]`，属于上次目标动作；费用计算却用实际漂移权重。首版的 `weights` 观察选择实际持仓权重以反映当前账户约束。它与上游 `last_action` 字段名称相似但语义有意不同。

### 3.2 EIIE、PVM 与回放

[EIIE 源码](https://github.com/CaioSBC/RLPortfolio/blob/071cf75da9ddf95ec62210627b971086eec6f18f/rlportfolio/policy/eiie.py) 用 `(1,k)` 与 `(1,L-k+1)` 卷积在每个资产的时间轴上共享参数，再拼接上期风险资产权重，用 `(1,1)` 卷积输出分数，加入现金分数后 softmax。共享参数能减少固定资产池上的自由参数，是首版低样本日频训练适合研究的归纳偏置。

本项目 `SharedTemporalExtractor` 使用对每个资产共享的 Conv1d 时间编码，输出按 ticker 排列的表示，再拼接实际持仓。编码器具有资产置换等变性，但下游 SB3 全连接 actor/critic 没有完整保留该性质。它不是论文 EIIE policy 的完整复现，也不把平均池化、现金分支、参数量说成与上游完全一致。

[PortfolioVectorMemory](https://github.com/CaioSBC/RLPortfolio/blob/071cf75da9ddf95ec62210627b971086eec6f18f/rlportfolio/algorithm/buffers/portfolio_vector_memory.py) 按时间保存组合向量、初始化全现金、随训练更新相应索引。它解决策略依赖上期分配而不能只看到价格的问题。首版通过观察中保留漂移仓位吸收该状态思想，不声称实现了原版 PVM 训练算法。

[Replay buffers](https://github.com/CaioSBC/RLPortfolio/blob/071cf75da9ddf95ec62210627b971086eec6f18f/rlportfolio/algorithm/buffers/replay_buffers.py) 包含连续样本段和偏向近期的几何抽样。此设计可作为非平稳市场的后续消融方向；首版 TD3 使用 SB3 自己的 transition replay buffer。PVM、序列回放与 TD3 replay 不能简单混为同一机制。

### 3.3 直接策略优化不等于 TD3

[`PolicyGradient._gradient_ascent`](https://github.com/CaioSBC/RLPortfolio/blob/071cf75da9ddf95ec62210627b971086eec6f18f/rlportfolio/algorithm/policy_gradient.py) 直接最小化批量负 log 组合增长，并不训练 TD3/DDPG 的 Bellman critic。源码中的手续费近似 `trf_mu` 在 `torch.no_grad()` 中计算，因此该路径并没有对交易费因子本身求完整梯度。它适合单独列为对照算法，而不应包装成“TD3 的精确可微手续费优化”。

同一文件 `train(...val_gradient_steps=1)` 与 `test(...gradient_steps=1)` 默认允许验证/测试期间在线学习，并可能使用训练回放池。若研究冻结模型的样本外性能，需要显式设置相应 `gradient_steps=0`，并记录是否使用在线适应。在线学习本身可以是有效实验，但与冻结测试是不同协议。

RLPortfolio 的完整安装声明包括 `quantstats/torch_geometric/jupyter/tensorboard` 等。首版读取并借鉴源码，不依赖整包运行，不安装 GPM 图网络相关栈，也不宣称已验证全部 RLPortfolio 实现。

## 4. 主算法与可检验的改进路径

选择 TD3 为连续权重控制的首版主线，DDPG 为同环境下的消融基准。TD3 的双 critic 最小目标、延迟 actor 更新和目标策略平滑，针对 DDPG 的函数近似误差与过估计问题；这是算法层面的选择依据，不是对 ETF 收益优势的保证。[原始 TD3 论文](https://arxiv.org/abs/1802.09477)、[SB3 官方 TD3 文档](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/td3.html)。

优先比较以下变化，每次只改一个关键因素：

| 对照 | 用于回答的问题 |
|---|---|
| DDPG vs TD3，保持环境/时间切分/训练预算一致 | 双 critic 等稳定化是否改善样本外风险收益及跨种子离散度 |
| 平铺价格 MLP vs 共享时间编码 | RLPortfolio 启发的参数共享是否有益 |
| 正常费用与成本压力情景 | 收益是否依赖高换手或低估费用 |
| 不同训练随机种子与多个时间窗口 | 结果是否来自偶然初始化或单一市场阶段 |
| 冻结模型 vs 明确标记的在线适应（后续） | 性能差异是否由测试期间再训练造成 |

MetaTrader 的多策略/多专家思路可在独立、冻结测试的基础模型之外逐步接入。首版若只做验证集选模型或简单集成，应准确说明它与原论文完整元策略学习的差别。

本项目不从历史回测直接推断可交易 alpha，也不把合成数据或几十步训练的工程 smoke test 当作投资效果。真实有效性需要预先定义资产池、完整成本口径、无前视时间对齐、多个随机种子及锁定测试集。

## 5. 当前适配器验证证据

运行命令：

```powershell
.venv/Scripts/python.exe -m pytest tests/test_finrl_adapter.py -q
```

开发时首先观察到 5 个失败测试（适配器与编码器模块尚不存在），实现后结果为 **5 passed**。测试包括：真实 FinRL TD3/DDPG 各完成 24 步学习、actor 参数更新、模型保存/加载预测一致、共享时间编码的资产置换等变性、持仓状态保留和梯度传播、显式 SB3 后端及无效参数拒绝。

这些证据证明框架接口和神经网络通路可执行，不证明任何市场收益结论。完整项目环境记账、数据时间切分和实数数据回测应由各自测试与实验报告验证。

数据接入另通过 `tests/test_download.py` 的 9 个测试：字段规范化、瞬时失败重试、部分完成后续传、不同 provider 的缓存/元数据隔离、跨文件系统缓存持久化，以及拒绝来源/复权/资产/请求区间身份错配。网络边界使用受控 worker fixture，生产代码没有伪造市场数据的后备路径；真实腾讯下载已另外执行并通过面板验证。最后合并运行两组测试，结果为 **14 passed in 14.34s**。
