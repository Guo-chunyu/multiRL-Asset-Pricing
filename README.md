# multiRL Asset Pricing

A 股 ETF 深度强化学习组合优化：五算法比较、MetaTrader 式策略选择与模拟成交研究。

**2026-10-23 数据口径更正：旧 v1/v2 的腾讯 hfq 为加法复权，价格比不能代表可交易总收益。旧复权回测的业绩解释已撤回，原记录保留供复核。新 v3 使用未复权价格、独立分红应收/支付和份额折算会计。** 见[数据纠错说明](docs/data-audit-v3.md)与[v3实验协议](docs/research-protocol-v3.md)。

最新结果：[v3回测与迭代结论](docs/results-v3.md)、[完整实验表与图](reports/runs/research_v3/report.md)、[指标CSV](reports/runs/research_v3/metrics.csv)。48个模型、960,000步训练、4个测试时期、3种成本假设均已执行；287项测试通过。**验证预选策略尚未稳定优于简单等权组合，目标未达成。**

v3 复现（先完成下方安装）：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_market_data_v3.py
.\.venv\Scripts\python.exe -m rlasset.study_v3 --stage prepare
.\.venv\Scripts\python.exe -m rlasset.study_v3 --stage fit --seed 7
.\.venv\Scripts\python.exe -m rlasset.study_v3 --stage fit --seed 17
.\.venv\Scripts\python.exe -m rlasset.study_v3 --stage fit --seed 29
.\.venv\Scripts\python.exe -m rlasset.study_v3 --stage select
.\.venv\Scripts\python.exe -m rlasset.study_v3 --stage evaluate --cost-bps 5
.\.venv\Scripts\python.exe -m rlasset.study_v3 --stage evaluate --cost-bps 10
.\.venv\Scripts\python.exe -m rlasset.study_v3 --stage evaluate --cost-bps 20
.\.venv\Scripts\python.exe -m rlasset.study_report_v3 --run runs/research_v3
```

下面保留 v2 架构与复现记录；其 hfq 收益不能继续作为市场业绩依据。v3 的三只宽基、长历史和会计均不同，也不能与旧表直接计算提升幅度。

这是以 FinRL 为真实算法入口的深度强化学习组合优化研究项目。支持 TD3、DDPG、SAC、PPO、A2C、冻结策略池上的 DQN 选择器，以及独立的整数份额模拟成交。研究路线对应 **Deep Deterministic Portfolio Optimization → MetaTrader → FinRL → RLPortfolio → 可执行组合优化实验**。这里的“大模型”指深度强化学习，不包含大语言模型。

核心问题是：在明确的时间顺序、成本与持仓约束下，连续控制能否学习有效组合；多种风险偏好的策略与 DQN 选择器能否改善稳定性。项目包含论文研究、会计环境、严格时间切分、基线、模型保存与重载、净值和风险指标。成功运行与实证优势需要分别判断；短预算训练用于检查研究流程。

历史研究入口：[五算法研究与成交模拟结果（复权绩效已撤回）](docs/results-v2.md)、[旧实验协议](docs/research-protocol-v2.md)、[旧自动统计报告](reports/runs/research_v2/report.md)、[执行引擎说明](docs/execution.md)。原[首轮结果](docs/results.md)和模型仍保留。已查看的行情不能恢复为未触碰的测试数据，研究仍属于回顾性、探索性。

## 本轮完整实验

当前数据覆盖 2022-01-04 至 2026-10-16，共 1,157 个交易日、4 只股票 ETF。主实验统一风险系数 2，每算法 10,000 步、3 个种子；滚动实验 3 个窗口×3 个种子，每算法 5,000 步。不同算法计算量不同，实际步数和训练用时均保留。

```powershell
# 首次下载项目后，先按下方“Windows 安装”安装，再获取行情
.\.venv\Scripts\python.exe -m rlasset.cli download --provider tencent --start 20220101 --end 20261016 --output data/processed/ashare_etf.csv

# 新实验会重新训练，结果写入本机 runs/；已发布的历史结果位于 reports/
powershell -ExecutionPolicy Bypass -File .\scripts\run_research.ps1 -Suffix my_study
# 仅在本机已有相同配置的 runs/ 实验时追加 -Resume

# 单独进行五算法研究
.\.venv\Scripts\python.exe -m rlasset.cli train --data data/processed/ashare_etf.csv --output runs/my_comparison --steps 10000 --seeds 7 17 29 --algorithms td3 ddpg sac ppo a2c --risk-levels 2

# 下载未复权价格并模拟订单，不连接券商
.\.venv\Scripts\python.exe -m rlasset.execution_data
.\.venv\Scripts\python.exe -m rlasset.paper --run runs/comparison_my_study --output runs/my_paper --seeds 7 17 29
```

`selection.json` 保存验证期选出的整体策略、单一 RL 策略和 DQN 候选顺序；比较报告另给出按各算法三个种子平均验证得分选出的统一候选，不根据测试收益挑赢家。`orders.csv` 与 `ledger.json` 分别保存模拟订单回执和每日资金/份额对账。真实行情正常回放与人工加入停牌、薄成交量、涨跌停阻挡的压力情景分别输出。

模拟资金默认为 100 万元；佣金每侧 3 bps、每单最低 5 元、滑点 2 bps 是可配置实验假设，不代表券商报价。执行采用未复权价和整数份额，收益未计现金分红，不能与复权研究净值当成同口径总回报比较。全天成交量只作离线容量代理，尚无逐笔排队、真实订单路由或券商实盘证据。

## 研究与实现对应

| 来源 | 本项目采用的内容 | 明确的边界 |
|---|---|---|
| [Deep Deterministic Portfolio Optimization](https://arxiv.org/abs/2003.06497) | 单资产二次控制解析反馈、独立 Bellman 检验、交易费用与风险的权衡 | 解析基准本身不训练 RL；不会把单资产解析保证外推到多资产市场 |
| [MetaTrader](https://arxiv.org/abs/2210.01774) | 先训练/冻结基础策略，再训练 DQN 每天选择一个策略；比较简单混合与固定最佳策略 | 使用不同算法/风险偏好和规则策略构建候选池，不是原论文专家行为克隆目标的完整复现 |
| [FinRL](https://github.com/AI4Finance-Foundation/FinRL) | 实际调用 `DRLAgent.get_model` 构造五种算法，底层为 Stable-Baselines3 | 采用窄导入适配器；不安装或使用全部券商和数据源模块 |
| [RLPortfolio](https://github.com/CaioSBC/RLPortfolio) | 共享时间卷积、持仓状态、费用与权重漂移的会计思想 | 不声称复现完整 EIIE、PVM 或原版直接策略梯度训练器 |

详细阅读记录：`docs/research/ddpo.md`、`docs/research/metatrader.md`、`docs/research/frameworks.md`；设计：`docs/design.md`；局限：`docs/limitations.md`。

## Windows 安装

需要 Windows、Git、Python 3.11 或更新版本。本机开发验证使用 Python 3.13，CPU 训练。公开仓库包含源码、测试、文档和经过路径净化的研究结果；行情快照、模型 ZIP、环境及第三方完整仓库需在本机获取或重新生成。命令均在项目根目录执行。

```powershell
# 首次获取项目
git clone https://github.com/Guo-chunyu/multiRL-Asset-Pricing.git
cd multiRL-Asset-Pricing

# 在新项目副本中创建独立 .venv，安装依赖并校验两个上游仓库
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1

# 直接使用环境内 Python，不依赖 PowerShell 激活脚本
.\.venv\Scripts\python.exe -m pytest -q
```

脚本默认新建环境不继承系统包；已有环境和上游源码不会被重置。如果当前 `.venv` 已继承系统包，脚本会提示并保留。需要干净复现时：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -VenvPath .venv-clean
.\.venv-clean\Scripts\python.exe -m pytest -q
```

后续命令中的 `.venv` 按需替换为 `.venv-clean`。仅读取检查现有安装可使用 `-CheckOnly`。仓库版本不匹配或有本地改动时脚本中止并保留现场，不执行 `reset` 或 `clean`。

固定上游版本：

| 仓库 | 提交 |
|---|---|
| FinRL | `8511e7c15fbe1fbe5eb25e9f89932ac1ecd546a0` |
| RLPortfolio | `071cf75da9ddf95ec62210627b971086eec6f18f` |

FinRL 适配器还校验实际算法入口与配置文件的 SHA-256。RLPortfolio 保留源码供研究，不作为完整运行时依赖。Python 依赖见 `pyproject.toml`；主要 RL 接口版本已固定，其他依赖使用版本范围，因此不是全部依赖的永久锁文件。

## 从离线演示到市场数据

先运行无需行情服务的模拟数据演示：

```powershell
.\.venv\Scripts\python.exe -m rlasset.cli demo --output runs/demo --steps 512 --seeds 7
```

演示数据明确标记为 synthetic。512 步只是工程检查，不适合判断收益、调参优劣或收敛情况。`--steps` 是训练预算参数，不是交易日数量；具体模型和选择器预算会记录在配置中。

再下载 A 股 ETF 日频后复权快照（本次实跑使用腾讯源）：

```powershell
.\.venv\Scripts\python.exe -m rlasset.cli download --provider tencent --output data/processed/ashare_etf.csv --start 20220101 --end 20261016
```

默认研究池为 `510300、510500、159915、510050`。支持显式选择 AKShare / Tencent 或 Eastmoney，未指定 provider 时仍默认 Eastmoney，不自动混源。本次东方财富连接不稳定，使用 `--provider tencent`，实得 2022-01-04 至 2026-10-16 的 1,157 个共同交易日。原拟 2018 年开始的区间包含 159915 在 2021-02-08 的停牌缺口，因此首版明确调整起点，没有补价或静默删除日期。

截止日是显式研究参数；重跑时应明确自己的截止日。下载器保存同名 `.metadata.json` 和逐标的缓存，记录来源、复权方式、下载时刻、实际日期范围、版本和 CSV 哈希。行情服务失败会有限重试并明确报错，不用模拟价格替代。复权价格是供应商定义的历史快照，不能据此宣称精确分红总回报或历史时点数据，核验细节见 `docs/data-audit.md`。

也可提供自有 CSV，必须为完整、无重复的长表：

```csv
date,tic,open,high,low,close,volume
2024-01-02,510300,3.50,3.55,3.45,3.52,1000000
2024-01-02,510500,5.40,5.48,5.35,5.44,800000
```

示例仅说明格式。实际数据需要至少 `lookback + 80` 个日期、每个日期都有所有资产、正数 OHLCV、合理最高/最低价和一致复权口径。不自动补齐缺失价格，零成交量也会拒绝，避免把停牌数据当成可执行交易。

训练和独立重载测试：

```powershell
.\.venv\Scripts\python.exe -m rlasset.cli train --data data/processed/ashare_etf.csv --output runs/ashare --steps 5000 --seeds 7 17 29
.\.venv\Scripts\python.exe -m rlasset.cli evaluate --run runs/ashare --seed 7
```

5000 步和三个种子是可执行起点，不是推荐的最终研究预算。训练较短时策略可能接近随机、未充分学习，真实数据结果完全可能落后于等权与现金。验证集用于选择最终策略，测试集不参与学习或选择种子。

输出目录默认必须为空，防止覆盖实验。中断后可在原 `train` 或 `walk-forward` 命令追加 `--resume`：数据哈希、配置、分段与软件版本必须一致，仅复用已经完整保存的模型；尚未完成的模型重新训练。它不是训练到任意中间步的恢复。

扩展训练窗口的滚动验证与冻结模型的成本压力测试：

```powershell
.\.venv\Scripts\python.exe -m rlasset.cli walk-forward --data data/processed/ashare_etf.csv --output runs/walkforward --folds 3 --steps 1500 --seeds 7
.\.venv\Scripts\python.exe -m rlasset.cli evaluate --run runs/ashare --seed 7 --cost-bps 20
```

滚动验证将最后30%的日期分成三个互不重叠的测试块，每折重新训练，基础训练窗口扩大，之前15%用于选择器训练、10%用于验证。后续折可以使用届时已成为历史的早期测试数据。每折从现金重新开始并舍弃跨边界收益，汇总不伪装成连续持有的收益率。`walkforward_summary.json` 保存各折结果。成本压力测试不再训练模型，也不覆盖原测试结果；写入 `stress_20bps_metrics.json`。

运行独立理论基准：

```powershell
.\.venv\Scripts\python.exe -m rlasset.cli theory --output runs/theory.json
```

这个命令比较单资产解析反馈、myopic 与零持仓；没有训练 TD3。它用于验证控制模型与实现的数学一致性，不能把输出称为“RL 已恢复最优解”。

## 交易规则与实验协议

观察截至交易日 t 收盘，执行发生在 t+1 开盘。旧仓承担隔夜收益，开盘调仓扣费，新仓承担日内收益。初始全现金，现金收益设为 0；组合 long-only、无杠杆，单 ETF 目标权重上限为 0.5，被动漂移可能超过目标上限。

训练环境费用按风险资产实际买卖金额计提，现金不重复计费。自融资比例使用固定点求解，奖励为扣费后对数收益减过去协方差估计的风险罚项。最终绩效使用账户净收益；风险罚项不会混入实际净值。训练环境采用可分割份额与假定开盘成交；独立 paper 模块再检验整手、最低佣金、T+1、停牌、涨跌停阻挡及部分成交，尚未建模真实盘口。

数据按日期划分为基础策略训练 50%、选择器训练 20%、验证 15%、测试 15%。warmup 单独保留；区间边界的跨段收益舍弃，历史窗口可读取之前已经发生的数据。选择器训练时基础模型冻结，候选的影子组合独立推进，选择器只能看到已实现的过去表现。

对照包括现金、等权每日调仓、买入持有、动量、反转、五种 RL 算法、不同风险惩罚的 TD3、固定等权混合和 DQN 选择器。不同种子用于观察训练不确定性，它们共享市场样本，不能当成独立市场证据。

## 输出及复查

每个实验目录保留配置、数据来源/哈希和切分记录，并按种子保存模型与测试轨迹。核心文件包括：

```text
runs/ashare/
  manifest.json
  summary.json
  report.md
  seed_7/
    models/*.zip
    selection.json
    validation_metrics.json
    test_metrics.json
    test_*.csv
    *.png
```

其余种子同样保存到对应目录。先读 `report.md` 与 `summary.json`，再检查 `selection.json` 是否仅来自 validation，以及 `test_*.csv` 的净值、交易和持仓记录。`evaluate` 会加载已保存模型重跑测试，不进行测试期训练。

指标口径：总收益与年化收益使用几何复利；年化波动使用样本标准差；Sharpe 的无风险收益假设为 0；Sortino 的目标收益为 0，下行偏差按全部期间计算负收益平方的均方根；最大回撤为正的损失比例并包含初始净值 1；零波动或无回撤导致的未定义比率输出 JSON `null`。年化默认 252 个交易日，短样本年化指标只能作为描述。

## 进一步研究

后续应优先冻结比较规则并积累未查看的新市场数据，补齐公司行动和真实盘口执行。共享时间编码、PER、候选多样性目标和在线适应仍需分别做消融。完整 MetaTrader 复现还需实现其行为克隆式多样性学习目标；完整 RLPortfolio 对照需采用其单独的直接策略优化训练协议，不能把两者默认当成当前实现的一部分。
