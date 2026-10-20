# 日频事件驱动模拟成交

`rlasset.execution` 把组合目标转为整数股订单，并独立维护现金、持股、当日买入和可卖数量。它用于检验策略经过交易约束后的行为；没有连接券商，也不会提交真实委托。

## 事件顺序与价格隔离

1. 交易日 t 收盘后，由策略在截至 t 的信号面板上生成目标权重。
2. 调用方提供下一个交易日的**未复权**执行数据，以开盘价计算账户开盘资产和目标股数。
3. `rebalance` 先执行卖单，再执行买单；每笔订单一次撮合，未成交余额立即撤销。
4. `close_snapshot` 用当日未复权收盘价估值，返回实际持仓权重，并关闭该交易日的下单阶段。
5. 推进至后续交易日后，前一交易日买入的股份转为可卖；同一天重复推进不会再次解锁。

`decision_date` 必须严格早于执行日期。调用方负责交易日历及 t 到下一交易日的对应关系；模块本身只检查日期先后，不能证明目标确实来自当时可用的信息。禁止先看执行日收盘价生成权重，再填入旧的 `decision_date`。

目标股数的计算只用目标权重、现有现金/股数和执行日开盘价。执行日收盘价、成交量、停牌或涨跌停标志不参与订单数量计算。开盘价成交也是简化：真实开盘集合竞价中不能保证观察到最终开盘价后仍按该价成交。

复权信号价格可以用于策略特征和连续收益研究，但不可当作每股成交价格。`ExecutionBar(price_basis="hfq")` 会被拒绝。默认 `price_basis="raw"` 是调用方的声明，程序无法仅从数值判断价格是否真的未复权。

## 接口

```python
from rlasset.execution import ExecutionBar, ExecutionConfig, PaperBroker

broker = PaperBroker(
    initial_cash=100_000,
    config=ExecutionConfig(
        lot_size=100,
        commission_rate=0.0003,
        min_commission=5,
        slippage_bps=5,
        participation_rate=0.1,
        t_plus_one=True,
    ),
)

bars = {
    "510300": ExecutionBar(
        date="2024-01-03",
        ticker="510300",
        open=3.50,
        close=3.52,
        volume=1_000_000,  # 单位：股/基金份额；不是“手”
        halted=False,
        limit_up=None,
        limit_down=None,
        price_basis="raw",
    ),
}
records = broker.rebalance(
    {"510300": 0.8},  # 只填风险资产，剩余 20% 隐含为现金
    bars,
    decision_date="2024-01-02",
    client_order_prefix="strategy-20240103",
)
snapshot = broker.close_snapshot(bars)
order_rows = [record.to_dict() for record in records]
```

所有执行价、收盘价必须有限且大于零，成交量必须有限且非负。同一批次的所有 bar 必须属于同一日期，字典键必须与 `ticker` 相同。目标权重有限、非负，合计不能大于 1；未出现的已有持仓被视为目标清仓。所有目标证券和已有持仓都必须有 bar。

| 接口 | 返回或行为 |
|---|---|
| `PaperBroker(initial_cash, config=None, initial_positions=None)` | 创建账户；导入持仓视为已结算且可卖 |
| `advance_date(date)` | 同日无操作；较早日期抛出 `ValueError`；跨日结算并重置当日成交容量 |
| `submit_order(client_order_id, ticker, side, quantity, bar)` | `side` 为 `buy` / `sell`；返回不可变 `OrderRecord` |
| `rebalance(target_weights, bars, *, decision_date, client_order_prefix)` | 先按开盘资产计算全部订单，再依证券代码顺序先卖后买；返回记录列表 |
| `close_snapshot(bars)` | 收盘估值、标记阶段结束，返回可 JSON 序列化的字典 |
| `cash`, `available_cash`, `reserved_cash` | 当前现金、可用现金、冻结现金；本 IOC 模型冻结现金恒为 0 |
| `positions`, `sellable` | 返回复制的股数账本，外部修改副本不影响账户 |
| `orders` | 已记录委托的不可变记录元组，包含拒单 |

`ExecutionConfig` 的所有默认值见示例，其中默认滑点为 **0 bps**，示例显式设为 5。费用、交易单位、T+1 和成交参与率都是模拟配置，不能自动代表某个账户、产品或日期的真实规则。允许设置 `t_plus_one=False`，但调用方必须按产品核实结算制度。

## 下单、费用与持仓规则

- 买单必须是 `lot_size` 的整数倍，默认 100 股/份。卖单可按整手出售；只有请求清空现有持仓时才允许一次性卖出非整手尾数。受到成交容量或 T+1 约束而无法完全清仓时，部分成交仍按整手向下取整。
- 不允许卖空。卖出请求超过总持股量直接拒绝；请求没有超过总量但超过可卖数量时，可以部分成交已结算股份。
- 默认 T+1 账本只禁止卖出当日新买入股份。已有的可卖股份不因同日另有买入而被冻结。跨日指调用方推进至下一次提供的交易日期，不是运行机器的自然时钟。
- 买价为 `open × (1 + slippage_bps / 10000)`，卖价为 `open × (1 - slippage_bps / 10000)`，提供价格限制时将模拟价限制在该区间内。
- 每个发生实际成交的 IOC 订单只收一次佣金：`max(min_commission, 成交金额 × commission_rate)`。拒单不收佣金。买入前把佣金纳入可用现金约束，必要时减少成交股数。
- 卖出净所得也不能令现金变负。例如零现金账户卖出极低价格的少量股份时，若所得不足最低佣金，该单拒绝。
- 多个买单按代码顺序消耗卖出所得和剩余现金。费用、滑点和整数股约束可能使后面的买单部分成交；实际权重通常与目标不同。

现金更新为：

```text
买入后现金 = 原现金 - 成交股数 × 成交价 - 佣金
卖出后现金 = 原现金 + 成交股数 × 成交价 - 佣金
收盘 NAV = 现金 + Σ(实际持股数 × 未复权收盘价)
```

没有挂单跨事件存续，因此无需持久冻结资金。每笔买单在更新账本前检查全部实际支出；未成交部分没有资金占用。这里的 `reserved_cash=0` 不代表真实券商未成交委托也不冻结资金。

## 停牌、涨跌停与成交容量

`halted=True` 时买卖均拒绝，但持仓仍使用所给收盘价估值。

提供 `limit_up` 时，在开盘价达到上限时拒绝买入；卖出不因此被阻挡。提供 `limit_down` 时，在开盘价达到下限时拒绝卖出；买入不因此被阻挡。未提供限制值时，模块不会自动推导交易所价格限制，也不会仅凭某一日涨跌幅猜测封板。

当日该证券全部买卖成交数量合计不能超过：

```text
floor(当日成交量 × participation_rate)
```

该额度在多个订单之间共享，同一日期重复调用不会重置。输入成交量单位必须是**股/基金份额**；若数据源返回“手”，调用方必须先换算并在数据元信息中记录单位。

**日线总成交量包含开盘之后的信息。** 本模块仅把它作为离线撮合容量假设；它不进入目标股数计算，但确实影响模拟成交结果。因此本结果不能称为严格盘中可实施的执行回放。真实开盘可成交量、盘口深度、排队优先级、集合竞价资格以及实际冲击需要更细粒度数据。这里拒绝不利方向的封板委托也是保守的情景规则，不是对真实逐笔撮合的重现。

## 记录、重放与拒单原因

`OrderRecord.to_dict()` 中日期为 ISO 字符串；未成交价为 `None`，无 NaN。主要字段：

```text
client_order_id, date, ticker, side
requested_quantity, filled_quantity, cancelled_quantity
filled_price, notional, commission, slippage_cost
status, reject_reason, cash_after, position_after, sellable_after
```

`status` 为 `filled`、`partially_filled` 或 `rejected`。未成交数量写入 `cancelled_quantity`，不留待后续交易日再撮合。完全成交时 `reject_reason=None`；部分成交时该字段解释剩余数量无法成交的约束。

| `reject_reason` | 含义 |
|---|---|
| `invalid_lot` | 买入/部分卖出数量不满足整手规则 |
| `halted` | 停牌情景 |
| `limit_up` / `limit_down` | 涨停买入 / 跌停卖出受限 |
| `no_volume` | 当日输入成交量为零 |
| `participation_limit` | 该证券当日剩余模拟容量不足 |
| `t_plus_one` | 当日新买入股份尚不可卖 |
| `insufficient_position` | 请求卖出超过总持股 |
| `insufficient_cash` | 支出或最低佣金超出可用现金 |
| `session_closed` | 已进行收盘估值，不能再按本日开盘价下新单 |

一个订单可能同时受多项约束限制；记录保留执行裁剪过程中最后起约束作用的原因，不输出完整的约束归因集合。

相同 `client_order_id`、日期、证券、方向、请求数量返回第一次的记录，不重复扣款、记佣金或消耗成交容量；这同样适用于跨日重放旧订单。相同 ID 配合不同订单意图抛出 `ValueError`，原账本不变。拒单也占用该 ID；改变条件后重试必须使用新 ID。

`client_order_prefix` 提供组合批次级幂等性，相同决策日期、执行日期和目标权重返回原批次记录；冲突前缀抛出 `ValueError`。ID 和前缀去重在当前对象内存中有效，进程重启后不会自动恢复。若需要跨进程恢复，需额外实现持久账本和事务性订单状态存储。

## 收盘快照与已知边界

快照包括 `date`、`nav`、`cash`、`available_cash`、`reserved_cash`、`positions`、`sellable`、`weights`、`mark_prices`、`cumulative_commission` 和 `cumulative_slippage`。`weights` 用真实股数和收盘市值计算，包含 `cash` 项。停牌证券也需提供估值价格；模块不自行前向填充缺失价格。零资产账户的各项权重置为零。

当前未建模：

- 分红现金入账、拆并份额、除权和其他公司行动。跨越这些事件时，raw 账本净值不能当作已含分红的总收益，更不能直接与复权策略净值按同一口径比较。
- 卖出税费、过户费、真实账户费率差异、价格最小变动单位和按分舍入的券商结算规则。
- 逐笔排队、延迟、开盘竞价成交可得性、动态冲击、撤单时序、融资和借券。
- 真实交易日历、产品级规则自动识别和外部持仓/资金对账。

`tests/test_execution.py` 用可手算的账户检查费用和滑点守恒、T+1、尾数清仓、跨订单成交容量、现金约束、拒单、重复委托、日期推进、先卖后买、请求量对下一日收盘价/成交量的独立性，以及连续多日收据与持仓净值的对账。这些测试证明的是所列模拟规则的实现一致性，不是实盘成交可达性或策略获利能力。
