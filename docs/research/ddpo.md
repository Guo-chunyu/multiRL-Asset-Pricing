# Deep Deterministic Portfolio Optimization：理论、源码核对与项目实现

研究日期：2026-10-19。论文版本：[arXiv:2003.06497v2](https://arxiv.org/abs/2003.06497v2)，2020-04-09，25 页。作者：Ayman Chaouki、Stephen Hardiman、Christian Schmidt、Emmanuel Sérié、Joachim de Lataillade。以下页码为 PDF 印刷页码，从 1 开始。

这篇论文为本项目提供的是**可检验的控制基准和实验方法**。它研究单个资产的连续持仓调整，交易费用与风险惩罚共同决定控制策略；其中两个参照策略的参数还需要数值搜索。它没有证明神经网络策略在真实多资产市场达到全局最优。论文结果属于模拟实验，见第 2 节、第 3 节与图 2 注释。[论文全文](https://arxiv.org/pdf/2003.06497v2)

## 1. 精确任务定义及时间顺序

依据论文第 1.1、2.1 节，式 (3)-(8)，第 3-5 页。为避免市场收益与 RL 奖励同名，本文分别记作 \(z\) 与 \(u\)。

\[
s_t=(\pi_t,p_t),\quad a_t=\pi_{t+1}-\pi_t,\quad
\pi_{t+1}=\pi_t+a_t,
\]
\[
p_{t+1}=\rho p_t+\epsilon^{p}_{t+1},\qquad
z_{t+1}=p_t+\epsilon^{r}_{t+1},
\]
\[
u_t=\pi_{t+1}z_{t+1}-c(a_t)-q(\pi_{t+1}).
\]

决策时已知当前持仓与预测信号；先形成下一期持仓，再获得下一期收益。噪声收益和下一期信号不能输入当前决策。持仓可以为负，动作是增量，没有多资产权重和为 1 的约束。预测过程不受交易影响；交易费用是奖励中的成本，并未建立永久市场冲击模型。

本项目解析基准明确采用 \(|\rho|<1\)、独立标准正态创新和

\[
\epsilon^{p}_{t+1}=\sqrt{1-\rho^2}\,\xi_{t+1},\quad
p_0\sim\mathcal N(0,1),\quad
\epsilon^{r}_{t+1}=\sigma_r\eta_{t+1}.
\]

这样预测信号的平稳方差精确为 1。零均值、独立、不可预测的收益噪声改变奖励方差，不改变该期望控制问题的最优反馈。这个结论依赖上述假设，不能推广到相关噪声、未知潜在状态或带尾部风险效用的情形。

## 2. 三个基准的适用边界

| 基准 | 每期奖励与约束 | 参照策略 | 本项目使用方式 |
|---|---|---|---|
| 二次费用、二次风险 | \(p_ty-\Gamma(y-\pi_t)^2-\lambda y^2\)，\(y\in\mathbb R\) | 平稳线性反馈 | 已实现解析反馈及独立 Bellman 检验 |
| 比例费用、二次风险 | \(p_ty-\Gamma|y-\pi_t|-\lambda y^2\) | 不交易区域；区域外交易到边界 | 保留为扩展实验，不把常宽区域称为精确解 |
| 比例费用、最大仓位 | \(p_ty-\Gamma|y-\pi_t|\)，\(|y|\le M\) | 信号超过正负阈值后交易至 \(\pm M\)，其余保持 | 保留为扩展实验，阈值需要估计 |

出处：第 2.1.1-2.1.3 节，式 (9)-(18)，第 6-8 页。[论文全文](https://arxiv.org/pdf/2003.06497v2)

第二个基准的常宽近似可明确写为

\[
m_t=\frac{p_t}{2\lambda},\qquad
y_t=\operatorname{clip}(\pi_t,m_t-b,m_t+b),\qquad a_t=y_t-\pi_t.
\]

论文用网格搜索选择 \(b\)；作者演示代码还允许同时拟合斜率与宽度。因此不能把任意 \(b\) 或一阶 myopic 阈值当作无限期精确最优解。第三个基准是

\[
y_t=\begin{cases}
M&p_t>q,\\
-M&p_t<-q,\\
\pi_t&|p_t|\le q.
\end{cases}
\]

形状明确不等于阈值已知；该论文实验同样用数值搜索近似 \(q\)。两种搜索都应在训练或验证模拟路径上完成，再用独立路径报告表现。

## 3. 本项目独立推导的二次基准

下面给出代码实际使用的推导，扩展论文第 2.1.1 节以明确折扣率。令当前状态 \((x,p)=(\pi_t,p_t)\)，下一持仓为 \(y\)，\(\Gamma\ge0\)、\(\lambda>0\)、\(0\le\gamma\le1\)。

对于 \(\gamma<1\)，无限期目标为

\[
V(x,p)=\sup E\sum_{t\ge0}\gamma^t\left[p_ty_t-\Gamma(y_t-x_t)^2-\lambda y_t^2\right].
\]

设 \(V(x,p)=-A x^2+Bxp+Cp^2+D\)。Bellman 方程中依赖控制 \(y\) 的部分为

\[
-H y^2+\left[2\Gamma x+(1+\gamma\rho B)p\right]y,
\qquad H=\Gamma+\lambda+\gamma A>0.
\]

比较 \(x^2\) 与 \(xp\) 系数得到

\[
A=\Gamma-\frac{\Gamma^2}{H},\qquad
B=\frac{\Gamma(1+\gamma\rho B)}H.
\]

将正根有理化，得到适合直接计算、且覆盖 \(\gamma=0\) 的公式：

\[
b=\lambda+\Gamma(1-\gamma),\qquad
A=\frac{2\Gamma\lambda}{b+\sqrt{b^2+4\gamma\Gamma\lambda}},
\]
\[
k_x=\frac\Gamma H,\qquad
B=\frac{k_x}{1-\gamma\rho k_x},\qquad
k_p=\frac{1}{2H(1-\gamma\rho k_x)},
\]
\[
\boxed{y^*=k_xx+k_pp,\qquad a^*=y^*-x.}
\]

其余系数为

\[
C=\frac{(1+\gamma\rho B)^2}{4H(1-\gamma\rho^2)},\qquad
D=\frac{\gamma C\operatorname{Var}(\epsilon^p)}{1-\gamma}.
\]

当 \(\gamma=1\) 时使用平均奖励方程 \(g+h=\max_y(u+Eh')\)，不能使用发散的常数 \(D\)。同一 \(A,B,C\) 给出相对值函数和稳定反馈，\(g=C\operatorname{Var}(\epsilon^p)\)。这要求平稳预测过程及稳定控制；有限期截断、期末强制清仓、仓位边界都需要另行求解。

三个可核验特例：

- 无交易费用：\(\Gamma=0\Rightarrow y^*=p/(2\lambda)\)。
- 只优化当期：\(\gamma=0\Rightarrow y^*=(p+2\Gamma x)/[2(\Gamma+\lambda)]\)。
- \(\rho=0.9,\Gamma=1,\lambda=0.3,\gamma=1\) 时，\(k_x=0.58210916542\)，\(k_p=0.61132852804\)，理论平稳平均奖励约为 \(0.64201457629\)。

与论文式 (12) 对照，\(\omega=1-k_x\)，\(\psi=\omega/[1-(1-\omega)\rho]\)，因此 \(y=(1-\omega)x+\omega\psi p/(2\lambda)\)。**第 7 页两个 \(f_c(x)\) 表达式印刷上不等价**：第一式根号里是 \(1+4/x\)，而第二式是 \(x(\sqrt{x^2+4}-x)/2\)。后式有理化应得 \(2/[1+\sqrt{1+4/x^2}]\)。已查看 PDF 页图确认；实现依据独立 Riccati 推导，并用矩阵方程数值解交叉验证，避免照抄此处排印。核对页图保存在 `research/papers/ddpo-page-07.png`。

## 4. DDPG、PER 与 TD3 选择依据

DDPO 使用 actor-critic、目标网络和经验回放；额外使用优先经验回放，并为最大仓位环境增加平滑惩罚。对应论文第 2.2 节、算法 1、附录 B，第 8-10、22-25 页。[论文全文](https://arxiv.org/pdf/2003.06497v2)

以下是标准实现所需的数学接口，\(d_i\) 表示真正终止：

\[
y_i=r_i+\gamma(1-d_i)Q_{\bar\theta}(s'_i,\mu_{\bar\phi}(s'_i)),\quad
\delta_i=y_i-Q_\theta(s_i,a_i),
\]
\[
P(i)=\frac{(|\delta_i|+\varepsilon)^\alpha}{\sum_j(|\delta_j|+\varepsilon)^\alpha},\quad
w_i=\frac{(NP(i))^{-\beta}}{\max_j(NP(j))^{-\beta}},
\]
\[
L_Q=\frac1N\sum_i w_i\delta_i^2,\qquad
L_\mu=-\frac1N\sum_i Q_\theta(s_i,\mu_\phi(s_i)).
\]

注意 importance weight 乘的是平方误差，而非先把误差乘权重再平方。优先回放改变抽样分布；新样本最高优先级、正数 epsilon、beta 退火和真正终止 mask 都必须明确。作者 `agent.py` 的实现可以核对这些细节。DPG 是可微连续策略的梯度理论；它本身不证明非线性 critic 与 actor 联合训练收敛。[原始 DPG 论文](https://proceedings.mlr.press/v32/silver14.html)

本项目建议 **TD3 为主实验，DDPG 为同预算对照**。TD3 对 critic 过估计、actor 追逐 critic 误差提供三个直接针对措施：双 critic 目标最小值、延迟 actor 更新、目标动作平滑。依据是原始 TD3 论文第 4.2、5.2、5.3 节及算法 1，不是 DDPO 论文提出了 TD3，也不是已经证明 TD3 在金融数据上更好。[TD3 论文与代码入口](https://proceedings.mlr.press/v80/fujimoto18a.html)

\[
\tilde a=\operatorname{clip}(\mu_{\bar\phi}(s')+\operatorname{clip}(\epsilon,-c,c),a_{min},a_{max}),
\quad
y=r+\gamma(1-d)\min_{j=1,2}Q_{\bar\theta_j}(s',\tilde a).
\]

项目层面的推论：先用标准均匀 replay 的 TD3 固定基线；PER 应作为单独消融，而不把“DDPO 使用了 PER”解释为任何金融场景都需要 PER。高 TD error 可能来自不可约收益噪声，也可能来自真实稀有状态。需要实验区分，不能只凭训练速度判断。目标动作若通过 softmax 转为组合权重，要明确 critic 接收的动作表示与环境执行的权重映射。

## 5. 作者源码核对与复现差异

已拉取作者仓库固定提交 [`1afd95def1d7837ba34ed416b4d52dee3fbeaa86`](https://github.com/CFMTech/Deep-RL-for-Portfolio-Optimization/tree/1afd95def1d7837ba34ed416b4d52dee3fbeaa86)，本地放在 `research/sources/ddpo/upstream/`。这份仓库仅作为研究源，运行实现为本项目自行推导、重新编写的模块。

| 读过的文件/区域 | 已核对的内容 | 对本项目的影响 |
|---|---|---|
| [env.py](https://github.com/CFMTech/Deep-RL-for-Portfolio-Optimization/blob/1afd95def1d7837ba34ed416b4d52dee3fbeaa86/env.py)，构造、reset、step、apply、test/test_apply | 实际状态顺序为 `(p, pi)`；step 使用 `pi + action`；费用按实际执行持仓变化计提 | 明确本项目参数名，避免把 target 和 trade 混淆 |
| 同一文件 58-61、224-256 行 | docstring 对 `trade_l1/l2` 的文字说明颠倒；实际 l1 为绝对值，l2 为平方 | 以执行代码与公式对照为准 |
| 同一文件 `apply` | quadratic 默认参数产生 myopic 策略；文档说明允许传拟合参数 | 不能直接把 `env.apply()` 默认输出叫精确动态最优 |
| [utils.py](https://github.com/CFMTech/Deep-RL-for-Portfolio-Optimization/blob/1afd95def1d7837ba34ed416b4d52dee3fbeaa86/utils.py)，`build_ou_process` | 离散 OU 增量，却按连续 OU 方差归一；其参数顺序为 `(T, theta, sigma, random_state)`，env 调用时传 `(T, sigma, theta, ...)` | 本项目用显式 AR(1) 和精确创新方差；论文演示参数二者均为 0.1，交换在该设置下不显现 |
| [agent.py](https://github.com/CFMTech/Deep-RL-for-Portfolio-Optimization/blob/1afd95def1d7837ba34ed416b4d52dee3fbeaa86/agent.py)，`step`、pretrain、train 更新段 | gamma 默认 0.99；TD target、PER 权重、软更新、梯度裁剪 | 报告折扣目标与平均奖励参照之间的区别 |
| [memory.py](https://github.com/CFMTech/Deep-RL-for-Portfolio-Optimization/blob/1afd95def1d7837ba34ed416b4d52dee3fbeaa86/memory.py)，Memory、SumTree 抽样、PrioritizedMemory | sum-tree 抽样、priority 更新、最高优先级与容量策略 | 理解 PER；不直接移植整套历史实现 |
| [models.py](https://github.com/CFMTech/Deep-RL-for-Portfolio-Optimization/blob/1afd95def1d7837ba34ed416b4d52dee3fbeaa86/models.py)，Actor/Critic | 小型 ReLU MLP，actor 输出线性无界 | 论文不是大语言模型架构，也不是 simplex actor |
| [summary.py](https://github.com/CFMTech/Deep-RL-for-Portfolio-Optimization/blob/1afd95def1d7837ba34ed416b4d52dee3fbeaa86/summary.py)，线性费用案例、网格搜索、训练/评估段 | 近似参照参数来自网格搜索；演示小网络和奖励缩放 | 分开记录理论解、拟合参照和训练策略 |

两个定量差异值得保留：

1. 原 `utils.py` 的归一化后，离散过程平稳方差为 \(1/(1-\theta/2)\)。当 \(\theta=0.1\) 时是 \(1.0526316\)，因此论文表中绝对 reward 数值不能不经模型对齐就与本项目单位方差信号比较。
2. 原 `env.test()` 在已经缩放的 reward 上加未同尺度缩放的风险项重建 PnL；`scale_reward=1` 时不受这个问题影响。本项目分别计算 PnL 与风险惩罚，避免由 reward 逆推。

## 6. 已落地的检验与不能外推的结论

`src/rlasset/theory.py` 实现解析反馈、myopic 对照、零持仓对照和共享路径评价；`tests/test_theory.py` 用独立的 SciPy 广义离散 Riccati 数值求解检查 Bellman 等式与控制点的一阶导数，并覆盖无费用、零折扣、负相关信号和非法模型参数。测试不是再次复制 scalar 闭式公式得到“期望答案”。

固定设置 `seed=7, episodes=12, horizon=5000, rho=.9, cost=1, risk=.3, discount=1` 的一次实际运行结果：

| 策略 | 平均每期奖励 | episode 均值的标准误 |
|---|---:|---:|
| 解析平稳反馈 | 0.658558 | 0.016847 |
| myopic | 0.615123 | 0.016419 |
| 零持仓 | 0.000000 | 0.000000 |

这是**解析基准实现的 smoke/数值检验，不是 RL 训练结果**。有限样本均值不要求等于平稳理论值，也不保证每条模拟路径都优于 myopic。若要评价 TD3 的控制学习能力，需要另外训练，并同时报告 reward gap、与解析动作/持仓的误差、多 seed 分布以及失败率。

从该理论迁移到真实多资产长期持有组合，需要重新定义自融资资金账户、现金、费用、权重漂移、执行时间、观察窗口和风险函数。\(w^\top\Sigma_t w\) 可以作为过去数据估计的风险代理，但不是未来真实方差。价格特征窗口也不保证系统完全 Markov。线性 reward 的风险正则与完整的期末财富 mean-variance 目标通常不是同一个问题；后者可能涉及时间不一致性。

因而本项目不能据此宣称：全局最优、必然收敛、稳定战胜基准、市场无套利或已完成资产定价均衡证明。应实际检验含费用净值、超额收益、风险、换手、不同随机种子、时间外样本和市场分段。论文图 2 对 maxpos 只展示收敛 agents 并明确提醒正向选择偏差；本项目应把发散、训练失败和差 seed 全部纳入报告。

## 7. 资料与实现复核记录

- 本地论文：`research/papers/ddpo-2003.06497v2.pdf`；文本：同名 `.txt`；公式核对图：`ddpo-page-07.png`。
- 核心阅读：DDPO 第 1、2、3、4 节和附录 A/B；特别核对式 (5)-(18)、算法 1、第 7 页公式与图 2 的样本选择注释。
- 补充主来源：[Deterministic Policy Gradient Algorithms](https://proceedings.mlr.press/v32/silver14.html)；[Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477)，第 4-6 节和算法 1。
- 实现验证命令：`.venv/Scripts/python.exe -m pytest tests/test_theory.py tests/test_metrics.py -q`，在项目根目录运行，包需安装或设置 `PYTHONPATH=src`。
- 最后独立检查结果：33 个测试通过（theory 18，metrics 15）。这只覆盖上述两模块；整个项目测试由主流程另行运行。
