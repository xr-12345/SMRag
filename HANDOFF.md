# PowerfulMedRAG 项目交接文档

> 面向接手本项目的研究与工程同事。本文记录研究动机、符号、模型、全部核心公式、
> 实验协议、现有结果、代码文件职责、产物结构、复现方法和当前边界。

## 1. 一句话概括

本项目研究的是：在医疗主动问诊中，将患者的**真实临床状态**与患者如何**观察、
记忆并报告**该状态分层建模，再用噪声感知的期望信息增益（EIG）选择问题；对于可能
损害诊断的历史回答，使用有轮数预算的 leave-one-out 回溯澄清。

当前推荐主方法是：

1. 使用结构化疾病—临床状态模型维护鉴别诊断后验；
2. 用 EIG 选择下一个问题；
3. 在准备停止或即将用完预算时，对历史回答计算
   `错误概率 × 诊断影响`；
4. 只复问效用最高且超过阈值的回答；
5. 回答冲突时弃权为 `UNKNOWN`，重放全部历史并重新判断是否应停止。

主方法配置名为 `retro_utility_u050_b1`：效用阈值 0.05，每例最多一次回溯澄清。

## 2. 研究问题与演化过程

最初目标是减少 MedRAG 式问诊的轮数，同时不牺牲诊断准确率。项目逐步形成了下面的
研究链条：

1. **概率化鉴别诊断**：不把疾病候选当作无权重列表，而是维护后验概率。
2. **主动选问**：选择预期最大程度降低疾病熵的问题，而不是只依赖图节点度数。
3. **显式回答噪声**：将“不知道、不确定、条件差异和真正误报”区别处理。
4. **历史可靠性门控**：根据此前出现的不确定/不知道线索动态启用误报先验。
5. **PaMis-style 基线**：局部异常回答触发一次同证据澄清。
6. **回溯澄清**：发现局部异常检测召回太低后，改为在更多证据到达后回看历史回答。
7. **预算化诊断效用**：不仅判断“回答是否可能错”，还判断“它是否足以改变诊断”。

最重要的研究认识是：**可能错误不等于值得复问**。只看错误概率的检测器很精确，但
几乎不改善诊断；错误概率乘以诊断影响后，中高噪声下才出现稳定的准确率收益。

## 3. 符号与随机变量

| 符号 | 含义 |
|---|---|
| $D$ | 疾病，取值 $d\in\mathcal D$ |
| $j$ | 临床变量/证据索引 |
| $Z_j$ | 患者关于变量 $j$ 的真实临床状态 |
| $R_j$ | 患者报告出来的回答；额外允许 `UNKNOWN` |
| $M_j$ | 隐藏报告模式：certain、uncertain、unknown、misreported |
| $C_j$ | 可观察确定性提示：none、certain、uncertain |
| $H_t$ | 第 $t$ 轮前的完整问诊历史 |
| $b_t(d)$ | $P(D=d\mid H_t)$，当前疾病后验 |
| $q_{d,j,t}(z)$ | 给定疾病与同变量既往报告后的真实状态分布 |
| $c_j$ | 问题成本 |
| $\tau$ | 诊断后验停止阈值 |
| $\rho$ | 模拟患者的总不可靠回答模式质量 |

`FeatureKey` 将时间、活动、部位等上下文作为变量身份的一部分。因此
`pain[activity=walking]` 与 `pain[activity=resting]` 是两个变量，不会被误判为直接矛盾。

## 4. 生成模型与训练公式

### 4.1 疾病先验

设训练集中疾病 $d$ 的病例数为 $N_d$，总病例数为 $N$，疾病数为 $K$，
疾病先验平滑参数为 $\alpha_D$：

$$
P(D=d)=\frac{N_d+\alpha_D}{N+K\alpha_D}.
$$

对应实现：`src/powerful_medrag/estimation.py`。

### 4.2 全局临床状态频率

对变量 $j$ 的状态 $z$，全局显式观察计数为 $G_{jz}$，状态空间大小为
$|\mathcal Z_j|$。Laplace/Dirichlet 平滑后的全局分布为：

$$
\bar p_j(z)=
\frac{G_{jz}+\alpha}
{\sum_{z'}G_{jz'}+\alpha|\mathcal Z_j|}.
$$

### 4.3 疾病条件临床状态分布

疾病 $d$ 下变量 $j$ 的显式状态计数为 $N_{djz}$，显式观察总数为
$N_{dj}=\sum_z N_{djz}$。经验贝叶斯层级强度为 $\lambda$：

$$
P(Z_j=z\mid D=d)=
\frac{N_{djz}+\alpha+\lambda\bar p_j(z)}
{N_{dj}+\alpha|\mathcal Z_j|+\lambda}.
$$

当 $\lambda=0$ 时退化为普通 Dirichlet–Categorical；二元变量时等价于
Beta–Bernoulli 平滑。**缺失状态不进入分子或分母，绝不自动算成阴性。**

## 5. 两层回答通道

### 5.1 隐藏报告模式

$$
M_j\in\{\text{certain},\text{uncertain},\text{unknown},\text{misreported}\}.
$$

默认通道中，每种模式的 `(correct, unknown, wrong)` 概率为：

| 模式 | correct | unknown | wrong |
|---|---:|---:|---:|
| certain | 0.98 | 0.01 | 0.01 |
| uncertain | 0.55 | 0.35 | 0.10 |
| unknown | 0.04 | 0.94 | 0.02 |
| misreported | 0.08 | 0.07 | 0.85 |

若 $R=Z$，使用 `correct`；若 $R=UNKNOWN$，使用 `unknown`；若是其他状态，
`wrong` 质量平均分配到其余 $|\mathcal Z_j|-1$ 个状态：

$$
P(R=r\mid Z=z,M=m)=
\begin{cases}
a_m,&r=z,\\
u_m,&r=UNKNOWN,\\
w_m/(|\mathcal Z_j|-1),&r\ne z,\ r\ne UNKNOWN.
\end{cases}
$$

### 5.2 确定性提示对应的模式先验

| 可观察提示 | certain | uncertain | unknown | misreported |
|---|---:|---:|---:|---:|
| none | 0.82 | 0.10 | 0.05 | 0.03 |
| certain | 0.93 | 0.03 | 0.01 | 0.03 |
| uncertain | 0.18 | 0.70 | 0.09 | 0.03 |

给定提示 $C=c$ 后，边缘回答通道为：

$$
P(R=r\mid Z=z,C=c)=
\sum_m P(M=m\mid C=c)P(R=r\mid Z=z,M=m).
$$

对应实现：`src/powerful_medrag/channel.py`。

### 5.3 报告模式后验

当前预测真实状态分布记为 $p_t(z)$。观察回答后：

$$
P(M=m\mid R=r,H_t)
=
\frac{\pi_m(C)\sum_z p_t(z)P(R=r\mid Z=z,M=m)}
{\sum_{m'}\pi_{m'}(C)\sum_z p_t(z)P(R=r\mid Z=z,M=m')}.
$$

代码中的 `UpdateResult.misreport_probability` 即该后验中 `MISREPORTED` 的质量。

## 6. 疾病后验与重复回答

### 6.1 单轮疾病似然

对疾病 $d$，回答 $r_t$ 的似然为：

$$
L_d(r_t)=\sum_z q_{d,j,t-1}(z)
P(R_j=r_t\mid Z_j=z,C_t).
$$

疾病后验更新：

$$
b_t(d)=
\frac{b_{t-1}(d)L_d(r_t)}
{\sum_{d'}b_{t-1}(d')L_{d'}(r_t)}.
$$

### 6.2 同一变量共享一个真实状态

重复询问同一个变量时，不会重新采样一个新的 $Z_j$。每个疾病下保存该变量的
状态后验：

$$
q_{d,j,t}(z)\propto q_{d,j,t-1}(z)
P(R_j=r_t\mid Z_j=z,C_t).
$$

这样两次报告是关于同一个真实临床状态的重复测量，而不是两个独立症状。

### 6.3 疾病熵

$$
H(D\mid H_t)=-\sum_d b_t(d)\log_2 b_t(d).
$$

对应实现：`src/powerful_medrag/belief.py`。

## 7. 主动选问公式

### 7.1 预测回答分布

对候选问题 $j$ 和可能回答 $r$：

$$
P(r\mid H_t,j)=\sum_d b_t(d)L_{d,j}(r).
$$

回答为 $r$ 时的反事实疾病后验：

$$
b_{t+1}^{(r,j)}(d)=
\frac{b_t(d)L_{d,j}(r)}
{P(r\mid H_t,j)}.
$$

### 7.2 期望信息增益

$$
EIG(j)=H(b_t)-
\sum_rP(r\mid H_t,j)H\!\left(b_{t+1}^{(r,j)}\right).
$$

单位成本效用：

$$
U(j)=\frac{EIG(j)}{c_j^\gamma},
$$

默认 $\gamma=1$。每轮选择效用最高且满足前置问题条件的未问变量。

`NumpyQuestionSelector` 是同一公式的向量化实现；测试确保它与参考实现一致。

### 7.3 静态流行率基线

若变量有默认/阴性状态 $z_0$，静态得分为：

$$
score_{prev}(j)=\frac1{|\mathcal D|}
\sum_d\left[1-P(Z_j=z_0\mid D=d)\right].
$$

### 7.4 MedRAG-RDC 结构化适配

在当前 Top-5 疾病相连的特征并集中，使用 MedRAG 公式 15：

$$
RDC(j)=\frac{n-1}{degree(j)},
$$

其中 $n$ 是知识图谱中的特征节点数。该实现是透明的结构化适配，不是原论文
未公开多轮模块的完整复现。

对应实现：`src/powerful_medrag/questioning.py`。

## 8. 模拟回答噪声与公平配对

总不可靠模式质量为 $\rho$ 时，生成侧模式先验为：

$$
P(M=\text{certain})=1-\rho,
$$

$$
P(M=\text{uncertain})=0.40\rho,\quad
P(M=\text{unknown})=0.30\rho,\quad
P(M=\text{misreported})=0.30\rho.
$$

同一病例、噪声率、特征和该特征第几次被问，由稳定哈希
`hash(seed, case_id, noise_rate, feature, occurrence)` 决定回答。因此不同策略即使选问
顺序不同，重叠特征仍得到相同模拟回答。这是配对比较成立的基础。

对应实现：`src/powerful_medrag/simulator.py` 与 `src/powerful_medrag/benchmark.py`。

## 9. 动态误报门控

### 9.1 一般动态先验注入

若门控为当前回答给出误报概率 $g_t$，则用它替换基础 cue prior 的 misreported
质量，其他模式按原相对比例归一化：

$$
\pi_{mis,t}=g_t,
$$

$$
\pi_{m,t}=(1-g_t)
\frac{\pi_m^{base}}{\sum_{m'\ne mis}\pi_{m'}^{base}},\quad m\ne mis.
$$

### 9.2 启发式惊讶度门控

预测概率 $p=P(R_t\mid H_t)$，惊讶度与超额惊讶度为：

$$
s=-\ln p,\qquad e=\max(0,s-\tau_s).
$$

未截断 logit 为：

$$
\ell=logit(p_0)+w_se
+\mathbb 1[C=\text{certain}]w_ce
-\mathbb 1[C=\text{uncertain}]w_u
-\mathbb 1[R=UNKNOWN]w_k
+w_x\min(conflicts,2).
$$

$$
g_t=clip(\sigma(\ell),p_0,cap),
$$

无直接冲突时 `cap=0.15`，有冲突时 `cap=0.40`。

### 9.3 稀疏惊讶度门控

只对确定且已知回答启用，按惊讶度 3.0、3.5、4.0 分段给出误报先验
0.20、0.35、0.49；直接冲突给 0.49。

### 9.4 历史可靠性门控

令此前 $T$ 个回答中 `UNKNOWN` 或明确 uncertain 的数量为 $u$：

$$
\widehat\rho_t=
\frac{u}{0.55(T+3)},
$$

$$
g_t=\min(0.12,0.30\widehat\rho_t).
$$

无不可靠历史线索时 $g_t=0$；直接冲突时为 0.30。该方法避免用“当前回答与当前
诊断不一致”判断当前回答不可信，从而减轻确认偏差。

对应实现：`src/powerful_medrag/gating.py`、`belief.py`、`gate_analysis.py`。

## 10. 澄清方法与核心创新

### 10.1 PaMis-style 在线惊讶度基线

在纳入当前回答前计算：

$$
s_i=-\ln P(R_i=r_i\mid H_{i-1}).
$$

`pamis_style_s30` 在回答确定、已知且 $s_i\ge3.0$ 时复问同一证据。

必须注意：这是“检测异常→受控澄清”的结构化机制代理，不包含 PaMis 的实体图、
结构熵检测器或自然语言问题生成器，不能称为完整 PaMis 复现。

### 10.2 澄清回答合并规则

设原回答为 $r$，复问回答为 $r'$：

$$
resolve(r,r')=
\begin{cases}
r,&r=r'\ne UNKNOWN,\\
UNKNOWN,&r\ne r'\ \text{或任一为 UNKNOWN}.
\end{cases}
$$

一致时证据只纳入一次，避免把同一信息双计；冲突时弃权。复问仍经过同一个患者噪声
通道，没有 oracle 可靠性加成，并严格消耗一轮。

### 10.3 Leave-one-out 疾病后验

对历史回答 $i$，删除它并用其他报告得到：

$$
b_{-i}(d)=P(D=d\mid H\setminus\{R_i\}).
$$

当每个特征只出现一次时，朴素贝叶斯结构允许精确快速计算：

$$
b_{-i}(d)\propto \frac{b(d)}{L_{d,i}(r_i)}.
$$

若特征重复或与初始观察重合，代码自动回退到完整历史重放。测试和探针证明快速公式
与逐回答重放逐病例完全一致。

### 10.4 Leave-one-out 错误概率

先计算删去回答 $i$ 后的预测真实状态分布：

$$
p_{-i}(z)=\sum_d b_{-i}(d)P(Z_i=z\mid D=d).
$$

固定人口级检测通道下，报告对应的真实状态后验为：

$$
P(Z_i=z\mid R_i=r_i,H_{-i})
=
\frac{p_{-i}(z)P(R_i=r_i\mid Z_i=z,C_i)}
{\sum_{z'}p_{-i}(z')P(R_i=r_i\mid Z_i=z',C_i)}.
$$

于是报告错误概率为：

$$
p_{err,i}=1-P(Z_i=r_i\mid R_i=r_i,H_{-i}).
$$

检测器始终使用固定的默认报告通道，**不能读取实验注入的真实噪声率、真实状态或真实
报告模式**，以防标签泄漏。

### 10.5 诊断影响

用完整疾病后验和 leave-one-out 后验的总变差距离：

$$
\Delta_i=TV(b,b_{-i})=
\frac12\sum_d|b(d)-b_{-i}(d)|.
$$

### 10.6 三个回溯策略

纯风险法：

$$
S_i^{risk}=p_{err,i}.
$$

配置 `retro_risk_r50_b1`：阈值 0.50，最多一次回溯。

主方法 Risk × influence：

$$
S_i^{utility}=p_{err,i}\Delta_i.
$$

配置 `retro_utility_u050_b1`：阈值 0.05，最多一次回溯。

混合法 `hybrid_s30_u025_b2`：先用在线 $s\ge3.0$ 处理极端即时异常，再用
$p_{err,i}\Delta_i\ge0.025$ 回溯，最多两次回溯；已经澄清的回答不会重复入选。

### 10.7 预算和停止

回溯审计发生在以下任一条件：

- 当前最大疾病后验达到停止阈值 $\tau$；
- 剩余总轮数已不多于剩余回溯预算。

若澄清将回答改成 `UNKNOWN` 并使置信度跌破 $\tau$，系统继续按 EIG 提问，直到
重新达到阈值、无问题可问或总计 15 轮用尽。因此结果不会因“推翻证据后仍强行停止”
而虚高。

对应实现：`src/powerful_medrag/clarification.py`。

## 11. 评测与统计公式

### 11.1 Top-1 准确率与轮数

$$
Accuracy=\frac1N\sum_{n=1}^N
\mathbb 1[\arg\max_db_n(d)=d_n^*].
$$

平均轮数包含普通问题和澄清问题；初始证据不计入追加轮数。

### 11.2 Brier 分数

$$
Brier_n=\sum_d\left(b_n(d)-\mathbb 1[d=d_n^*]\right)^2.
$$

越低越好，用于判断“少问”是否只是错误的过早自信。

### 11.3 误报检测指标

当前模拟中的 harmful report 定义为：真实模式为 `MISREPORTED`、回答已知且回答值不等于
真实状态。

$$
Precision=\frac{\#\ detected\ harmful}
{\#\ detected\ harmful+\#\ false\ clarifications},
$$

$$
Recall=\frac{\#\ detected\ harmful}{\#\ harmful},
$$

$$
Mitigation=\frac{\#\ detected\ harmful\ resolved\ to\ nonwrong}
{\#\ detected\ harmful}.
$$

澄清率为：

$$
ClarificationRate=\frac{\#\ clarifications}{\#\ primary\ questions}.
$$

### 11.4 Wilson 二项比例区间

令 $\hat p=x/n$，置信系数 $z=1.96$：

$$
center=\frac{\hat p+z^2/(2n)}{1+z^2/n},
$$

$$
margin=\frac{z}{1+z^2/n}
\sqrt{\frac{\hat p(1-\hat p)}n+\frac{z^2}{4n^2}}.
$$

区间为 `[center-margin, center+margin]`。

### 11.5 平均轮数标准误

$$
SE(\bar Q)=\frac{SD(Q)}{\sqrt n}.
$$

### 11.6 病例聚类 Bootstrap

多种子实验先在同一病例内平均策略差值，再以病例为抽样单位有放回抽样，默认 2,000
或 5,000 次；报告 bootstrap 均值分布的 2.5% 与 97.5% 分位数。这样不会把同一病例
的多个随机种子错误地当成独立样本。

### 11.7 精确 McNemar 检验

设策略 A 独对的病例数为 $b$，策略 B 独对为 $c$，$n=b+c$：

$$
p=\min\left(1,
2\sum_{k=0}^{\min(b,c)}{n\choose k}2^{-n}\right).
$$

多种子分别计算，正式报告取最大 p 值作为保守汇总。

### 11.8 AUROC

门控诊断使用成对定义：

$$
AUROC=P(s^+>s^-)+\frac12P(s^+=s^-).
$$

`gate_analysis.py` 还计算 average precision 和固定激活阈值下的 precision/recall。

### 11.9 同轮数曲线插值

候选平均轮数 $q$ 位于参考曲线相邻点 $(q_l,a_l)$、$(q_h,a_h)$ 之间时：

$$
a_{ref}(q)=a_l+
\frac{q-q_l}{q_h-q_l}(a_h-a_l).
$$

只做区间内插值，不外推。报告
$a_{ref}(q)-a_{candidate}(q)$。该比较是描述性的，不等同于显著性检验。

若参考点满足 $q_{ref}\le q_{cand}$、$a_{ref}\ge a_{cand}$，且至少一个严格不等，
则参考点离散 Pareto 支配候选点。

对应实现：`benchmark.py`、`ablation.py`、`evaluation.py`、`curve_analysis.py`。

## 12. 已完成实验与结论

### 12.1 EIG 对 MedRAG-RDC：正式测试集

DDXPlus train 1,025,602 例；独立 test 按疾病最多 100 例，共 4,836 个病例；三个回答
种子；后验阈值 0.85：

| 噪声 | EIG 准确率/轮数 | RDC 准确率/轮数 |
|---:|---:|---:|
| 0% | 94.33% / 6.68 | 67.37% / 11.78 |
| 10% | 87.94% / 7.35 | 62.79% / 11.95 |
| 20% | 80.15% / 8.02 | 57.35% / 12.10 |
| 30% | 71.10% / 8.56 | 51.52% / 12.22 |

结论只应写为“EIG 优于本项目的 MedRAG-RDC 结构化公式适配”，不能写成完整复现并
击败原始 MedRAG。

### 12.2 报告层正式消融

`unknown_as_negative` 最差，证明不知道/不确定不能视为阴性；固定完整两层模型改善
准确率和 Brier，但通常比 `no_misreport` 多问约 1.2–1.7 轮。由此推动动态门控。

### 12.3 历史可靠性门控：正式测试集

4,836 病例、三个种子、阈值 0.85：

| 噪声 | no-misreport | 固定完整模型 | 历史门控 |
|---:|---:|---:|---:|
| 0% | 93.82% / 5.50 | 94.33% / 6.68 | 93.86% / 5.52 |
| 10% | 86.93% / 5.97 | 87.94% / 7.35 | 87.66% / 6.28 |
| 20% | 78.55% / 6.40 | 80.15% / 8.02 | 80.51% / 7.04 |
| 30% | 69.42% / 6.83 | 71.10% / 8.56 | 73.03% / 7.82 |

历史门控在高噪声下改善 Pareto 位置，但当前惊讶度门控会把纠正错误诊断的真证据误当
异常，存在确认偏差。

### 12.4 PaMis-style 在线澄清：验证集

980 例、单种子、阈值 0.85。30% 噪声时 precision 44.8%、recall 7.4%、准确率
70.71%、6.92 轮。检测到后缓解效果很好，但漏掉绝大多数有害误报，促成回溯方案。

### 12.5 三种回溯方法：验证集

980 例、49 病每病 20 例、单种子、阈值 0.85：

| 方法 | 30% 噪声准确率 | 轮数 | precision | recall |
|---|---:|---:|---:|---:|
| 在线 s30 | 70.71% | 6.92 | 44.8% | 7.4% |
| 纯风险 r50-b1 | 70.71% | 6.84 | 60.6% | 16.6% |
| Risk × influence u050-b1 | 74.49% | 7.68 | 30.9% | 24.6% |
| Hybrid s30-u025-b2 | 76.33% | 8.65 | 29.1% | 53.3% |

相对在线 s30，主方法在 20%/30% 噪声下提高 2.55/3.78 个百分点，病例 bootstrap
区间均不含 0，精确 McNemar p 为 0.00126/0.000110。

同轮数下，主方法在中高噪声优于在线 s30 和历史门控；低噪声时因误触发而较弱。
混合法原始准确率和召回最高，但问题效率不如主方法。

### 12.6 当前方法选择

- 最少额外轮数/高检测精度：`retro_risk_r50_b1`；
- 最佳中高噪声准确率—轮数折中：`retro_utility_u050_b1`；
- 最高召回、问题成本次要：`hybrid_s30_u025_b2`。

论文主方法暂定 `retro_utility_u050_b1`，另外两种作为机制消融。

## 13. 数据划分与泄漏边界

- 疾病状态模型只从 `release_train_patients.zip` 估计。
- 回溯阈值和预算只在 `release_validate_patients.zip` 的平衡 980 例上选择。
- 回溯三方法尚未运行 `release_test_patients.zip`。
- 但是 test split 已在更早阶段用于 EIG/RDC、报告层消融和历史门控。因此从整个项目
  生命周期看，它已不是完全未观察的最终测试集。若要形成最严格的论文主结果，建议
  使用新的外部数据集、额外留出集或预注册的 nested evaluation。
- 模拟器的真实 $Z$、真实 $M$ 和注入噪声率只用于生成回答与事后评测，不进入回溯
  检测分数。

## 14. 仓库文件逐项说明

### 14.1 根目录

| 文件 | 内容 |
|---|---|
| `README.md` | 项目简介、主要命令、当前实验边界的较短版本 |
| `HANDOFF.md` | 本交接文档；研究设计、全部核心公式、文件说明和复现步骤 |
| `pyproject.toml` | Python 包元数据；Python ≥3.10；可选实验依赖 numpy、matplotlib；CLI 入口 |
| `.gitignore` | GitHub 上传规则；排除原始患者数据、训练模型、缓存和大型逐病例产物 |

### 14.2 `src/powerful_medrag/`

| 文件 | 具体职责 |
|---|---|
| `__init__.py` | 导出公共 API：schema、通道、信念跟踪器、模型和选问器 |
| `__main__.py` | 支持 `python -m powerful_medrag ...`，转发到 CLI |
| `schema.py` | `FeatureKey`、`VariableSpec`、`ClinicalCase`、`Observation`、确定性枚举与校验 |
| `data.py` | 通用 schema JSON 与病例 JSONL 的读写、字典转换 |
| `demo_data.py` | 三种玩具疾病、变量定义和可重复合成病例，供演示与单元测试 |
| `ddxplus.py` | DDXPlus evidence/condition 解析、ZIP/CSV 流式读取、原子变量展开、平衡蓄水池抽样、训练统计 |
| `estimation.py` | 疾病先验与 `P(Z|D)` 的 Laplace/层级估计、充分统计、模型序列化 |
| `channel.py` | 四种隐藏报告模式、混淆率、cue prior、回答概率、模式后验与通道消融 |
| `belief.py` | 疾病后验、同变量共享真实状态、冲突识别、动态误报先验注入、更新诊断信息 |
| `questioning.py` | 参考 EIG、NumPy EIG、固定顺序、静态流行率和 MedRAG-RDC 选问器；前置问题门控 |
| `simulator.py` | 结构化患者、噪声 profile、特征级确定性随机回答、停止规则和端到端问诊 |
| `evaluation.py` | 合成队列准确率、轮数、Brier、unknown 率和误报识别 AUROC |
| `benchmark.py` | DDXPlus 配对曲线、逐病例结果、跨种子平均、Wilson/Bootstrap/McNemar、绘图 |
| `ablation.py` | 报告层消融、动态门控运行、病例聚类对照统计和四面板 Pareto 图 |
| `gating.py` | 启发式惊讶度、稀疏惊讶度、历史可靠性三个误报门控 |
| `gate_analysis.py` | 收集逐轮门控事件，计算 AUROC、AP、激活 precision/recall，保存诊断表 |
| `clarification.py` | PaMis-style 在线澄清、leave-one-out 回溯、三种策略、预算/重放和检测诊断 |
| `curve_analysis.py` | 加载曲线、同平均轮数线性插值、离散 Pareto 判断、比较图和独立 CLI |
| `cli.py` | `fit`、`fit-ddxplus`、`inspect`、`demo`、`benchmark-ddxplus`、`ablate-ddxplus`、`analyze-gate-ddxplus`、`clarify-ddxplus` |

### 14.3 `tests/`

| 文件 | 覆盖内容 |
|---|---|
| `test_estimation.py` | 缺失不等于阴性、Beta–Bernoulli/Laplace 公式、模型序列化 |
| `test_belief.py` | unknown 无信息、阳性更新、上下文差异、直接冲突、共享真实状态、误报后验 |
| `test_questioning.py` | 判别性、回答分布归一化、前置问题、NumPy 等价、RDC 稀有特征排序 |
| `test_simulator.py` | 总问题预算、不重复选问、特征级噪声与选问顺序无关 |
| `test_ddxplus.py` | 二元、分类和多选 DDXPlus 证据到原子变量的转换 |
| `test_benchmark.py` | 多种子病例配对统计与聚类汇总 |
| `test_ablation.py` | 通道消融、三种门控、门控诊断指标和小型端到端消融 |
| `test_clarification.py` | 在线触发、合并规则、总轮数、回溯评分非突变和回溯预算 |
| `test_curve_analysis.py` | 同预算插值、禁止外推、离散 Pareto 支配 |

当前共 35 项测试。

### 14.4 `data/ddxplus/`

| 文件 | 内容与 Git 策略 |
|---|---|
| `release_train_patients.zip` | DDXPlus 训练患者；134 MB；本地放置，不提交 |
| `release_validate_patients.zip` | DDXPlus 验证患者；18 MB；本地放置，不提交 |
| `release_test_patients.zip` | DDXPlus 测试患者；18 MB；本地放置，不提交 |
| `release_evidences.json` | 证据元数据、值域和问题文本；约 0.11 MB；提交 |
| `release_conditions.json` | 疾病—证据关联；约 0.02 MB；提交 |
| `model-full.json` | 从完整训练集拟合的 12 MB 模型；可再生成，不提交 |

### 14.5 `references/`

| 文件 | 内容 |
|---|---|
| `medrag/knowledge_graph_ddxplus.xlsx` | MedRAG 公式适配用的小型 DDXPlus 疾病—证据知识图谱 |

### 14.6 `artifacts/`

命名中 `n20/n100` 表示每病目标病例数，`s3` 表示三个回答噪声种子，`probe` 表示仅用于
开发检查的小样本。主要目录：

| 目录 | 内容 |
|---|---|
| `ddxplus-pilot-n20/` | 早期独立测试集 EIG vs 静态流行率 pilot |
| `ddxplus-n100-s3/` | EIG vs MedRAG-RDC 三种子正式测试 |
| `ddxplus-ablation-n100-s3/` | 四种报告层推断消融正式测试 |
| `adaptive-gate-diagnostic-probe/` | 启发式门控逐轮小样本诊断 |
| `adaptive-gate-diagnostic-validation-n20/` | 58,800 逐轮事件的验证集门控诊断 |
| `adaptive-gate-probe/`、`adaptive-gate-validation-n20/` | 启发式门控端到端开发结果 |
| `adaptive-sparse-validation-n20/` | 稀疏惊讶度门控验证结果 |
| `adaptive-history-validation-n20/` | 历史可靠性门控验证和基线比较 |
| `adaptive-history-test-n100-s3/` | 历史可靠性门控三种子正式测试 |
| `pamis-style-probe/` | 在线澄清阈值小样本探针 |
| `pamis-style-validation-n20/` | 在线阈值 2.5/3.0/3.5/4.0 选择 |
| `pamis-style-s30-validation-curve-n20/` | 冻结 s30 的六阈值验证曲线与报告 |
| `retrospective-probe-n1/` | 原始逐历史重放回溯探针 |
| `retrospective-probe-fast-n1/` | 快速 leave-one-out 等价性探针 |
| `retrospective-validation-n20/` | 回溯阈值/预算筛选与 0.85 配对统计 |
| `retrospective-u050-validation-curve-n20/` | 冻结主方法的完整曲线与报告 |
| `remaining-retro-methods-validation-curve-n20/` | 纯风险、混合与主方法三方法完整比较 |
| `ablation-probe*`、`probe-*` | 并行执行、种子一致性、RDC 和流程开发检查；不是主结果 |

目录内通用文件：

| 文件名模式 | 内容 |
|---|---|
| `REPORT.md` | 可阅读的实验设置、结果、边界和结论；提交 |
| `*_curve_results.csv` | 策略×噪声×阈值汇总：准确率、CI、轮数、SE、Brier；提交 |
| `*_comparisons*.csv` | 病例配对差值、Bootstrap CI、McNemar p；提交 |
| `*_diagnostics.csv` | 门控或澄清的 precision/recall/mitigation 汇总；提交 |
| `*.png` | 准确率—轮数曲线和比较图；提交 |
| `*_case_outcomes.csv` | 大体积逐病例原始结果；可复现，默认忽略 |
| `ddxplus_gate_events.csv` | 大体积逐轮门控事件；可复现，默认忽略 |

### 14.7 `tmp/`

包含 MedRAG/相关论文 PDF、文本抽取、逐页渲染图与临时表格检查脚本，仅为研究过程
缓存，不属于可发布源码，整个目录默认忽略。

## 15. 环境安装与测试

PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[experiment]"
$env:PYTHONPATH='src'
python -m unittest discover -s tests
```

预期：35 项测试通过。

## 16. 从原始数据复现模型

先由同事按 DDXPlus 的授权/分发方式取得三份患者 ZIP，放入 `data/ddxplus/`。然后：

```powershell
$env:PYTHONPATH='src'
python -m powerful_medrag.cli fit-ddxplus `
  --patients data/ddxplus/release_train_patients.zip `
  --evidences data/ddxplus/release_evidences.json `
  --output data/ddxplus/model-full.json
```

## 17. 复现当前回溯主实验

### 17.1 阈值/预算筛选（验证集）

```powershell
$env:PYTHONPATH='src'
python -m powerful_medrag.cli clarify-ddxplus `
  --model data/ddxplus/model-full.json `
  --patients data/ddxplus/release_validate_patients.zip `
  --evidences data/ddxplus/release_evidences.json `
  --output-dir artifacts/retrospective-validation-n20 `
  --cases-per-disease 20 `
  --max-questions 15 `
  --noise-rates 0 0.1 0.2 0.3 `
  --thresholds 0.85 `
  --variants pamis_style_s30 retro_risk_r50_b1 `
             retro_utility_u025_b1 retro_utility_u050_b1 `
             retro_utility_u025_b2 hybrid_s30_u025_b2 `
  --seed 2026 `
  --sample-seed 2026 `
  --workers 12 `
  --executor process
```

### 17.2 冻结主方法六阈值曲线

```powershell
python -m powerful_medrag.cli clarify-ddxplus `
  --model data/ddxplus/model-full.json `
  --patients data/ddxplus/release_validate_patients.zip `
  --evidences data/ddxplus/release_evidences.json `
  --output-dir artifacts/retrospective-u050-validation-curve-n20 `
  --cases-per-disease 20 `
  --max-questions 15 `
  --noise-rates 0 0.1 0.2 0.3 `
  --thresholds 0.60 0.70 0.80 0.85 0.90 0.95 `
  --variants retro_utility_u050_b1 `
  --seed 2026 `
  --sample-seed 2026 `
  --workers 12 `
  --executor process
```

### 17.3 两个补充方法六阈值曲线

```powershell
python -m powerful_medrag.cli clarify-ddxplus `
  --model data/ddxplus/model-full.json `
  --patients data/ddxplus/release_validate_patients.zip `
  --evidences data/ddxplus/release_evidences.json `
  --output-dir artifacts/remaining-retro-methods-validation-curve-n20 `
  --cases-per-disease 20 `
  --max-questions 15 `
  --noise-rates 0 0.1 0.2 0.3 `
  --thresholds 0.60 0.70 0.80 0.85 0.90 0.95 `
  --variants retro_risk_r50_b1 hybrid_s30_u025_b2 `
  --seed 2026 `
  --sample-seed 2026 `
  --workers 12 `
  --executor process
```

## 18. 当前未完成事项与建议顺序

1. **历史可靠性启动回溯**：低噪声时关闭回溯；出现足够 unknown/uncertain 历史线索
   后启用 `Risk × influence`。这是当前最直接的下一创新。
2. **冻结后外部评测**：不要继续用现有验证集微调；优先寻找新的外部数据或重新划分
   nested validation/test。
3. **真实 PaMis 对照**：当前只是 PaMis-style proxy。要作强主张，应实现实体图结构熵，
   或迁移到带对话实体图/自然语言 utterance 的数据集。
4. **自然语言桥接**：用 LLM/分类器把自由文本映射为 `Observation(value, certainty)`，
   单独报告解析误差。
5. **报告通道估计**：默认通道是可解释初值；真实应用必须用带复核标签的问诊数据估计。
6. **安全约束**：本阶段按决定未实现不可牺牲的红旗症状/急诊转介约束，不能把当前原型
   当作临床部署系统。
7. **多重比较**：正式论文应预注册主终点，对大量消融结果作多重检验校正或明确标为探索性。

## 19. 推荐论文表述与禁止过度主张

可以表述：

> 在 DDXPlus 结构化模拟问诊中，leave-one-out 报告错误风险与诊断影响的乘积可以在
> 有限澄清预算下识别值得复核的历史回答，并在中高回答噪声条件下改善准确率—问诊轮数
> Pareto 前沿。

不应表述：

- “已经完整复现并击败 MedRAG/PaMis”；
- “已经在真实患者上证明临床有效”；
- “0% 噪声代表绝对没有回答错误”（CERTAIN 模式仍有 1% wrong、1% unknown）；
- “混合法一定最好”（它原始准确率高，但同轮数效率不如主方法）；
- “测试集完全未观察”（上游正式基线已使用 test split）；
- “系统具备临床安全保证”（当前未实现安全约束）。

## 20. GitHub 提交建议

首次提交前：

```powershell
git init
git add .
git status
```

重点确认 `git status` 中没有：

- `release_*_patients.zip`；
- `model-full.json`；
- `tmp/`；
- `*_case_outcomes.csv`；
- `ddxplus_gate_events.csv`；
- `.env` 或任何密钥。

若确实需要发布逐病例数据，先确认 DDXPlus 的许可与隐私/分发要求，再使用 Git LFS、
release asset 或独立数据仓库；不要直接强推入普通 Git 历史。
