# Decision-Sensitive Action Value：问题分析与变量说明

本文整理当前 VeriMedRAG 策略价值函数的局限，以及更完整的统一动作价值定义。核心观点是：AskNew、VerifyOld 和 Stop 不应依赖彼此独立的启发式分数，而应统一衡量“执行动作后，临床风险预计能够下降多少”。

## 第一部分：为什么当前价值函数过于简单

目前的策略价值函数偏简单。它适合验证“检索能不能帮助找到可疑回答”，但不适合作为最终核心方法。

Phase 2C 的结果正好暴露了这个问题：系统找错回答很准，触发核验精度达到 61.7%，但核验后诊断基本没有变化。

原因是当前价值函数主要回答：

> 这条回答可能错吗？删除后检索变化大吗？

但真正应该回答：

> 核验这条回答以后，诊断错误、分诊风险或不安全停止的概率能下降多少？

### 1. 当前价值函数

当前 Retrieval Joint Gate 近似为：

$$
G_i=
P_{\mathrm{error}}(i)
\cdot
\Delta_{\mathrm{retrieval}}(i)
-C_{\mathrm{verify}}
$$

其中：

| 变量 | 含义 |
|---|---|
| $i$ | 第 $i$ 条已有患者陈述 |
| $P_{\mathrm{error}}(i)$ | 第 $i$ 条陈述错误的概率 |
| $\Delta_{\mathrm{retrieval}}(i)$ | 删除或干预该陈述后，检索结果的变化程度 |
| $C_{\mathrm{verify}}$ | 核验一次旧回答的成本 |
| $G_i$ | 当前启发式核验分数 |

### 2. 检索变化不等于临床价值

删除一条回答后，Top-k 文档可能发生很大变化：

$$
\Delta_{\mathrm{retrieval}}\gg 0
$$

但如果最终疾病后验几乎没有变化：

$$
P(D\mid H_t)
\approx
P(D\mid H_t\setminus y_i)
$$

那么这条回答虽然影响检索，却不影响诊断。

这正是当前 RAG Gate 找到很多真实误报，但 Top-1 没有明显改善的原因。

### 3. 错误概率与影响力不是简单的线性乘积

当前近似假设：

$$
\text{价值}
\approx
\text{错误概率}\times\text{检索影响}
$$

这个假设存在以下局限：

- 错误概率可能没有被充分校准；
- BM25 排名变化可能只是相似或同义文档之间的替换；
- 同样大小的检索变化，对不同疾病的临床意义不同；
- 红旗症状的小变化，可能比普通症状的大变化更重要；
- 两个信号的数值尺度和交互关系不一定适合简单相乘。

### 4. 当前价值没有充分考虑核验能否纠错

即使患者第一次回答错了，重新问一次也不一定能够得到正确答案。系统还需要估计：

$$
P(y_i^{\mathrm{verify}}=x_i\mid H_t)
$$

其中：

- $x_i$ 是真实临床事实；
- $y_i^{\mathrm{verify}}$ 是核验后得到的新回答。

如果患者一直误解某个医学概念，重复原问题可能继续得到错误答案。因此，当前价值更接近“发现错误的价值”，还没有完整表示“核验成功并改变决策的概率”。

### 5. 当前方法主要是单步决策

当前方法属于 myopic one-step decision：哪个动作眼前分数最高，就选择哪个。

但有些新问题本身信息量不高，却能帮助下一轮提出一个关键问题；有些核验可以解除冲突，使系统提前安全停止。这些后续影响尚未被完整纳入价值。

### 6. 当前问题成本过于统一

当前近似认为：

$$
C_{\mathrm{AskNew}}\approx C_{\mathrm{VerifyOld}}
$$

但真实患者负担具有差异：

- 重复询问同一问题可能引起反感；
- 询问敏感信息的成本更高；
- 一个复合问题可能包含多个原子问题；
- 后期问题通常比问诊开始时的问题带来更高疲劳成本。

### 7. Stop 还不够安全敏感

停止不能只判断：

$$
\max_d P(D=d\mid H_t)>\tau
$$

如果当前诊断置信度为 0.9，但删除一条低可靠度关键证据后下降到 0.5，系统仍不应该停止。

因此，停止还应考虑当前结论是否依赖低可靠度证据，以及在证据扰动下是否保持稳定。

### 8. 一个直观例子

假设存在两条患者回答。

#### 回答 A

- 错误概率为 90%；
- 删除后检索变化很大；
- 但无论是否核验，最终诊断始终是流感。

当前方法可能给出：

$$
0.9\times 0.8=0.72
$$

但真实临床收益接近：

$$
V_{\mathrm{verify}}(A)\approx 0
$$

#### 回答 B

- 错误概率只有 30%；
- 检索变化中等；
- 如果该回答错误，可能把肺栓塞误判成普通胸痛。

虽然启发式乘积只有：

$$
0.3\times 0.4=0.12
$$

但因为潜在误诊成本很高：

$$
V_{\mathrm{verify}}(B)
\gg
V_{\mathrm{verify}}(A)
$$

这说明真正需要优化的是临床决策后果，而不是错误概率或检索变化本身。

---

## 第二部分：统一动作价值及变量说明

### 1. 统一动作价值

更合理的统一价值为：

$$
\boxed{
V(a\mid H_t)
=
R(b_t)
-
\mathbb E_{o\sim P(o\mid b_t,a)}
\left[
R\big(T(b_t,a,o)\big)
\right]
-
C(a)
}
$$

通俗地说：

$$
\text{动作价值}
=
\text{当前临床风险}
-
\text{执行动作后的预期临床风险}
-
\text{动作成本}
$$

价值越大，动作越值得执行。

### 2. 基本变量

| 变量 | 含义 |
|---|---|
| $t$ | 当前问诊轮次 |
| $H_t$ | 截至第 $t$ 轮的全部问诊历史 |
| $a$ | 候选动作，如 AskNew、VerifyOld |
| $D$ | 患者真实疾病 |
| $X$ | 患者真实临床特征集合 |
| $Y$ | 患者实际给出的回答集合 |
| $M$ | 患者报告模式，如确定、不确定、未知、误报、时间混淆 |
| $K$ | 外部医学知识和当前检索结果 |
| $b_t$ | 当前系统对疾病、真实特征和报告模式的联合信念 |
| $o$ | 执行动作后可能得到的新观察 |
| $P(o\mid b_t,a)$ | 执行动作 $a$ 后得到观察 $o$ 的概率 |
| $T(b_t,a,o)$ | 得到观察 $o$ 后的信念更新函数 |
| $R(b_t)$ | 当前信念状态对应的临床风险 |
| $C(a)$ | 执行动作带来的患者负担和计算成本 |
| $V(a\mid H_t)$ | 动作 $a$ 在当前历史下的预期净价值 |

当前联合信念可以写成：

$$
b_t=P(D,X,M\mid H_t,K)
$$

它不仅表示患者可能患什么病，也表示真实症状可能是什么、哪些回答可能不可靠，以及患者可能处于何种报告模式。

## 3. 临床风险的组成

临床风险可以写成：

$$
R(b_t)
=
\lambda_{\mathrm{dx}}R_{\mathrm{dx}}(b_t)
+
\lambda_{\mathrm{cal}}R_{\mathrm{cal}}(b_t)
+
\lambda_{\mathrm{safe}}R_{\mathrm{safe}}(b_t)
$$

| 变量 | 含义 |
|---|---|
| $R_{\mathrm{dx}}$ | 诊断错误风险 |
| $R_{\mathrm{cal}}$ | 概率不校准或不确定性风险 |
| $R_{\mathrm{safe}}$ | 漏掉红旗、欠分诊或过早停止风险 |
| $\lambda_{\mathrm{dx}}$ | 诊断错误风险的权重 |
| $\lambda_{\mathrm{cal}}$ | 校准风险的权重 |
| $\lambda_{\mathrm{safe}}$ | 临床安全风险的权重 |

### 3.1 诊断风险

最简单的诊断风险可以使用：

$$
R_{\mathrm{dx}}(b_t)
=
1-\max_d P(D=d\mid H_t,K)
$$

如果当前最高疾病概率为 0.8，则风险近似为：

$$
1-0.8=0.2
$$

更合理的形式是引入疾病误诊成本矩阵：

$$
R_{\mathrm{dx}}(b_t)
=
\min_{\hat d}
\sum_d
P(D=d\mid H_t,K)
C_{\mathrm{dx}}(d,\hat d)
$$

| 变量 | 含义 |
|---|---|
| $d$ | 可能的真实疾病 |
| $\hat d$ | 系统准备输出的疾病 |
| $C_{\mathrm{dx}}(d,\hat d)$ | 将疾病 $d$ 判断成 $\hat d$ 的临床成本 |

这样，把严重疾病误判为轻症的成本，可以高于两种轻症之间的混淆。

### 3.2 校准风险

如果当前疾病分布为 $b_t(d)$，可以使用期望 Brier 风险：

$$
R_{\mathrm{cal}}(b_t)
=
1-\sum_d b_t(d)^2
$$

分布越分散，风险越高。

### 3.3 安全风险

安全风险可以定义为：

$$
R_{\mathrm{safe}}(b_t)
=
P(\text{漏掉红旗或发生欠分诊}\mid b_t)
$$

当前项目还没有完整的临床成本矩阵和红旗系统，因此第一版可以先使用诊断风险和 Brier 风险，安全风险作为后续扩展。

## 4. AskNew 的价值

假设准备询问一个尚未获得的特征 $j$：

$$
a=AskNew(j)
$$

它的价值为：

$$
V_{\mathrm{new}}(j)
=
R(b_t)
-
\sum_{y\in\mathcal Y_j}
P(y\mid b_t,AskNew(j))
R(b_{t+1}^{j,y})
-
C_{\mathrm{new}}(j)
$$

| 变量 | 含义 |
|---|---|
| $j$ | 尚未询问的临床特征 |
| $\mathcal Y_j$ | 该问题可能得到的回答集合 |
| $y$ | 一个可能的患者回答 |
| $P(y\mid b_t,AskNew(j))$ | 患者给出回答 $y$ 的概率 |
| $b_{t+1}^{j,y}$ | 得到回答 $y$ 后的新信念 |
| $C_{\mathrm{new}}(j)$ | 询问该问题的患者负担 |

回答集合不应只有 yes 和 no，也应考虑：

$$
\mathcal Y_j
=
\{yes,no,unknown,uncertain,misreport\}
$$

因此，该价值考虑了患者回答不可靠的可能性。

## 5. VerifyOld 的价值

假设核验第 $i$ 条旧回答：

$$
a=VerifyOld(i)
$$

其价值为：

$$
V_{\mathrm{verify}}(i)
=
R(b_t)
-
\sum_{y_i'\in\mathcal Y_i}
P(y_i'\mid b_t,VerifyOld(i))
R(b_{t+1}^{i,y_i'})
-
C_{\mathrm{verify}}(i)
$$

| 变量 | 含义 |
|---|---|
| $i$ | 已有患者陈述的编号 |
| $y_i$ | 患者第一次给出的回答 |
| $y_i'$ | 核验后可能得到的新回答 |
| $\mathcal Y_i$ | 核验时可能出现的回答集合 |
| $P(y_i'\mid b_t,VerifyOld(i))$ | 核验后出现回答 $y_i'$ 的概率 |
| $b_{t+1}^{i,y_i'}$ | 用核验结果更新后的新信念 |
| $C_{\mathrm{verify}}(i)$ | 重复询问的患者负担 |

这个式子同时考虑：

1. 原回答是否可能错误；
2. 核验后是否可能获得不同答案；
3. 新答案是否会改变诊断；
4. 核验是否值得付出成本。

## 6. RAG 在价值模型中的位置

当前检索变化可以表示为：

$$
\Delta_{\mathrm{retrieval}}(i)
=
d\left(
\mathcal R(q(H_t)),
\mathcal R(q(H_t\setminus y_i))
\right)
$$

其中：

| 变量 | 含义 |
|---|---|
| $q(H_t)$ | 根据当前问诊历史生成的检索查询 |
| $\mathcal R(q)$ | 查询 $q$ 返回的医学文档集合 |
| $H_t\setminus y_i$ | 删除第 $i$ 条回答后的历史 |
| $d(\cdot,\cdot)$ | 两次检索结果之间的距离，如 Jaccard 或排名差异 |

关键区别是：

$$
\Delta_{\mathrm{retrieval}}(i)
\neq
\Delta_{\mathrm{clinical}}(i)
$$

真正需要预测的是：

$$
\Delta_{\mathrm{clinical}}(i)
=
R(b_t)
-
\mathbb E[R(b_{t+1})\mid VerifyOld(i)]
$$

然后：

$$
V_{\mathrm{verify}}(i)
=
\Delta_{\mathrm{clinical}}(i)
-
C_{\mathrm{verify}}(i)
$$

因此，RAG 的检索变化应该作为预测临床收益的输入特征，而不应直接充当最终临床价值。

可以为每条陈述构造：

$$
\phi_i=
[P_{\mathrm{error}},
\Delta_{\mathrm{document}},
\Delta_{\mathrm{disease\ evidence}},
\Delta_{\mathrm{posterior}},
\Delta_{\mathrm{triage}},
\mathrm{contradiction},
C_i]
$$

再学习：

$$
\hat V_\theta(i)=f_\theta(\phi_i,H_t)
$$

训练标签是真实核验收益：

$$
z_i
=
L_{\mathrm{before}}
-
L_{\mathrm{after\ verification}}
-
C_i
$$

## 7. Stop 的处理

一个直观的停止条件是：

$$
\max\{V_{\mathrm{new}},V_{\mathrm{verify}}\}\leq 0
$$

表示继续询问已经不值得。

同时满足安全约束：

$$
P(\text{unsafe stop}\mid H_t)\leq\epsilon
$$

以及鲁棒性约束：

$$
\max_{i\in\mathcal L}
d(b_t,b_t^{-i})
\leq\delta
$$

| 变量 | 含义 |
|---|---|
| $\mathcal L$ | 当前低可靠度关键陈述集合 |
| $b_t^{-i}$ | 删除或弱化第 $i$ 条陈述后的疾病后验 |
| $d(b_t,b_t^{-i})$ | 干预可疑陈述前后的诊断变化 |
| $\delta$ | 允许的最大诊断变化 |
| $\epsilon$ | 允许的不安全停止概率上限 |

通俗地说：没有值得继续询问的动作，而且删除最可疑回答后诊断仍然稳定，系统才允许停止。

## 8. 数值示例

假设当前临床风险为：

$$
R(b_t)=0.30
$$

### AskNew

询问一个新症状后，预期风险下降到 0.22，问题成本为 0.03：

$$
V_{\mathrm{new}}
=
0.30-0.22-0.03
=
0.05
$$

### VerifyOld

核验一条旧回答后，预期风险下降到 0.18，成本为 0.05：

$$
V_{\mathrm{verify}}
=
0.30-0.18-0.05
=
0.07
$$

因为：

$$
V_{\mathrm{verify}}>V_{\mathrm{new}}
$$

系统应选择 VerifyOld。

如果某条回答错误概率很高，但核验后风险仅从 0.30 降到 0.29：

$$
V_{\mathrm{verify}}
=
0.30-0.29-0.05
=
-0.04
$$

即使它可能是错误回答，也不值得核验。

## 9. 最终学习目标

最终方法应学习：

$$
\boxed{
\hat V_\theta(H_t,K,a)
\approx
R(b_t)
-
\mathbb E[R(b_{t+1})\mid a]
-
C(a)
}
$$

其中：

- 输入是问诊历史、可靠性特征、检索变化和当前疾病后验；
- 训练标签来自训练集上的反事实动作模拟；
- 输出是 AskNew 或 VerifyOld 的临床价值；
- 推理时选择价值最大的动作；
- 训练真值只用于生成标签，不能进入测试决策。

因此，当前简单价值函数可以保留为 heuristic baseline 和机制验证工具，但最终核心方法应统一优化：

$$
\boxed{
\text{Action Value}
=
\text{Expected Clinical Risk Reduction}
-
\text{Interaction Cost}
}
$$
