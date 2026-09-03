# Phase 8B — 联合报告通道模型规格 (Model Specification)

## 0. 问题

Phase 8A 之前，四个量来自**不同的**概率模型：

1. 疾病后验 `b_t(d) = P(D=d | H_t)` — `BeliefTracker` 的序贯更新；
2. 逐回答可靠度 `p_i^mode = P(E_i=MISREPORTED | H_t)` — 独立的后验；
3. 逐回答错误率 `p_i^wrong = P(Z_i ≠ Y_i | H_t)` — 独立的状态后验；
4. VerifyOld 结果预测 `P(Y_i' | H_t, VerifyOld(i))` — `SurprisalClarificationProtocol` 的
   独立再问通道。

且两次回答矛盾时，旧路径直接压缩为 `UNKNOWN`（`resolve`）。Phase 8B 用一个共享的
联合模型 `P(D, Z, E, Y, Y')` 统一这四个量，并废除 UNKNOWN 压缩。

## 1. 共享生成模型

对一个被问两次的特征 `i`（首答 + 一次 VerifyOld 再问）：

```
P(y_i, y_i' | z_i, v)
    = Σ_{e_i, e_i'}  P(e_i | cue) · P(y_i | z_i, e_i)
                     · T_v(e_i' | e_i) · P(y_i' | z_i, e_i')
```

其中

* `z_i` — 该特征的真实临床状态（latent）；
* `e_i` — 首答的潜在报告模式 `ReportMode ∈ {CERTAIN, UNCERTAIN, UNKNOWN, MISREPORTED}`；
* `e_i'` — 再问的潜在报告模式；
* `v` — 验证类型（本阶段仅实现 `REPEAT`，其余为可扩展接口）；
* `cue` — 首答的可观测确定性提示（`CertaintyCue`）。

## 2. 模式转移 `T_v`

```
T_v(e' | e) = rho · 1[e' = e] + (1 - rho) · P_reask(e')
```

* `rho = repeat_mode_persistence` ∈ [0,1]，**可配置**。
* `rho = 0` → 再问与首答独立（联合模型退化为两个单答边际的乘积）。
* `rho = 1` → 报告模式完全持续（再问模式 = 首答模式）。

**关键恒等式**：`Σ_e P(e) T_v(e' | e) = P(e')`，即再问的**边际**模式分布等于首答的
边际模式分布，**与 rho 无关**。因此「独立再问是联合模型的特例」，且单答边际与
`AnswerChannel.marginal_probability` 精确相等。

## 3. 单特征似然因子（一次证据 = 一个因子）

```
L_i(d) = Σ_z P(z | d) P(y_i, y_i' | z, v)     (已核验)
L_i(d) = Σ_z P(z | d) P(y_i | z)               (仅单答)
```

疾病后验 `b(d) ∝ P(d) · ∏_i L_i(d)`。一个特征无论被问几次，都只贡献**一个**因子，
因此「两次相同回答」不会被重复计数，矛盾回答也不会被压缩成 UNKNOWN。

## 4. 四个量来自同一模型

| 量 | 定义 | 实现 |
| --- | --- | --- |
| `b_t(d)` | `P(D=d \| H_t)` | `JointReliabilityBeliefTracker.disease_belief()` |
| `p_i^mode` | `P(E_i=MISREPORTED \| H_t)` | `p_mode_misreported(key)`，经 `C_i(d,m)` 分解 |
| `p_i^wrong` | `P(Z_i ≠ Y_i \| H_t)` | `1 - P(Z_i = y_i \| H_t)`（`y_i=UNKNOWN` 时恒为 1） |
| `P(y' \| H_t, v)` | VerifyOld 再问预测 | `reask_predictive(key)` |

模式后验分解：`P(E_i=m | H_t) ∝ Σ_d P(d) · C_i(d,m) · ∏_{j≠i} L_j(d)`，其中
`C_i(d,m) = Σ_z P(z|d) P(m) P(y_i|z,m) · [reask 项]`，`Σ_m C_i(d,m) = L_i(d)`。

再问预测：`P(y' | H_t, Y_i, v) ∝ Σ_{d,z,e} P(d) P(z|d) P(e) P(y_i|z,e)
∏_{j≠i}L_j(d) · Σ_{e'} T_v(e'|e) P(y'|z,e')`，与疾病后验共享同一归一化常数。

## 5. 模式后验的独立性

`single_probability(y|z,cue) ≡ AnswerChannel.marginal_probability(y,z,cue)`。联合通道
只读取冻结的 `AnswerChannel` 混淆率与 cue 先验，绝不读取真实疾病、latent 状态、
噪声标签或真实错误性（红线）。

## 6. 决策层

`JointChannelBrierAuditPolicy` 复用 Phase 8A **修正后**的审计逻辑（Prompt #18）：

* AskNew：`V_new(j) = R_B(b_t) - E_y[R_B(b_{t+1})] - C_new,j`；
* VerifyOld：`V_verify(i) = R_B(b_t) - E_{y'}[R_B(b_{t+1})] - C_verify`（再问预测用 `reask_predictive`）；
* Stop 可靠度：`max_i p_i^mode ≤ suspicious_report_threshold`；
* 总收益 `G_t^verify > τ_V` 仅**阻止** Stop（严格 `>`），不再强制 VerifyOld；
* AskNew/VerifyOld 的净 Brier 比较带全局优势边际 `τ_A`。

`τ_V = τ_A = 0.03`（冻结，同 Phase 8A 修正）。
