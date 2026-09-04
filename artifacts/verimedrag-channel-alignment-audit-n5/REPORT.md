# Phase 8D — 报告通道对齐 + p_wrong 校准诊断审计

**问题**：Phase 8C 的 `p_wrong` ECE ≈ 0.156，是否由环境 vs 推断的报告通道失配造成？

**结论（一句话）**：**不是主要由通道失配造成。** 0.156 约 77% 来自 **UNKNOWN 混入**
（`p_wrong` 对"患者说不知道"恒返 1.0，而内容错配标签把它记作 0）；通道先验失配
（MISREPORTED 0.03 vs 0.06/0.09 + cue 机制）是**真实但次要**的成分，且可被
`train_fixed`/`cue_conditioned` 部署修复（内容错配 ECE 0.040→0.011，p_mode 0.043→0.008）。

---

## 1. 复现（sanity）

| 指标 | 本审计（legacy） | Phase 8C | 一致 |
|---|---|---|---|
| p_wrong ECE（全 first，含 UNKNOWN） | **0.155523** | 0.1555 | ✅ |
| n（first-answer 行） | 13171 | 13171 | ✅ |
| p_mode ECE | 0.0433 | 0.0433 | ✅ |

管线与 Phase 8C 逐位对齐（matched 0/0，245 case × 2 noise × 3 seed = 1470 轨迹）。

## 2. 静态通道审计（§二）

5 组件中，**失配集中在组件 2（cue 机制）与组件 1/4（MISREPORTED 先验）**：

| 组件 | env | inference | 失配 |
|---|---|---|---|
| 1 P(E) 先验 | (.80/.70, .08/.12, .06/.09, **.06/.09**) | (.82,.10,.05,**.03**) | MISREPORTED 低估 2–3× |
| 2 P(E\|NONE) | (.08,.29,.58,.04) | (.82,.10,.05,.03) | **KL 1.57/1.72（主导）** |
| 3 首答 P(Y\|Z,E) | 同一 rates | 同一 rates | 无 |
| 4 复问先验 P(e') | MISREPORTED .06/.09 | .03 | 低估 2–3× |

## 3. 离线配对审计（§四/§五）

4 配置：legacy（默认）、oracle（env 真 P(E|cue)，特权）、train_fixed（跨 noise 平均、
无 cue 条件，可部署）、cue_conditioned（跨 noise 平均 + cue 机制，可部署）。

### 3.1 p_wrong 的 ECE 分解（核心）

| 子集 | n | legacy | oracle | train_fixed | cue_conditioned |
|---|---|---|---|---|---|
| 全 first 含 UNKNOWN（=8C 口径） | 13171 | **0.1555** | 0.1304 | 0.1303 | 0.1284 |
| 排除 UNKNOWN（binary+categorical） | 11584 | 0.0398 | 0.0113 | **0.0112** | **0.0090** |
| 仅 binary | 7884 | 0.0395 | 0.0144 | 0.0163 | 0.0119 |
| 仅 UNKNOWN | 1587 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**关键读数**：
- 全口径 0.1555 在**即使 oracle 特权先验**下也只降到 0.130 —— 说明那 0.12（≈UNKNOWN
  层，n=1587=12%，每行 |1.0−0|=1.0）**不是先验能修的**，是 `p_wrong` 对 UNKNOWN
  恒返 1.0 的语义混入（`joint_reliability_belief.py:290-291`）。
- 排除 UNKNOWN 后，legacy 0.0398 被三种先验一致压到 0.009–0.011（**3.5–4.4×**），
  且 **train_fixed（可部署）≈ oracle（特权）**，cue_conditioned 更优（0.0090）。

### 3.2 按值分层（p_wrong，legacy）

| value | n | 错配率 | ECE |
|---|---|---|---|
| present | 2183 | 0.269 | 0.107 |
| absent | 5701 | 0.029 | 0.014 |
| categorical | 3700 | 0.097 | 0.043 |
| **unknown** | 1587 | 0.000（标签） | **1.000** |

unknown 层是"标签=0 vs p_wrong=1.0"的纯混入，ECE 恒 1.0（四配置一致）。present
层错配率最高（0.269），legacy 欠校准最明显（ECE 0.107 → oracle 0.038）。

### 3.3 p_mode 的 ECE

| 子集 | legacy | oracle | train_fixed | cue_conditioned |
|---|---|---|---|---|
| 全 first（n=13171） | 0.0433 | 0.0130 | **0.0079** | 0.0127 |

p_mode（latent_misreport 标签）无 UNKNOWN 混入，先验对齐后 ECE 0.043→0.008–0.013。
**train_fixed 最优**（0.0079），因其 MISREPORTED 先验 0.075 ≈ 池化基率 0.078。

### 3.4 判别力（AUROC/AUPRC）几乎持平

先验对齐以极小判别力代价换大校准收益：

- p_wrong binary：AUROC legacy 0.874 → 0.857–0.863；AUPRC 0.498 → 0.426–0.438。
- p_mode：AUROC legacy 0.820 → 0.812–0.818；AUPRC 0.396 → 0.344–0.382。

校准斜率（p_wrong binary）：legacy slope **0.267**（<<1，系统欠估计）；固定先验后
slope≈0、intercept≈0.87–0.92（近平坦、近正确）。

### 3.5 分层一致性（§五）

- **noise 0.2 vs 0.3**：p_wrong binary legacy ECE 0.027 vs 0.053；固定先验后 0.015–0.021 vs 0.017–0.021（跨 noise 对齐）。
- **certainty**：certain 主导（n=7277，ECE legacy 0.042→0.011–0.016）；uncertain（n=607）错配率高（0.158），固定先验后 ECE 仍偏高（0.026–0.051）。
- **verified 前/后**：复问后错配率高达 0.499（复问即高风险信号），ECE 复问前 0.029→0.015、复问后 0.168→0.041（oracle）。
- **timepoint reask**：样本小（n=381），CI 宽，报告如实标注。
- **seed**：2026/2027 legacy ECE 0.038/0.036 → oracle 0.010/0.019（跨 seed 一致）。

## 4. 根因诊断（回答研究问题）

| 根因 | 贡献（约） | 证据 | 可修复？ |
|---|---|---|---|
| **UNKNOWN 混入**（p_wrong=1.0 vs 标签=0） | ~0.12 / 0.156（77%） | unknown 层 ECE 恒 1.0，四配置一致 | 是，需改 p_wrong（独立修复，非先验） |
| 通道先验失配（MISREPORTED 0.03 vs 0.06/0.09） | ~0.024（binary） | 排除 UNKNOWN 后 ECE 0.040→0.011 | 是，train_fixed/cue_conditioned 部署修复 |
| cue 机制失配（P(E\|NONE) KL 1.57） | 并入上项 | 静态审计组件 2；oracle 略优于 train_fixed | 是，cue_conditioned（0.0090） |
| categorical 多类 | ~0.012 | categorical 层 ECE 0.043→0.012 | 部分，先验对齐改善 |

**结论**：研究问题（"0.156 是否通道失配造成"）→ **否定为主**。通道失配是次要真实
成分且可部署修复；但把 0.156 归因于通道失配是错误的，主因是 UNKNOWN 语义混入。

## 5. learned gate（§六，详见 LEARNED_GATE_COMPARISON.md）

- 训练标签确认 = **harmful_misreport**（≠ wrong_report），intercept −3.427。
- `is_unknown` 系数 **−0.293**：gate 已学到"UNKNOWN ≠ 错配"，与 Step 4 根因互证。
- learned gate 在其自身标签上 ECE 0.019（Prompt #7），但标签不同、语境不同（非 joint
  BeliefTracker，3/9 特征不可得）→ **不融合、不直接比校准**。

## 6. 红线合规

- 只跑 matched 0/0；无 N=20、无策略修改、无在线比较、无 ρ/τ 调参。
- 先验为参数化（跨 noise 平均），未用 val/test 标签拟合。
- 预测路径不读 true_state/noise/latent/mode 链。
- 未 git commit；未覆盖旧 artifacts（新目录 `verimedrag-channel-alignment-audit-n5/`）。
