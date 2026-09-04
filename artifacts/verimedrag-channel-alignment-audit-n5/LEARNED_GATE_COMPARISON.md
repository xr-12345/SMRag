# Learned Gate 离线对照（Phase 8D §六）

轻量对照：确认 learned gate 的训练标签、特征口径与联合后验的关系，给出
4 信号对照表，并落结论——**标签不同、不融合、不可直接比校准**。

## 1. 训练标签确认

`evaluate_gate.py:55` `PRIMARY = "harmful_misreport"`（= wrong_report ∧ latent_misreport，
即"说错"且"蓄意误报"）。learned gate 的 logistic 截距 −3.4274 对应基率约 0.031，
与 harmful_misreport 的低流行率一致（wrong_report 基率约 0.085，明显更高）。

**因此 learned gate 的标签 ≠ p_wrong 的标签。** p_wrong 标定的是 `wrong_report`
（内容错配，Z≠Y），learned gate 标定的是 `harmful_misreport`（错配 ∧ 误报）。两者
不可直接比 ECE/Brier。

## 2. 关键系数证据

learned gate 对 `is_unknown` 的系数为 **−0.293（负）**，即模型学到"回答 UNKNOWN"
会**降低** harmful_misreport 概率。这与 Step 4 发现的根因一致：UNKNOWN（"我不知道"）
语义上 ≠ "说错/误报"。而生成式 p_wrong 对 UNKNOWN 值恒返 1.0，正是把"未知"混入
"说错"的失配来源（§三、Step 4 的 `value=unknown` 层 ECE=1.0）。

## 3. 特征口径限制（joint 语境不可得）

learned gate 的 9 个特征定义在**非 joint 的 `BeliefTracker`（VerifyOld 语境）**上：

- `surprisal` ← `HeuristicMisreportGate.predictive_probability(tracker, observation)`
- `diagnostic_impact` ← `tracker.observation_likelihoods(observation, base_mode_prior)`
- `history_unreliable_fraction` / `direct_conflicts` ← `tracker.history`（观测列表）

而 Phase 8D 用的是 `JointReliabilityBeliefTracker`，接口不同：有 `disease_belief()`
与 `self.memory`（key→ReportBundle），**无** `tracker.history`、`observation_likelihoods`
与 `HeuristicMisreportGate` 依赖。因此 learned gate 的 3/9 个特征（surprisal、
diagnostic_impact、history 序列）在 joint 语境**不能直接原样计算**，需适配或注明局限。

结论：learned gate 分数**不填入** joint 后验；本阶段只做离线对照，不训练融合模型。

## 4. 四信号对照表（各对其自身标签）

| 信号 | 目标标签 | AUROC | AUPRC | ECE | 数据来源 |
|---|---|---|---|---|---|
| 生成式 p_wrong（legacy） | wrong_report（binary first） | 0.874 | 0.498 | 0.0395 | Phase 8D Step 4 |
| 生成式 p_wrong（legacy） | wrong_report（全 first 含 UNKNOWN） | 0.762 | 0.158 | 0.1555 | Phase 8D（复现 8C） |
| learned gate | harmful_misreport | 0.844 | 0.290 | 0.0190 | Prompt #7（pooled_noisy） |
| calibrated surprisal | harmful_misreport | 0.778 | 0.302 | 0.0245 | Prompt #7（pooled_noisy） |
| fixed prior（训练基率） | harmful_misreport | 0.500 | 0.045 | 0.0132 | Prompt #7（pooled_noisy） |

说明：
- learned gate 在其自身标签（harmful_misreport）上已**良好校准**（ECE 0.019），
  且判别力高（AUROC 0.844）。它的 ECE 低，部分因为它是**监督拟合**、且标签是
  harmful_misreport（不含 UNKNOWN 混入）。
- p_wrong 在其自身标签（wrong_report）上，当排除 UNKNOWN 混入后 ECE 也仅 0.0395
  （binary）/ 0.0090（cue_conditioned），但**全 first 含 UNKNOWN 的 ECE 0.1555** 是
  UNKNOWN 混入造成的，不是 learned gate 可比的量。
- 两者标签不同、信号不同、语境不同 → **不融合、不直接比校准**。

## 5. 结论

1. learned gate 训练标签确认 = harmful_misreport，≠ p_wrong 的 wrong_report。
2. learned gate 的 `is_unknown` 负系数与 Step 4 的 UNKNOWN 混入根因相互印证。
3. learned gate 特征口径依赖非 joint BeliefTracker，joint 语境不可直接计算（3/9 特征）。
4. 按要求：learned gate **不填入** joint 后验、**不训练**融合模型。
