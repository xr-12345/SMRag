# Phase 8D — GO / NO-GO 判定

## 判定：**NO-GO（对原假设）+ 附带可部署修复 GO**

原问题「p_wrong 的 ECE≈0.156 是否由报告通道失配造成」→ **否定为主**：0.156 约 77%
来自 **UNKNOWN 语义混入**（`p_wrong` 对"患者说不知道"恒返 1.0，标签记 0），**不是**
通道先验失配。通道失配是**真实但次要**的成分，且 **train_fixed/cue_conditioned 已能
部署修复**（内容错配 ECE 0.040→0.011，p_mode 0.043→0.008）。因此：不把"对齐先验"
当作 0.156 的完整解释；但"对齐先验"作为独立可部署修复，其本身 GO。

---

## 逐项判定

### ✅ 1. 复现逐位一致（sanity）
legacy p_wrong ECE = **0.155523**（n=13171），与 Phase 8C 0.1555 完全一致；p_mode ECE
0.0433 亦一致。管线对齐无误。

### ✅ 2. 静态通道审计定位失配（§二）
5 组件中失配集中在 **组件 2（cue 机制，P(E|NONE) KL 1.57/1.72）** 与 **组件 1/4
（MISREPORTED 先验 0.03 vs 0.06/0.09）**；组件 3（首答通道）对齐。失配方向与幅度已量化。

### ❌ 3. 原假设（"通道失配解释 0.156"）→ 否定为主
全口径 ECE 0.1555 在**即使 oracle 特权先验**下只降到 0.130。那 0.12 是不可约的
UNKNOWN 层（n=1587=12%，每行 |1.0−0|=1.0，四配置 ECE 恒 1.0）。通道失配只能解释
余下约 0.04。

### ✅ 4. 通道失配确为次要真实成分，且可部署修复
排除 UNKNOWN 后（binary+categorical，n=11584）：legacy ECE 0.0398 →
oracle 0.0113 / **train_fixed 0.0112** / cue_conditioned 0.0090（3.5–4.4×）。
**train_fixed（可部署，跨 noise 平均，无 cue 条件）≈ oracle（特权）**。

### ✅ 5. p_mode 校准同步修复
p_mode ECE 0.0433 → train_fixed **0.0079**（最优）/ oracle 0.0130 / cue_conditioned 0.0127。
无 UNKNOWN 混入，先验对齐即达标。

### ⚠️ 6. UNKNOWN 混入需独立代码修复（非先验可及）
`joint_reliability_belief.py:290-291` `p_wrong` 对 `value==UNKNOWN` 恒返 1.0。这是
语义 bug（"未知"≠"说错"），需单独修复（UNKNOWN 单独报告，或 p_wrong 对 UNKNOWN
返回非退化值），不在本次先验对齐审计范围内。

### ✅ 7. learned gate 标签与融合边界确认（§六）
训练标签 = harmful_misreport（≠ wrong_report）；`is_unknown` 系数 −0.293（已学到
"未知≠错配"）；3/9 特征在 joint 语境不可得。learned gate **不填入** joint 后验、
**不训练**融合模型。

### ✅ 8. 无泄漏、无红线违反
预测路径不读 true_state/noise/latent/mode 链；先验为参数化（跨 noise 平均），未用
val/test 标签拟合；无 N=20、无策略修改、无在线比较；未 commit、未覆盖旧 artifacts。

---

## 结论与建议

1. **归因**：p_wrong ECE 0.156 的主因是 **UNKNOWN 混入**，不是通道失配。后续若要修
   0.156，应先改 `p_wrong` 的 UNKNOWN 处理，而非继续调先验。
2. **可部署修复（GO）**：`train_fixed`（跨 noise 平均先验，无 cue 条件）以零特权信息
   把通道失配成分的 ECE 压到 oracle 级（内容错配 0.0112、p_mode 0.0079）。若需 cue
   机制，`cue_conditioned` 对 p_wrong 更优（0.0090）。
3. **不做**：不把 learned gate 填入 joint 后验；不据本审计训练任何融合模型。

## 一句话结论

**通道失配不是 0.156 的主因（UNKNOWN 混入才是），但 train_fixed 先验对齐可作为
独立可部署修复落地——原假设 NO-GO，附带修复 GO。**
