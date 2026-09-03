# Phase 8C — GO / NO-GO 判定 (GO_NO_GO)

## 判定：**GO（有条件继续）**

复问环境对齐机制的验证**通过**，错配代价**小且可度量**，联合通道策略**无退化**。
但 N=5 上的诊断增益不显著，需 N=20 确认；本阶段不下「joint 优于 heuristic」的
部署结论。

---

## 逐项判定

### ✅ 1. 复问相关模拟器已实现，且 rho_env=0 与旧模拟器字节一致

`JointStructuredPatientSimulator(rho_env)` 已落地；`rho_env=0` 时走
`blake2b(seed|key.token|occurrence)` + `channel.sample()` 独立路径，与旧模拟器
字节一致（byte-identical 回归检查通过）。旧模拟器默认行为未改（红线 #1）。

### ✅ 2. 模拟器频率与理论联合概率一致

`max_absolute_probability_error = 0.0052` < 0.02 容差（`probability_errors.csv`）。
模拟器采样频率与 `JointReportChannel` 的 `P(Y, Y')` 理论值吻合。

### ✅ 3. 机制 smoke A–E 方向正确

- A 独立复问（0/0）→ 与旧独立通道一致；
- B 高相关一致（0.9/0.9）→ 一致回答不重复计权；
- C 高相关环境/模型误判独立（0.9/0）→ 过度自信（posterior 0.754，与 A 持平）；
- D 独立环境/模型误判相关（0/0.5）→ 第二次回答被低权（0.740 < A 的 0.754）；
- E 冲突回答 → 两回答均保留，联合似然更新（joint/independent 比 0.099）。

### ✅ 4. 对齐敏感性可度量且小

`rho` 0→0.9（matched）Top-1 变化 ≤ 0.01、Brier ≤ 0.018，且集中在 0→0.5 的
复问触发率跳变。敏感性存在但幅度小，且机制已定位（见 N5_SCREEN_REPORT §3）。

### ✅ 5. 错配代价小

`rho_model=0.5` 固定时，env 在 {0, 0.5, 0.9} 变化引起的 Top-1 差异 < 0.007、
Brier < 0.007，均在一个标准误（0.011）内。错配不放大为诊断误差。

### ✅ 6. joint 策略无退化，但诊断增益不显著

- joint(0/0) vs heuristic：Top-1 +0.009（不显著，< 1σ）、Brier −0.018。
- joint(0/0) vs corrected：Top-1 +0.049（显著）。
- **结论**：联合通道策略不劣于基线、显著优于 corrected；但「优于 heuristic」
  需 N=20 确认（本阶段 N=5 不允许下此结论）。

### ✅ 7. 两个 Phase 8B 遗留缺陷已修复

- 非可问 spec 枚举（正确性）→ `_asknew_brier_values` 过滤 `spec.askable`；
- 联合似然重复计算（性能）→ 两层纯函数缓存，单轨迹 30s → 1.9s。
- 修复均为等价计算或正确性修复，不改成功标准或策略语义；229 tests 全绿。

### ✅ 8. 无泄漏、无红线违反

- 预测路径不读 true_disease / latent / noise / 隐藏模式链 / `rho_env`
  （LEAKAGE_AUDIT.md）；
- 不访问 test split、不跑 N=20、不从 validate/test 拟合 `rho`、不据 N=5 选最佳
  `rho`、不加 LLM/dense/SFT/RL、不重训 learned gate、不删旧策略、不 commit。

---

## 需在 N=20 阶段回答的开放问题

1. **joint vs heuristic 的诊断增益是否显著**：N=5 上 +0.009（Top-1）在噪声内。
2. **`rho_model=0` 是否应为部署默认**：本阶段显示独立假设（rho_model=0）因激活
   复问而 Top-1 最高，但这是「复问价值 > 相关假设保守」的结果，非「真实 rho=0」
   的证据。部署默认应由临床复问相关性的先验决定，不由 N=5 拟合。
3. **`p_wrong` 校准偏弱**（ECE≈0.156）是否影响下游决策，需在更大样本上评估。

## 红线合规确认

13 条红线全部遵守（详见各审计文件与 `git status`：未 commit、未覆盖旧 artifacts、
旧策略与旧模拟器默认行为未改）。

## 一句话结论

**机制成立、错配鲁棒、策略无退化 → GO；诊断增益待 N=20 确认，N=5 不下部署结论。**
