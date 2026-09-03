# Phase 8B — 概率正确性审计 (Probability Audit)

对 `JointReportChannel` 与 `JointReliabilityBeliefTracker` 逐条验证的数学性质，
全部由 `tests/test_joint_report_channel.py`（26 项）与 smoke（Cases A–E）锁定。

## 1. 单答边际 = AnswerChannel 边际（test_01）

`joint.single_probability(v, z, states, cue) == AnswerChannel.marginal_probability(v, z, states, cue)`
对所有 `z ∈ states`、`v ∈ states ∪ {UNKNOWN}`、`cue ∈ CertaintyCue` 逐位相等（1e-12）。

→ 联合模型的单答特例与原通道**逐位一致**，独立再问是联合模型的特例。

## 2. 联合概率归一（test_02）

对固定 `z`，`Σ_{y,y'} P(y, y' | z, v) = 1`（1e-10）。

## 3. 再问条件概率归一（test_03）

对固定 `z, e`，`Σ_{y'} reask_conditional_probability(y', z, e, v) = 1`（1e-10）。

## 4. 模式转移边界 + 边际恒等式（test_04）

* `rho=1`：`T(e'|e) = 1[e'=e]`（完全持续）。
* `rho=0`：`T(e'|e) = P_reask(e')`（完全独立）。
* 任意 `rho`：`Σ_e P(e|NONE) T(e'|e) = P(e')`（再问边际 = 首答边际，与 rho 无关）。

## 5. 独立再问 = 边际乘积（test_05）

`rho=0` 时 `joint_probability(y, y', z, v) = single_probability(y,z) · single_probability(y',z)`。

## 6. 完全持续下的再问发射（test_06）

`rho=1` 时 `reask_conditional_probability(y', z, e, v) == channel.probability(y', z, states, e)`；
对 `e=MISREPORTED` 且 `y'=z`，等于错误报告的「正确率」0.08（冻结率）。

## 7. 单答轨迹与 BeliefTracker 逐位一致（test_07）

仅单答时，`JointReliabilityBeliefTracker` 的疾病后验与旧 `BeliefTracker` 逐位相等
（1e-9）。→ 联合模型不改变单答行为，是纯增量。

## 8. 一次证据 = 一个因子，不重复计数（test_08）

`joint_likelihood(d, orig, verif) ≠ single_likelihood(d, orig) · single_likelihood(d, verif)`。
被问两次的特征是一个联合因子，而非两个独立单答因子相乘（旧路径的重复计数）。

## 9. 矛盾回答不被压缩为 UNKNOWN（test_09）

`observe_verification` 后 `ReportBundle` 同时保留 `original` 与 `verification`，
`all_answers == ["present", "absent"]`，无 UNKNOWN。

## 10. 联合似然 = 模式分解之和（test_10）

`L_i(d) == Σ_m C_i(d, m)`（`_feature_likelihood == Σ _feature_contribution`，1e-10）。
直接联合似然与按报告模式的切片分解完全一致。

## 11. 假设后验与观察一致（test_11）

`observe_single` + `observe_verification` 后的后验 == `posterior_after_verification` 的
假设后验（1e-9）。决策所用的「假设更新」与实际观察更新一致。

## 12. 状态后验归一（test_12）

单答与已核验两种情形下 `Σ_z P(Z_i=z | H_t) = 1`（1e-10）。

## 13. p_wrong 与状态后验一致（test_13）

`p_wrong(key) = 1 - P(Z_i = y_i | H_t)`；`y_i = UNKNOWN` 时恒为 1（UNKNOWN 永非真实状态）。

## 14. 模式后验归一 + 先验回退（test_14）

未观察特征 `mode_posterior == channel.mode_prior(NONE)`（精确回退）；观察后
`Σ_e P(E_i=e|H_t) = 1`，`p_mode_misreported ∈ [0,1]`。

## 15. 再问预测归一（test_15）

`Σ_{y'} reask_predictive(key) = 1`，且覆盖 `states ∪ {UNKNOWN}`。

## 16. 重复回答改变可靠度（test_16）

在 `rho=0.9` 下，两次一致回答会集中模式后验（`p_mode` 变化且 ∈[0,1]），可靠度
后验不冻结在先验上。

## 结论

四个量（疾病后验、p_mode、p_wrong、再问预测）在数学上**严格来自同一模型**，
所有归一化与分解恒等式成立，无重复计数、无 UNKNOWN 压缩。
