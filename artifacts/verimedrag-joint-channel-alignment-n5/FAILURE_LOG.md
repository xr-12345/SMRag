# 失败与问题日志 (FAILURE_LOG)

记录 Phase 8C 期间发现并处理的问题。均为实现/性能缺陷，非结果伪造；修复均为
等价计算优化或正确性修复，不改动任何成功标准或策略语义。

## 1. Joint 策略枚举了非可问（non-askable）spec（正确性 + 性能缺陷，已修复）

**现象**：`JointChannelBrierAuditPolicy` 在 DDXPlus 上单条轨迹耗时 ~30s，
profile 显示 `_asknew_value` 每 turn 被调用 888 次。

**根因**：`_asknew_brier_values` 枚举 `for key in self.model.specs`（全部 889 个
spec），未过滤 `spec.askable`。而 corrected/heuristic 策略通过
`NumpyQuestionSelector.rank` 正确过滤 `not spec.askable`（questioning.py:177/241）。
DDXPlus 有 889 spec，其中仅 218 可问、671 不可问。joint 策略因此：
- 把不可问 evidence 当作可提问对象（语义错误）；
- 多做 ~4x 的 AskNew Brier 枚举（性能缺陷）。

**修复**：`joint_channel_policy.py::_asknew_brier_values` 加
`if not spec.askable: continue`，与其它策略一致。

**验证**：新增 `tests/test_joint_report_channel.py::test_23b_nonaskable_specs_are_excluded_from_asknew`；
229 tests 全绿。

**为何不在 Phase 8B 发现**：Phase 8B 的 joint 策略测试用 toy 模型（6 个 spec 全可问），
无不可问 spec，无法触发。DDXPlus 首次在 Phase 8C N=5 运行，暴露此缺陷。

## 2. 联合似然重复计算导致 ~30s/轨迹（性能，已优化）

**现象**：修复 #1 后仍 ~4s/轨迹。profile 显示 `single_likelihood` 每 turn 调用
~26.7 万次，其中大量为同一 `(disease, bundle)` 在 leave-one-out 乘积里被反复
计算（`_unnormalized_belief` 每候选 × 每回答调用一次，全部重算 memory 里每个
bundle 的似然）。

**修复**：`joint_reliability_belief.py` 加两层纯函数缓存：
- `_feature_lh_cache`：按 `(disease, bundle)` 缓存 `_feature_likelihood`
  （`ReportBundle` 是 frozen dataclass，可哈希），在 `observe_single` /
  `observe_verification` 时清空；
- `_single_lh_cache`：按 `(disease, key.token, value, certainty)` 缓存
  `single_likelihood`（纯函数，无 memory 依赖，永不清空）。

**结果**：单条 joint 轨迹 29.97s → 1.88s（~16x）。两次优化均为**等价计算优化**，
结果逐位一致（229 tests 全绿，含 byte-identical 回归检查）。

**为何安全**：`_feature_likelihood` / `single_likelihood` 是
`(disease, bundle/channel/model)` 的纯函数；缓存键覆盖全部自变量；channel/model
在对话内冻结。

## 3. 后台旧 smoke 进程残留（进程管理，已处理）

**现象**：本会话续接时，compaction 前的 N=5 smoke 任务（未优化代码，~30s/轨迹）
仍在后台运行，与新 smoke 争抢 18 CPU，导致新 smoke 前期 ~20 分钟 CPU 被占。

**处理**：`TaskStop` 停掉旧任务并 `pkill` 清理其 worker 进程，随后重跑 smoke 271s
完成。

## 4. 复问预测对已验证 feature 抛错（测试重构，非缺陷）

`test_joint_patient_simulator.py::test_18` 最初在 `observe_verification` 之后调用
`reask_predictive`，而后者对已验证 feature 抛 `ValueError`。重构为在验证前捕获
`reask_predictive`，再检查验证后的 belief/state/mode 后验。这是测试写法问题，
非实现缺陷。

## 结论

三个生产代码问题（#1 正确性、#2 性能、#3 进程）均已解决。修复不改动：
旧模拟器默认行为、旧策略、成功标准、RAG、learned gate、test split、N=20、
`rho` 拟合。所有修复由 229 tests + byte-identical 回归检查覆盖。
