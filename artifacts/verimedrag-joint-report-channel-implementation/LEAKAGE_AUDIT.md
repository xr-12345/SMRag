# Phase 8B — 泄漏审计 (Leakage Audit)

逐路径检查预测路径是否读取任何特权信息（真实疾病 / latent 状态 / 噪声标签 /
真实错误性 / oracle）。结论：**无泄漏**。

## 1. 预测路径读什么

| 模块 | 读取来源 | 是否特权 |
| --- | --- | --- |
| `JointReportChannel` | `AnswerChannel` 冻结率 + cue 先验 + 可观测回答 | 否 |
| `JointReliabilityBeliefTracker` | `DiseaseStateModel`（从训练集拟合的状态条件）+ 回答 + 冻结通道 | 否 |
| `JointChannelBrierAuditPolicy` | tracker（后验）+ 冻结 config | 否 |
| `run_joint_channel_dialogue`（决策分支） | 仅 `patient.answer()`（采样回答）+ tracker | 否 |

`DiseaseStateModel` 是**离线**从训练集拟合的 `P(z|d)` 与疾病先验——这是合法先验，
不是逐例 oracle。

## 2. 特权信息只出现在「评估侧」标签

`run_joint_channel_dialogue` 中以下变量**只用于事后打标签**，绝不进入决策：

* `patient.latent_states` — 计算 `original_wrong / resolved_wrong`（`verification_was_unnecessary`
  等评估列）；
* `patient.diagnosis` — 计算 `ReliabilityAwareDialogueResult.true_diagnosis`；
* `true_mode`（`patient.answer()` 返回值）— 只写入 `PolicyTurn.true_report_mode` 日志。

这些与 `choose_action` / `rank_actions` / 值函数完全隔离。

## 3. 无 oracle 接口面（test_24）

`JointChannelBrierAuditPolicy.choose_action()` 的签名**只有 `self`**——没有
`oracle_states` / `report_risks` / latent / true-disease 参数。与旧策略不同，联合
策略在构造层面就没有 oracle 入口。

`last_decision_log` 的键为 `{best_new_key, best_new_net_value, best_verify_key,
best_verify_gross_gain, best_verify_net_value, max_mode_misreport,
suspicious_report_threshold, reliability_ready, verification_audit_threshold,
verification_advantage_margin, base_stop_ready, audit_blocks_stop, chosen_action}`——
全部 label-free，无 `true_disease / latent / oracle / noise`。

## 4. rho 不从 val/test 拟合（红线）

`repeat_mode_persistence` 是构造参数（默认 0.5），不在任何 split 上拟合。联合通道
只读 `AnswerChannel.parameters`（冻结），不读 `patient.latent_states`、
`patient.diagnosis`、噪声率。

## 5. RAG 保留但未接入

联合策略是 standalone，不实例化 `MedicalRetriever`；旧 RAG 路径（`decision.py` /
`worthiness_policy.py`）保持不变。本阶段不删除 RAG，联合策略也不新增 RAG 依赖。

## 6. 测试锁定

`test_24_prediction_path_has_no_oracle_surface` 锁定签名与日志键；`test_07` 锁定
单答轨迹与旧 `BeliefTracker` 逐位一致（证明联合模型没有偷偷读特权信息）。smoke
Cases A–E 全部基于公开回答与冻结通道构造，无特权输入。
