# 泄漏审计 (LEAKAGE_AUDIT)

审计目标：确认 Phase 8C 的预测路径（策略决策）没有读取任何特权信息——
真实疾病、latent 状态、噪声标签、真实错误标签、模拟器隐藏模式链、或 `rho_env`。

## 1. 审计面

三条策略的对话循环：

- **heuristic_baseline** / **unified_brier_audit_corrected**：
  `run_reliability_aware_dialogue(patient, policy, initial_observations)`
  （`decision.py`），策略只读 `reports / tracker / model / corpus`。
- **joint_channel_brier_audit**：`run_joint_one`（`run_screen.py`），策略
  `JointChannelBrierAuditPolicy` 只读 `tracker / model / channel`。

## 2. 策略签名与决策日志

- `JointChannelBrierAuditPolicy.choose_action(self)` 只接受 `self`，无
  oracle/latent/noise/true-disease 参数（`test_24`）。
- `last_decision_log` 只含 `max_mode_misreport / best_new_net_value /
  best_verify_net_value / audit_blocks_stop / ...`，断言不含 `true_disease`、
  `latent`、`oracle`、`noise`（`test_24`）。
- `test_21_policy_never_reads_simulator_privilege` 确认策略从不读取模拟器的
  特权属性（`latent_states` / 隐藏模式链 / `rho_env`）。

## 3. 真实状态只出现在评估侧

`run_screen.py` 里真实状态只在**结果计算**（`compute_outcomes`、
`run_joint_one` 的 reliability_rows、`run_belief_tracker_one` 的评估）中读取，
用于计算 `correct_top1 / true_wrong / true_misreported` 等评估指标。这些值：

- 从不进入策略的 `choose_action` / `rank_actions` / `_asknew_value` /
  `_verify_value` / `reask_predictive`；
- 只写入 `case_outcomes.csv` / `reliability_predictions.csv` 的评估列。

## 4. `rho_env` 与 `rho_model` 的分离

- `rho_env` 只传给 `JointStructuredPatientSimulator`（数据生成侧），策略对象
  构造时**不接收** `rho_env`（`run_joint_one` 里 `JointChannelBrierAuditPolicy`
  只收 `model / channel / tracker`）。
- `rho_model` 通过 `JointReportChannel(repeat_mode_persistence=rho_model)` 进入
  推断侧，是显式的场景设定，不是从数据学习。
- 二者在 `case_outcomes.csv` 中作为独立列记录，永不混用。

## 5. RAG 边界（spec §十三）

`JointChannelBrierAuditPolicy` 是**独立策略**，从不实例化 retriever、不做
检索、不读 `corpus`。`run_joint_one` 全程无 RAG。红线上「JointChannelBrierAuditPolicy
未实例化 retriever」成立。旧 RAG 代码/测试未被修改（`git status` 中
`retrieval.py` / `test_retrieval.py` 无改动）。

## 6. 测试切分边界

- 数据：`release_validate_patients.zip`（validate split），frozen N=20 manifest
  取前 5/disease = 245 case（`run_screen.py` 的 `MANIFEST_SOURCE` 指向 Phase 2C
  的 `validation_case_manifest.csv`）。
- **不访问 test split**（`release_test_patients` 从未加载）。
- 模型：`model-full.json`（在全部数据上拟合的冻结模型，非本实验训练）。

## 7. 结论

无泄漏：预测路径只读可观测回答与冻结模型/通道；特权信息严格隔离在评估侧；
`rho_env` 与 `rho_model` 分离记录；joint 策略无 RAG；不访问 test split。
