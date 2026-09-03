# Prompt #18 — Leakage Audit

目标：证明修正后的 `CorrectedUnifiedBrierAuditPolicy` 的**预测路径**（
`rank_actions` / `choose_action` 及其调用的 `asknew_value` / `verify_value` /
`brier_risk`）不读取任何真值侧信息。检索（RAG）保留，但检索信号不进入 Brier 价值。

## 1. 预测路径读取的唯一输入

`CorrectedUnifiedBrierAuditPolicy` 的动作价值只读取：

* `tracker.belief`（当前信念，由已观察报告得到）
* 拟合模型 `tracker.model`（`DiseaseStateModel`）
* 冻结的回答通道（`selector` 的 `predicted_answers`，label-free）
* `spec.cost_j`（问题成本，schema 静态字段）

`choose_action` 中的置信度/可靠性/安全判断读取：`tracker.ranked_diseases()`（信念）、
`_verification_scores`（label-free 的 retrospective error probability ×
diagnostic influence，不含检索影响）、`safety_constraint`（默认 None）。

## 2. 明确禁止且未出现的真值侧信号

| 信号 | 是否出现 |
| --- | --- |
| `patient.latent_states` | 否（`rank_actions`/`choose_action` 均 `del oracle_states`，且源码无该引用，除 docstring） |
| 真值疾病 `case.diagnosis` | 否（预测路径无引用；仅评估侧 `run_screen.py` 在对话结束后计算指标） |
| noise label / 噪声类型 | 否 |
| 真错误 / 干净答案 / oracle 报告标签 | 否 |
| 检索 Jaccard 进入 Brier 价值 | 否（`verify_value` / `asknew_value` 为 retrieval-free） |

源码级 grep：`worthiness_policy.py` 中 `latent_states` / `true_diagnosis` /
`noise` / `oracle_correction` 仅出现在 docstring/注释（第 20、152 行），非可执行代码。

## 3. 测试锁死

* `test_13_corrected_prediction_path_reads_no_true_state`：`oracle_states=None` 与
  伪造 oracle 字典，`choose_action` 返回动作与 `last_decision_log` **逐字段相等**。
* `test_14_rag_does_not_enter_corrected_brier_value`：NO_RAG 与 DYNAMIC_RAG（
  `retrieval_impact_weight=1.0` + 真实 corpus）的 VerifyOld 效用排序**完全一致**。
* 既有 `test_12`（`test_unified_brier_audit.py`）与 `test_08`（`test_worthiness_policy.py`）
  覆盖 base 策略预测路径不读真值。

## 4. 评估侧（唯一读真值处）

`artifacts/verimedrag-unified-brier-audit-corrected-n5/run_screen.py` 的 `run_one`
在 `run_reliability_aware_dialogue` **结束后**才读 `case.diagnosis` /
`case.states`，用于计算 `correct_top1` / `brier_score` / `unnecessary_verifications`
等**评估指标**。真值侧信息从不回传给策略。

## 5. 结论

无预测侧真值泄漏。RAG 保留但只作为检索日志副作用，不进入 Brier 价值。
