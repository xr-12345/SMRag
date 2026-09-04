# Phase 8E — Leakage Audit（无泄漏 / 无红线违反）

## 结论

Phase 8E 的预测路径（`p_wrong`、`is_nonresponse`、`p_mode`、联合后验、决策）**不读取**
真值状态、真实疾病、噪声类型、干净回答或报告模式链。train_fixed 先验为参数化
（协议噪声集合平均），未用 validation/test 标签拟合。无 test split、无 N=20、
无重训练、无 RAG 改动、无阈值扫描。

## 逐项核对（§九 红线）

| 红线 | 状态 | 证据 |
|---|---|---|
| 不访问 test split | ✅ | 只读 `release_validate_patients.zip`（validate）；manifest 为冻结 N=20 的前 5 例 |
| 不运行 N=20 | ✅ | 仅 245 例 × 2 noise × 3 seed × 2 config（matched 0/0） |
| 不融合 learned gate | ✅ | 未引入任何 gate 分数到 joint 后验 |
| 不重新训练模型 | ✅ | 只加载冻结 `model-full.json` |
| 不加入 LLM / dense / SFT / RL | ✅ | 无 |
| 保留 RAG 模块，不删除或弱化 | ✅ | `src/powerful_medrag/retrieval.py` 未改动；joint 策略本就无 RAG（Phase 8A 冻结） |
| 不改变 heuristic baseline | ✅ | 未改动 `worthiness_policy.py` 的 heuristic 路径 |
| 不覆盖旧 artifacts | ✅ | 全部写入新目录 `verimedrag-unknown-semantics-train-fixed-n5/` |
| 不修改预注册标准 | ✅ | 指标/阈值/manifest 与 Phase 8C 一致 |
| 不用 validation/test 拟合 prior/校准器 | ✅ | prior 为参数化；无校准器拟合 |
| 不把 UNKNOWN 改成 p_wrong=0 | ✅ | UNKNOWN 返回 `None`（不可定义），非 0 |
| 不 git commit | ✅ | 未 commit |

## 预测路径签名（无 oracle 表面）

`p_wrong` / `is_nonresponse` / `p_mode_misreported` / `mode_posterior` /
`state_posterior` 均只有 `(self, key)` 两个参数，无 `true_state`、`latent_mode`、
`noise`、`clean_answer` 入参（`tests/test_unknown_semantics.py::test_10` 用
`inspect.signature` 逐方法断言）。

## UNKNOWN 语义修复对轨迹的隔离（无信息泄漏、也无行为漂移）

`JointChannelBrierAuditPolicy.choose_action` 只调用 `p_mode_misreported`
（Stop 审计）与 `_verifyold_brier_values`（其已排除 UNKNOWN 候选），**从不调用
`p_wrong`**。因此 `p_wrong` 对 UNKNOWN 返回 `None` 的改动只影响日志/校准/解释/接口，
不进入决策路径——`tests/test_unknown_semantics.py::test_11` 用 mock 把 `p_wrong`
替换为抛异常，完整对话仍正常终止，直接证明策略不依赖 `p_wrong`。

## 训练集与真值访问范围

- `run_n5_screen.py` 的 `run_joint_one` 中，`patient.latent_states`（真值）仅在
  **离线评测**时读取，用于计算 `true_wrong`/`true_misreported` 标签；它**不进入**
  任何策略/后验/先验。
- `pilot_seed` 只用于模拟器 RNG，与推断无关。
