# 模型-环境相关性矩阵 (MODEL_ENVIRONMENT_MATRIX)

本矩阵枚举 N=5 筛选里 `rho_env`（环境：模拟器真实复问相关性）与 `rho_model`
（推断：`JointReportChannel.repeat_mode_persistence`）的全部组合，明确每个
scenario 的语义。

## 1. 定义

- **rho_env** ∈ [0,1]：患者模拟器 `JointStructuredPatientSimulator` 生成复问时，
  真实报告模式 `E` 的持久化概率：
  `E_i^(k) ~ rho_env * 1[e'=e] + (1-rho_env) * P(e')`。
  `rho_env=0` 表示两次回答独立；`rho_env=0.9` 表示第二次回答的模式 90% 与第一次
  相同（→ 两次回答高度相关）。

- **rho_model** ∈ [0,1]：推断模型 `JointReportChannel` 对复问模式相关性的假设。
  `rho_model=0` 假设独立；`rho_model=0.9` 假设高度相关。它决定
  `reask_predictive` / `joint_likelihood` 对第二次回答的证据权重。

## 2. JOINT 策略 scenario（spec §六）

| label | rho_env | rho_model | 类型 | 语义 |
|---|---|---|---|---|
| matched_0_0 | 0.0 | 0.0 | 对齐 | 环境独立，模型假设独立 —— 与旧独立通道一致 |
| matched_0.5_0.5 | 0.5 | 0.5 | 对齐 | 环境中等相关，模型假设中等相关 |
| matched_0.9_0.9 | 0.9 | 0.9 | 对齐 | 环境高度相关，模型假设高度相关 |
| mismatched_0_0.5 | 0.0 | 0.5 | 错配 | 环境独立，模型误以为中等相关 → 低估第二次回答信息量（Case D） |
| mismatched_0.9_0.5 | 0.9 | 0.5 | 错配 | 环境高度相关，模型只假设中等相关 → 部分过度自信（Case C 的弱化） |

注：spec §六原文 "Mismatched (rho_env=0.0/rho_model=0.5, 0.9/0.5)"，格式为
`rho_env / rho_model`。

## 3. 基线策略的 rho_env 轴

`heuristic_baseline` 与 `unified_brier_audit_corrected` 使用旧 `BeliefTracker`
（其验证预测基于独立复问通道，等价于 `rho_model=0` 固定）。它们在同一
`rho_env ∈ {0.0, 0.5, 0.9}` 环境里运行：

| strategy | rho_env | rho_model（隐含） | 语义 |
|---|---|---|---|
| heuristic_baseline | 0.0 / 0.5 / 0.9 | 0（独立假设） | 基线；env=0 对齐，env>0 错配 |
| unified_brier_audit_corrected | 0.0 / 0.5 / 0.9 | 0（独立假设） | Phase 8A 修正策略；env=0 对齐，env>0 错配 |

这两个基线在 `case_outcomes.csv` 里 `rho_model` 列为空字符串（表示无该参数）。

## 4. 可识别性边界（要点）

- 在**对齐** scenario（`rho_env == rho_model`）下，模型正确地按环境的真实相关性
  折减/放大第二次回答的权重，因此病后验应当对环境相关性**不敏感**（理想）。
- 在**错配** scenario 下，模型对第二次回答的权重与真实相关性不一致，会产生
  Case C（过度自信）或 Case D（低估）的系统偏差。
- **注意**：`rho_env` 是模拟器的隐藏参数，推断模型永远只读到回答序列，读不到
  `rho_env` 本身。因此「对齐」在部署中不可直接选择——本实验只能证明「当假设正确
  时模型行为正确」，并测量「假设错配时的代价」。详见 IDENTIFIABILITY_LIMITATIONS.md。

## 5. 红线合规

- 不从 validation/test 拟合 `rho`（`rho_env` 由 harness 设定，`rho_model` 由
  scenario 固定，二者均非从数据学习）；
- 不根据 N=5 结果选择「最佳 rho」作为部署参数（对齐是评估性结论，不是超参搜索）。
