# Phase 22A — 泄漏审计（Prediction-side Leakage Audit）

**结论**: 无预测侧泄漏。停止/动作决策路径不读取任何真值、隐状态、噪声标签或干净标签。

## 1. `choose_action` 签名

`inspect.signature(JointChannelBrierAuditPolicy.choose_action).parameters == ["self"]`

无 `true_disease` / `latent` / `noise` / `clean` / `oracle` 参数。

## 2. 决策日志字段白名单

本轮新增的每-turn 字段仅来自**推断侧**的可观测量：

| 字段 | 来源 | privileged? |
|---|---|---|
| `posterior_confidence` / `posterior_margin` | `tracker.ranked_diseases()` | 否 |
| `max_p_mode` | `tracker.p_mode_misreported` | 否 |
| `max_p_wrong` | `tracker.p_wrong`（UNKNOWN→None） | 否 |
| `max_gross_verify_gain` | `gross_verify_gain`（信念 Brier 风险） | 否 |
| `best_net_verify_value` / `best_net_new_value` | Brier 尺度动作价值 | 否 |
| `stop_*_ready` / `stop_block_reason` | 上述量的布尔组合 | 否 |
| `chosen_action` / `chosen_index` | 动作与特征 token | 否 |

**禁止字段**（test_16 断言 `assertNotIn`）：`true_disease`, `true_state`, `true_mode`,
`true_wrong`, `latent`, `oracle`, `noise`, `clean`, `misreport_label`。

## 3. 新增方法的数据依赖

- `gross_verify_gain(key)`：读 `brier_risk(self.tracker.belief)`、`self.tracker.reask_predictive(key)`、
  `self.tracker.posterior_after_verification(...)` —— 均为推断侧后验/预测分布，无真值。
- `net_verify_value(key)` = `gross_verify_gain(key) − config.verification_cost`。
- `_max_p_wrong()`：读 `self.tracker.p_wrong(key)`，跳过 `None`（UNKNOWN 非响应），**不读真值**。

## 4. 与 Phase 8E 的一致性

Phase 8E 曾证明 `choose_action` 只调 `p_mode_misreported`、从不调 `p_wrong`。Phase 22A 为了
§七 日志新增 `_max_p_wrong()` 调用 —— 但 `p_wrong` 的**值只用于日志**，不进入停止判定或动作
比较（`stop_reliability_ready` 由 `p_mode` 与 gross gain 决定，动作比较由净值决定）。test_07
（`max_p_wrong` 记录为 None 且不驱动决策）与 test_11（强制 `p_wrong=None` 后轨迹逐字不变）共同
锁定这一性质。

## 5. 环境侧与推断侧未交叉

`run_joint_channel_dialogue` 中 `patient.answer()` 产生的 `true_mode` / `true_state` 只用于
事后记录 `PolicyTurn` 的审计字段，**从不回传**给 `choose_action` 或 `tracker` 更新路径。
