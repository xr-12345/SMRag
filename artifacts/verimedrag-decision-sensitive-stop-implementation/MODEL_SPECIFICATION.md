# Phase 22A — 模型规格（Decision-Sensitive Stop）

## 1. 停止审计模式

`StopAuditMode(str, Enum)`，位于 `decision.py`：

- `HARD_PROBABILITY_GATE = "hard_probability_gate"`（默认）
- `DECISION_VALUE_AUDIT = "decision_value_audit"`

配置字段 `ReliabilityAwarePolicyConfig.stop_audit_mode`，冻结默认 = `HARD_PROBABILITY_GATE`（旧结果可复现）。

## 2. 符号

- `b_t`：第 t 轮疾病信念（Brier 尺度）。
- `R_B(b)`：信念 `b` 的 Brier 风险。
- `G_i^verify = R_B(b_t) − E_{y_i' ~ P(y_i'|H_t,Y_i,v)}[R_B(b_{t+1}^{(i,y_i')})]`：核验特征 `i` 的**毛增益**（不扣成本）。
- `G_t^verify = max_i G_i^verify`。
- `V_verify(i) = G_i^verify − C_verify`：核验 `i` 的**净值**（动作比较用）。
- `V_new(j) = R_B(b_t) − E_{y~P(y|b_t)}[R_B(b_{t+1}^{j,y})] − C_{new,j}`：AskNew 净值。

## 3. 停止判定

```
confidence_ready = top_prob ≥ τ_conf(0.85) OR margin ≥ τ_margin(0.70)
utility_low      = best.utility ≤ minimum_action_utility(0.03)
safety_ready     = safety_constraint is None or is_clear(...)

HARD_PROBABILITY_GATE:
  stop_reliability_ready = (max_i p_i^mode ≤ τ_p=0.05) AND NOT (G_t^verify > τ_V=0.03)

DECISION_VALUE_AUDIT:
  stop_reliability_ready = (G_t^verify ≤ τ_V=0.03)
  # max_i p_i^mode 仅记录

stop_ready = confidence_ready AND utility_low AND stop_reliability_ready AND safety_ready
```

## 4. 动作比较（两模式一致）

Stop 被阻后：

- 若 `best_verify.utility > best_new.utility + τ_A(0.03)` → **VerifyOld**
- 否则 → **AskNew**
- 若无任一正效用动作 → **Stop（返回不确定性）**

**G_t^verify > τ_V 只意味着「不能停止」，绝不强制 VerifyOld。**

## 5. 决策日志（每轮新增 §七 字段）

`stop_audit_mode`, `posterior_confidence`, `posterior_margin`, `max_p_mode`,
`max_p_wrong`, `max_gross_verify_gain`, `best_net_verify_value`,
`best_net_new_value`, `verification_audit_threshold`,
`verification_advantage_margin`, `stop_confidence_ready`,
`stop_reliability_ready`, `stop_utility_ready`, `stop_safety_ready`,
`stop_ready`, `stop_block_reason`, `chosen_action`, `chosen_index`。

- `chosen_index` = 特征 `token`（NEW/VERIFY），STOP 为 `None`。
- UNKNOWN 的 `p_wrong=None` 写为 `null`（空），**从不**记为 0/1。
- 日志不含任何 privileged 真值（true disease / latent / noise / clean）。

## 6. 不变式（§六）

- UNKNOWN `p_wrong=None`，非 VerifyOld 候选 —— 不变。
- `protocol_fixed` prior 仍可选、默认仍 legacy —— 不变。
- 无 learned gate、无 RAG 改动 —— 不变。
- `gross_verify_gain` 永不扣成本；`net_verify_value = gross − C_verify`。
