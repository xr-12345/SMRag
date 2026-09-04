# Phase 22A — Decision-Sensitive Stop 实现报告

**状态**: DONE（代码 + 测试，无正式实验）

## 1. 一句话结论

已实现 `StopAuditMode` 双停止模式：默认 `HARD_PROBABILITY_GATE` 与修改前**逐字节等价**（git HEAD 轨迹逐字复现），新 `DECISION_VALUE_AUDIT` 模式改用**毛核验增益** `G_t^verify` 门控停止（`max_i p_i^mode` 只记录、不再作为停止条件）。259/259 测试全绿，无预测侧泄漏。

## 2. 修改范围

| 文件 | 改动 | 说明 |
|---|---|---|
| `src/powerful_medrag/decision.py` | +17 | 新增 `StopAuditMode` 枚举 + `stop_audit_mode` 配置字段（默认 `HARD_PROBABILITY_GATE`） |
| `src/powerful_medrag/joint_channel_policy.py` | +126/−10 | 拆分 gross/net 增益、双模式 `choose_action`、`_max_p_wrong`、决策日志扩充 |
| `tests/test_decision_sensitive_stop.py` | 新增 | 18 类规格测试 + 1 smoke |
| `tests/test_unknown_semantics.py` | 改 2 测试 | test_07/test_11 由「p_wrong 永不被调用」改为「p_wrong 值不驱动决策」 |

## 3. 核心行为对照

| 维度 | 旧规则（=默认模式） | 新 `DECISION_VALUE_AUDIT` |
|---|---|---|
| 停止可靠性门 | `max_i p_i^mode ≤ τ_p` **且** `G_t^verify ≤ τ_V` | 仅 `G_t^verify ≤ τ_V` |
| `max_i p_i^mode` | 停止条件 | 仅记录 |
| τ_p / τ_V | 0.05 / 0.03 | — / 0.03 |
| gross vs net | `g_verify = v_verify + C_verify` | `G_i^verify`（无成本）；动作比较用 `V_verify(i)=G_i^verify−C_verify` |
| Stop 被阻后 | 不强制 VerifyOld | 不强制 VerifyOld（`> V_new* + τ_A` 才 Verify，否则 AskNew） |

## 4. 关键证据

- **默认模式零回归**: git HEAD（`d23858a`）的 `joint_channel_policy.py` 与当前 `HARD_PROBABILITY_GATE` 在两条确定性轨迹（influenza/seed=3、common_cold/seed=2）产出**逐字节相同**的动作序列与停止理由。
- **新阈值边界**: τ_V=0.03 处 `≤` 放行、`>0.03` 阻止；τ_p=0.05 处 `≤` 放行、`>0.05` 阻止（test_07）。
- **UNKNOWN 语义**: `p_wrong=None` 在日志记为 `null`（空），从不强制为 0/1（test_16）；UNKNOWN 非 VerifyOld 候选（test_14）。
- **无泄漏**: `choose_action` 签名仍为 `["self"]`，日志无 `true_*`/`latent`/`oracle`/`noise`/`clean` 字段（test_16）。

## 5. 测试

- 完整套件: `python -m unittest discover -s tests -v` → **Ran 259 tests / OK**（240 旧 + 19 新）。
- 新文件 `tests/test_decision_sensitive_stop.py`: 19 测试（18 规格类别 + 1 决策模式 smoke），见 [TEST_LOG.txt](TEST_LOG.txt)。

## 6. 验收（§十 六项，全部满足）

1. 两模式可选 ✓（`stop_audit_mode` 配置）
2. 默认旧路径无回归 ✓（轨迹逐字复现）
3. 新模式用毛增益审计 ✓（`max_gross_verify_gain ≤ τ_V`）
4. AskNew/VerifyOld 用净值 ✓（`best_verify.utility = g − C_verify`）
5. 阻止停止不强制核验 ✓（test_10）
6. UNKNOWN 语义正确、全绿、无预测侧泄漏 ✓

## 7. 红线合规（§九）

不跑 N=5/N=20、不访问 test split、不做阈值 sweep、不改 prior 数值、不训练/融合 learned gate、不删除/弱化 RAG、不覆盖旧 artifacts（新目录）、不改预注册标准、不 git commit、不把 toy/smoke 写成正式结论 —— 全部遵守。
