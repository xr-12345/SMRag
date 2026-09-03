# Prompt #18 — 修正 Stop Audit 强制核验问题

本报告记录对 Prompt #17（Phase 8A）`UnifiedBrierReliabilityAuditPolicy` 的两处理论
问题的修正，以及对应实现、测试与红线合规。

## 1. 缺陷描述

Prompt #17 实现的 `_audit_verify_old` 存在一个**强制核验**缺陷：只要
`G_t^verify >= verification_audit_threshold`，就**直接返回 `VerifyOld`**，绕过了统一
动作价值比较（AskNew 与 VerifyOld 的净 Brier 价值比较）。这违反了「Stop 审计只应
阻止 Stop，不应强制 VerifyOld」的设计本意。

正确的语义（spec §五）：

```
best_new    = argmax_all_asknew_by_net_brier()
best_verify = argmax_all_verifyold_by_net_brier()
max_gross_verify_gain = best_verify.utility + C_verify   (if best_verify and not at max)

1. base_stop_ready = confidence_ready AND safety_ready AND reliability_ready
                     AND best_acquisition_utility <= minimum_action_utility
2. audit_blocks_stop = best_verify is not None AND not at max AND
                       max_gross_verify_gain > verification_audit_threshold   # 严格 >
   # G_verify <= tau_V 不阻止 Stop；G_verify > tau_V 阻止 Stop；不得用 >=
   if base_stop_ready and not audit_blocks_stop: return STOP
3. if best_new and best_verify:
       if best_verify.utility > best_new.utility + verification_advantage_margin:
           return best_verify
       return best_new
   if best_new: return best_new
   if best_verify and best_verify.utility > 0: return best_verify
   return uncertainty_stop
```

## 2. 两处修正

### 2.1 净收益 `G_t^verify` 只用于「阻止 Stop」，不再强制 VerifyOld

`CorrectedUnifiedBrierAuditPolicy` 用纯布尔量 `audit_blocks_stop` 取代了
`_audit_verify_old` 的强制返回。`G_t^verify > tau_V` 只禁止 Stop（严格 `>` 边界），
绝不强制核验。实际动作选择回到统一净 Brier 价值比较。

### 2.2 `verification_advantage_margin = tau_A` 对所有 AskNew/VerifyOld 比较全局生效

新增配置 `verification_advantage_margin: float = 0.03`（tau_A）。VerifyOld 只有在
`V_verify > V_new + tau_A` 时才胜出；该边际对**每一次** AskNew/VerifyOld 比较生效，
不只在 Stop 被审计阻止之后。

## 3. 文件改动

| 文件 | 改动 |
| --- | --- |
| `src/powerful_medrag/decision.py` | `ReliabilityAwarePolicyConfig` 新增 `verification_advantage_margin: float = 0.03`（纳入非负校验）。旧策略不读该字段，行为不变。 |
| `src/powerful_medrag/worthiness_policy.py` | 新增 `CorrectedUnifiedBrierAuditPolicy`（继承 `UnifiedBrierReliabilityAuditPolicy`，重写 `choose_action`）；`rank_actions` 额外暂存 `_last_best_new` / `_last_best_verify`；新增策略枚举 `UNIFIED_BRIER_AUDIT_FORCED_VERIFY` 与 `UNIFIED_BRIER_AUDIT_CORRECTED`；`build_policy` 新增两分支。 |
| `tests/test_unified_brier_audit_corrected.py` | 18 个新测试（见下）。 |

## 4. 策略命名（§七.3 / §七.4）

* `unified_brier_audit_forced_verify` — **保留 Prompt #17 的强制核验行为作为独立消融策略**（`UnifiedBrierReliabilityAuditPolicy` 原样保留）。
* `unified_brier_audit_corrected` — 修正后的策略（`CorrectedUnifiedBrierAuditPolicy`）。
* `unified_brier_reliability_audit` — 旧名，向后兼容映射到 forced-verify（不删除）。

默认策略仍为 `heuristic_verify`（`ReliabilityAwareActionPolicy`），byte-identical。

## 5. 每轮日志（§七.8）

`CorrectedUnifiedBrierAuditPolicy.last_decision_log` 每轮至少记录 15 个字段：
`best_eig_question / best_brier_question / eig_brier_agreement /
best_new_gross_gain / best_new_net_value / best_verify_report_index /
best_verify_gross_gain / best_verify_net_value / verification_audit_threshold /
verification_advantage_margin / base_stop_ready / audit_blocks_stop /
stop_blocked_by_verify_audit / chosen_action / retrieval_triggered`（全部 label-free）。

## 6. 测试（18 项，`tests/test_unified_brier_audit_corrected.py`）

A. Stop 审计只阻止 Stop：`test_01`（净收益阻止 Stop 但不强制核验）、`test_02`
（低于阈值正常 Stop）、`test_03`（严格 `>` 边界：G==tau_V 不阻止）。
B. tau_A 全局边际：`test_04`（超边际才 Verify）、`test_05`（边际内 AskNew 胜）、
`test_06`（base_stop 未就绪时边际仍全局生效）。
C. 动作选择回退：`test_07`（无 Verify 候选）、`test_08`（无 AskNew 且正净值）、
`test_09`（两者皆非正 → uncertainty Stop）。
D. 净收益与预算：`test_10`（gross=net+cost）、`test_11`（maximum_verifications 上限）。
E. 日志：`test_12`（15 字段齐全）。
F. 红线：`test_13`（预测路径不读真值）、`test_14`（RAG 不进 Brier 价值）、
`test_15`（forced-verify 消融保留）、`test_16`（heuristic baseline byte-identical）、
`test_17`（负 tau_A 校验）。
G. 对话 smoke：`test_18`（终止且尊重核验预算）。

全套测试：**181 OK**（163 既有 + 18 新增）。

## 7. 红线合规

* 默认策略不变，新策略 opt-in。
* 预测路径不读 latent / 真值疾病 / noise / 真错误 / 干净答案 / oracle（`test_13`）。
* RAG 保留（retriever/retrieval_mode 仍驱动检索日志），但检索 Jaccard 永不进入
  Brier 价值（`test_14`）。
* 不删旧策略/旧测试/旧 artifact；不覆盖既有产物。
* 未 git commit。
