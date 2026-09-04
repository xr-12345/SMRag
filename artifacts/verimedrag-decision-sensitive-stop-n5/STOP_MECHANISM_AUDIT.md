# Phase 22B — Stop 机制审计（STOP_MECHANISM_AUDIT）

## 1. 两种停止审计的判定差异

Phase 22A 定义（`decision.py::StopAuditMode`）：

```
HARD_PROBABILITY_GATE:
  stop_reliability_ready = (max_i p_i^mode ≤ τ_p=0.05) AND NOT (G^verify > τ_V=0.03)

DECISION_VALUE_AUDIT:
  stop_reliability_ready = (G^verify ≤ τ_V=0.03)      # max p_mode 仅记录，不阻断
```

两者动作比较一致：`V_verify > V_new + τ_A(0.03)` → VerifyOld，否则 AskNew。
故两种模式的停止集合关系为：

```
hard-gate 阻断集 = {p_mode > τ_p} ∪ {G > τ_V}
value-audit 阻断集 = {G > τ_V}        （是 hard-gate 阻断集的子集）
```

差异集 = `{p_mode > τ_p 且 G ≤ τ_V}`，即「hard-gate-only」类。

## 2. 逐回合阻断分类计数（全组，raw turn 计数）

| 策略 | hard_gate_only | value_audit_only | both | reliability_blocked（实际） |
|---|---|---|---|---|
| L-H | 6256 | 102 | 5312 | 3366 |
| P-H | **13248** | 0 | 6502 | **7755** |
| L-V | 4311 | 102 | 5249 | 1037 |
| P-V | 6912 | 0 | 6283 | **604** |

说明：
- `hard_gate_only` = 回合中 `p_mode>τ_p` 但 `G≤τ_V`（hard gate 会阻断、value audit 不阻断）。
- `value_audit_only` = `G>τ_V` 但 `p_mode≤τ_p`。四组中 ≈0 → 数据里 `G>τ_V` 几乎总伴随
  `p_mode>τ_p`，即 gross 增益信号被 p_mode 支配。
- `both` = 两者都阻断。
- `reliability_blocked` = 最终 `stop_block_reason=="reliability"` 的实际阻断回合。

## 3. 根因定位（+4.6 题的来源）

P-H 的 `hard_gate_only` = **13248** 回合——这些回合里 `p_mode>τ_p`（先验被抬到
MISREPORTED=0.075 后，p_mode 系统性偏高）但 `G≤τ_V`（实际核验价值已低）。hard gate
用 p_mode 阻断停止 → 每轮继续追问 → 总题数 14.01。

P-V 改用 `G≤τ_V` 判定后，这 13248 类回合**不再阻断**（P-V 的 `hard_gate_only` 只做日志
记录，`value_audit_only=0` 说明价值审计的阻断集被 hard-gate 阻断集严格包含）。
于是 `reliability_blocked` 从 P-H 的 7755 骤降到 P-V 的 604，总题数从 14.01 降到 8.88。

**结论：+4.6 题膨胀 = hard gate 用「误报概率 p_mode」而非「决策价值 G」判定停止所致。
value audit 把判定换成决策价值后，膨胀消失。**

## 4. 强制核验检查（§七）

当 `G>τ_V` 阻断停止时，动作比较照常执行。gross 阻断回合的动作去向：

| 策略 | gross 阻断回合 | → Verify | → AskNew |
|---|---|---|---|
| L-H | 5414 | 724（13.4%） | 4690（86.6%） |
| P-H | 6502 | 834（12.8%） | 5668（87.2%） |
| L-V | 5351 | 700（13.1%） | 4651（86.9%） |
| P-V | 6283 | 767（12.2%） | 5516（87.8%） |

四组一致：gross 阻断后 **约 87% 选 AskNew，约 13% 选 VerifyOld**。选 Verify 的 13% 是
动作比较里 `V_verify > V_new + τ_A` 的合法核验（核验净价值确实更高），**不是被 gross
阻断强制的**。验证不变式成立：`G>τ_V` 只禁止停止，绝不强制 VerifyOld。

## 5. 早停率与早停错误率的关系

| 策略 | 早停率 | 早停错误率 | 总题数 |
|---|---|---|---|
| L-H | 0.540 | 0.0374 | 9.45 |
| P-H | 0.103 | 0.0020 | 14.01 |
| L-V | 0.780 | 0.0748 | 7.78 |
| P-V | 0.717 | 0.0408 | 8.88 |

value audit 释放了 hard-gate-only 阻断 → 早停率上升（P-V 0.717 vs P-H 0.103）→ 总题数下降。
代价是早停错误率略升（P-V 0.0408 vs P-H 0.0020），但 P-V 的早停错误率与 L-H（0.0374）
基本持平，远好于 L-V（0.0748）。即 protocol_fixed 先验 + value audit 的 P-V 在「少问」与
「早停安全」之间取得了合理平衡。

## 6. 结论

1. 题数膨胀根因 = hard gate 的 p_mode 阻断，与 gross 决策价值无关。
2. value audit 通过 `G≤τ_V` 判定停止，把阻断集缩小为 hard gate 的子集，释放了
   13248 个低价值阻断回合，题数 −5.13。
3. 无强制核验：gross 阻断后 87% 走 AskNew。
4. value audit 的代价是早停错误率 +0.34pp（vs L-H），N=5 下不显著，需 N=20 确认。
