# Phase 22B — 泄漏审计（LEAKAGE_AUDIT）

**结论**: 无预测侧泄漏。四组策略的停止/动作决策路径不读取任何真值、隐状态、噪声标签或干净标签。

## 1. 预测侧 vs 评估侧的文件分离

| 文件 | 性质 | 允许字段 |
|---|---|---|
| `turn_decision_logs.csv` | **预测侧**（每轮决策日志） | case_id/seed/noise_rate/strategy/turn + 决策日志字段（后验、p_mode、gross gain、net value、stop_*_ready、chosen_action 等） |
| `case_outcomes.csv` | **评估侧** | 真值诊断、correct_top1/top3、Brier、NLL、unnecessary/resolved 核验计数等**事后**评分 |
| `reliability_predictions.csv` | **评估侧** | true_mode/true_state/true_wrong/true_misreported/is_nonresponse + p_mode/p_wrong |

**禁止字段**（true disease / latent state / true wrongness / noise type / clean answer）只出现在
`case_outcomes.csv`（diagnosis）与 `reliability_predictions.csv`（true_*），**绝不**出现在
`turn_decision_logs.csv` 或任何预测侧结构。

## 2. 决策循环的数据流（run_n5_screen.py）

```
while len(turns) < max_total_turns:
    action = policy.choose_action()          # 签名 ["self"]，无任何真值参数
    log   = policy.last_decision_log
    turn_rows.append(_turn_row(...))         # 只读 last_decision_log（推断侧）
    if action is STOP: break
    if action is NEW:
        obs, true_mode = patient.answer(key)     # obs = 观察（推断侧）
        tracker.observe_single(obs)              # tracker 只用 obs 更新
        true_state = patient.latent_states...    # 真值，仅用于 reliability_rows 审计
        true_wrong = ...                          # 同上
        reliability_rows.append({...true_*...})   # → 评估侧文件
    else:  # VERIFY
        clarification, true_mode = patient.answer(key)
        tracker.observe_verification(key, clarification)  # 只用 clarification
        original_wrong / resolved_wrong = ...（真值，仅审计）
```

关键点：
- `choose_action()` 与 `observe_single/observe_verification` 的入参**只含观察/澄清**，
  **从不含** `true_disease` / `latent` / `true_wrong` / `true_mode` / `noise` / `clean`。
- `true_state`、`true_wrong`、`true_misreported`、`original_wrong`、`resolved_wrong` 均在
  **动作与观察产生之后**才计算，只写入评估侧文件（reliability_predictions / case_outcomes），
  **不回传**给 policy 或 tracker。

## 3. noise_rate 的定位

`noise_rate`（0.2/0.3）是**实验级模拟器参数**（stratification key），出现在预测侧
`turn_decision_logs.csv` 作为行标签。它**不是**逐答案的「噪声类型」（true_mode=MISREPORTED），
且**从不**被 policy/tracker 读取——policy 用固定的跨噪声平均先验
`protocol_fixed (.75,.10,.075,.075)`，不读该 case 的 noise。逐答案噪声类型 `true_mode` 只在
评估侧 `reliability_predictions.csv`。

## 4. 与 Phase 22A 泄漏审计的一致性

Phase 22A 已锁定（test_16 断言 `assertNotIn`）：决策日志不含
`true_disease, true_state, true_mode, true_wrong, latent, oracle, noise, clean, misreport_label`。
Phase 22B 的 `_turn_row` 只搬运 `policy.last_decision_log`，未新增任何真值字段，性质不变。

## 5. 结论

- 预测路径不读真值 / 隐状态 / 噪声 / 干净标签 ✅
- 真值只存于评估侧文件 ✅
- 无逐答案噪声类型泄漏到预测侧 ✅
- 无 test split 访问、无 N=20 访问 ✅
