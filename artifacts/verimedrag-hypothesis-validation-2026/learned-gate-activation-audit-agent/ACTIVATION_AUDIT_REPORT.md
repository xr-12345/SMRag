# Learned Gate Activation 主点审核报告

日期：2026-08-31 ｜ HEAD：e661311 ｜ 冻结主点：posterior threshold 0.85，3 seeds，2 策略，980 例/seed。

## 0. 一句话结论

**activation 修复部分改变了图景，但 NO-GO 维持。** 与 Prompt #8「全面更差」不同，用 `activation_threshold=0.01` 主动打开 VerifyOld 后，`joint_learned_gate` 在 **Top-1 / Brier 上反超 baseline**（+0.4 ~ +1.05 pp，Brier 改善）；但这是靠「**多核验 ~0.5 次**」堆出来的——**不必要核验率不降反升 +13~20 pp、冲突解决率不升反降 −12.7~18.6 pp、总问题数 +0.5**。gate 的核心目标（更精准地识别「该核验哪份报告」）**依然失败**，故不进入六阈值 sweep，转入四信号离线机制诊断。

---

## 1. 运行快照与完整性（全部通过）

| 检查项 | 结果 |
|---|---|
| 冻结主点 | posterior 0.85；980 例/seed × 3 seeds；noise 0/0.1/0.2/0.3；max_turns 15 |
| 策略 | joint_new_verify_stop（baseline）+ joint_learned_gate |
| 病例级记录 | **23,520**（=980×3×4×2），每个 (strategy,noise) 组恰 **2,940**，无缺失 |
| 两策略配对 | baseline 11,760 = learned 11,760，**0 只在一边**（case/seed/noise 完全配对） |
| **baseline 回归** | **PASS**：新 `joint_new_verify_stop` 与 frozen matched-budget t085 **byte-identical 11,760/11,760** |
| 3 进程 | 全部 exit 0，0 Traceback |
| **test split** | **未访问**（输入仅 `release_validate_patients.zip`） |
| 旧结果 | 全新目录 `learned-gate-activation-audit-agent/`，未覆盖旧 artifacts，未删除 Prompt #8 阴性结果 |
| 定向测试 | `tests.test_decision` + `tests.test_reliability_experiment` **19/19 通过** |

---

## 2. 核心指标（learned − baseline，按 noise）

| noise | ΔTop-1 | ΔBrier | Δ总问题 | Δ核验问题 | Δ无谓核验率 | Δ冲突解决率 | Δ提前停 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | **+0.0027** | **−0.0019** | +0.486 | +0.657 | +0.023 | −0.024 | −0.0003 |
| 0.1 | **+0.0099** | **−0.0210** | +0.551 | +0.393 | +0.130 | −0.127 | −0.012 |
| 0.2 | **+0.0041** | **−0.0107** | +0.539 | +0.170 | +0.170 | −0.160 | −0.010 |
| 0.3 | **+0.0105** | **−0.0245** | +0.505 | +0.037 | +0.202 | −0.186 | −0.019 |

中高噪声（0.2 / 0.3）明细：

| 指标 | baseline(0.2) | learned(0.2) | baseline(0.3) | learned(0.3) |
|---|---:|---:|---:|---:|
| Top-1 | 0.837755 | 0.841837 | 0.760204 | 0.770748 |
| Brier | 0.231129 | 0.220401 | 0.336920 | 0.312457 |
| 平均总问题 | 8.378 | 8.917 | 9.340 | 9.846 |
| 无谓核验率 | 0.688 | 0.858 | 0.578 | 0.780 |
| 冲突解决率 | 0.294 | 0.134 | 0.390 | 0.205 |
| 提前停率 | 0.034 | 0.024 | 0.050 | 0.031 |

---

## 3. 机制解读

activation 修复把 `joint_learned_gate` 从「只在历史门控打开后才替换评分」变成「learned_risk > 0.01 即可**主动打开** VerifyOld」。净效果是**核验更激进**：

- **好处**：核验次数增多 → 提前停（premature stop）下降 → 更多案例在「不确定就停」之前被核验 → **Top-1 反超 baseline +0.4~1.05 pp、Brier 改善**。这是 Prompt #8 没有的新事实。
- **代价**：learned gate 的**选择性**（该核验哪份报告）并未变强，反而把更多核验打在正确报告上——无谓核验率 +13~20 pp、冲突解决率 −12.7~18.6 pp、总问题 +0.5。

**结论**：精度/Brier 的微小反超是「用更多核验换来的」，不是「核验更聪明」。gate 存在的意义（降低 68.8% 无谓核验、提高核验命中率）**没有达成，反而恶化**。

---

## 4. 决策规则判定

**进入六阈值 sweep 的条件（需全满足）：**

| 条件 | 判定 |
|---|---|
| 不必要核验率下降 | ❌ 上升 +13~20 pp |
| 冲突解决率上升 | ❌ 下降 −12.7~18.6 pp |
| Top-1 下降 ≤ 0.5 pp | ✅ 反升 +0.4~1.05 pp |
| Brier 相对恶化 ≤ 5% | ✅ 改善 |

→ 关键的两条核验效率指标（不必要核验率、冲突解决率）**反向恶化**，**不允许进入六阈值 sweep**。

**维持 NO-GO 的判定：** gate 的核验选择性没有改善（无谓核验率未降、冲突解决率未升、问题数未降），**维持 NO-GO，转入四信号离线机制诊断**（落盘 `learned_risk` / `retrospective_error_probability` / `diagnostic_influence` / `true_wrongness`）。

---

## 5. 回答审核问题

1. **activation 修复是否改变旧 NO-GO？** **部分改变、结论不变。** 图景从 Prompt #8 的「全面更差」变为「精度/Brier 略好、核验效率更差」；但 NO-GO 的核心（gate 没解决识别瓶颈）不变。
2. **中高噪声指标变化？** 见第 2 节：Top-1 +0.4~1.05 pp、Brier 改善；但无谓核验率 +17~20 pp、冲突解决率 −16~18.6 pp、问题 +0.5。
3. **是否需要完整六阈值 sweep？** **不需要。** 核验效率两指标反向恶化，不满足进入条件。
4. **是否应进入四信号离线诊断？** **是。** 落盘并分析 `learned_risk` / `retrospective_error_probability` / `diagnostic_influence` / `true_wrongness`，定位「为什么 learned_risk 的选择性差于 retrospective」。

---

## 6. 产物清单

`CONFIG.json` ｜ `ENVIRONMENT.txt` ｜ `git_diff.txt` ｜ `run_activation_audit.sh` ｜ `config/analysis_script.py` ｜ `seed-<SEED>/ddxplus_reliability_{outcomes,summary}.csv`（3×2）｜ `case_outcomes.csv`（23,520 行）｜ `summary_by_noise.csv`（8 行）｜ `paired_deltas.csv`（4 行）｜ `baseline_regression_check.txt` ｜ `FAILURE_LOG.md` ｜ 本报告。
