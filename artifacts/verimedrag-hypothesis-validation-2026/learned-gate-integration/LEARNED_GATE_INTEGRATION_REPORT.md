# Learned Gate 接入 Joint Policy 端到端验证报告（prompt #8）

日期：2026-08-31 ｜ HEAD：e661311 ｜ 阶段：8 验证方法阶段（非确认性）—— 将 AUROC 0.844 的 learned gate 以 `learned_risk_i × diagnostic_influence_i` 替换 VerifyOld 的 retrospective error probability，回答 4 个问题。

---

## 0. 一句话结论

**NO-GO。** 把 learned gate（AUROC 0.844）接入 VerifyOld 后，`joint_learned_gate` 在**全部 4 噪声档、6 冻结阈值**上严格劣于既有 `joint_new_verify_stop`：精度下降（−0.3 ~ −0.5 pp）、**不必要核验率反升**（+12.3 ~ +19.6 pp，而非下降）、冲突解决率下降（−12.2 ~ −17.5 pp），且在 matched-budget 下被 baseline joint **13/13 离散支配**。集成代码本身无 bug（回归 byte-identical、无泄漏），这是一个真实的阴性结果：learned gate 在「识别 harmful misreport」上的 AUROC 优势，**不能转化为「该核验哪份报告」的更高判别力**。

---

## 1. 运行快照与完整性（全部通过）

| 检查项 | 结果 |
|---|---|
| 输入 gate | `gate-evaluation/model/learned_gate.json`（label `harmful_misreport`，未 retrain/重校准） |
| 策略 | `joint_new_verify_stop`（baseline）+ `joint_learned_gate`（仅 VerifyOld 可靠性评分不同） |
| 冻结参数 | 980 例 × 3 seeds(2026/2027/2028) × 4 noise(0/0.1/0.2/0.3) × 6 阈值(0.60–0.95) |
| 病例级记录 | **141,120**（=980×3×4×2×6），每个 (策略,noise,阈值) 组恰 2,940，无缺失 |
| 18 进程 | 全部 exit 0，0 Traceback |
| **test split** | **未访问**（仅 `release_validate_patients.zip`） |
| **回归检查** | **PASS**：重跑 `joint_new_verify_stop` 与 frozen matched-budget **70560/70560 byte-identical** |
| 分析脚本 bug | 已修 1 处（CSV 写回时对 str/None 字段误用 `:.6f` 格式），**不影响 runner 产物** |

---

## 2. 核心指标对比（frozen 参照点：noise 0.2 / 阈值 0.85）

| 指标 | baseline joint | learned joint | Δ(learned−baseline) |
|---|---:|---:|---:|
| Top-1 精度 | 0.837755 | 0.832653 | **−0.005102** |
| 不必要核验率 | 0.688073 | 0.824561 | **+0.136488** |
| 冲突解决率 | 0.293578 | 0.167464 | **−0.126114** |
| 平均总原子问题数 | 8.378231 | 8.356803 | −0.021429 |

全噪声档方向一致（完整 24 行见 `analysis/learned_vs_baseline_deltas.csv`）：

| noise | Δ精度 | Δ不必要核验率 | Δ冲突解决率 | Δ问题数 |
|---|---:|---:|---:|---:|
| 0.0 | −0.0020 | +0.0033 | −0.0033 | −0.019 |
| 0.1 | −0.0041 | +0.1230 | −0.1215 | −0.019 |
| 0.2 | −0.0051 | +0.1365 | −0.1261 | −0.021 |
| 0.3 | −0.0034 | +0.1962 | −0.1750 | +0.006 ~ +0.009 |

---

## 3. 回答 4 个问题

1. **AUROC 0.844 能否转化为更高核验命中率？** 否。冲突解决率**不升反降**：参照点 0.294 → 0.167（−12.6 pp），噪声 0.3 时 −17.5 pp。learned gate 选出的核验目标命中真错误的概率更低。
2. **能否降低 68.8% 的不必要核验？** 否，**反向**。参照点 0.688 → 0.825（+13.6 pp），噪声 0.3 时 +19.6 pp。learned gate 把更多核验打在正确报告上。
3. **能否降低总问题数而不伤精度？** 否。问题数几乎不变（−0.02 ~ +0.01，量级可忽略），但精度**同步下降** −0.3 ~ −0.5 pp。无「省问题」红利，只有精度损失。
4. **learned joint 是否仍满足 matched-budget Pareto？** 否。13/13 个 matched-budget 点（`budget_matches_joint_learned_gate_vs_joint_new_verify_stop.csv`）上，baseline joint **离散支配** learned joint（`reference_discretely_dominates=True` 全部），参照精度优势 +0.002 ~ +0.005。

---

## 4. 机制读数与失败假说

**观测事实**：learned gate 的 AUROC 0.844 是在「该报告是否为 harmful misreport」这一**二分类排序**任务上取得的；而 VerifyOld 需要的是「核验哪份报告能最大化诊断改变量」的**影响力加权、已校准**信号。接入后三个方向全部劣化，说明两者的目标分布不一致。

**假说（未经本阶段证伪，留待后续）**：原 VerifyOld 的 retrospective `error_probability` 是 leave-one-out 诊断影响力加权估计，天然对齐核验价值；learned gate 输出被夹在 `[minimum_probability=0.005, single_answer_cap/conflict_cap]` 的窄区间，且未按诊断影响力加权，作为 VerifyOld 的 `learned_risk_i` 会稀释其与 `diagnostic_influence_i` 的乘积判别力。若需定位，需在 outcomes 层额外记录每份报告的 `learned_risk` 分布（当前 CSV 未落该字段）。

---

## 5. 冻结成功标准核对

引用 `configs/preregistered_success_criteria.json`。本阶段只评估与 learned gate 接入相关的门槛：

| 门槛 | 判定 | 依据 |
|---|---|---|
| 接入不改变疾病后验 / BeliefTracker misreport gate | ✅ 通过 | 仅改 VerifyOld 评分；`HistoryReliabilityMisreportGate` 未替换 |
| 接入无未来信息泄漏（update 前评分、不见 latent） | ✅ 通过 | 单测覆盖 + 代码审查 + 回归 byte-identical |
| gate 使核验更准 / 更省（隐含验证目标） | ❌ 未通过 | 冲突解决率↓、不必要核验率↑（第 3 节） |
| learned joint 满足 matched-budget Pareto | ❌ 未通过 | 13/13 被 baseline joint 离散支配 |

**整体结论：learned gate 集成未达到预期，No-Go。** 这不是实现 bug（集成正确、无泄漏、可复现），而是 learned gate 的判别信号与核验价值目标不匹配的阴性结果。

---

## 6. Go / No-Go

- **No-Go（本阶段）**：不把 `joint_learned_gate` 作为可部署策略替换 `joint_new_verify_stop`；不 retrain/重校准 gate（红线）；不访问 test split；不 commit。
- **后续可选方向（不属本阶段）**：① 落盘 `learned_risk` 分布以证伪第 4 节假说；② 改为用 learned gate 做**阈值门控**（超过阈值才启用 learned risk，否则回退 retrospective），而非全量替换；③ 对 gate 输出做诊断影响力重加权/重校准。均需新 prompt，不在本阶段执行。

## 7. 产物清单

`reliability-sweep/t0XX/seed-YYYY/ddxplus_reliability_{outcomes,summary}.csv`（18×2 文件）｜ `analysis/regression_check.txt` ｜ `analysis/summary_by_strategy_noise_threshold.csv`（48 行）｜ `analysis/learned_vs_baseline_deltas.csv`（24 行）｜ `analysis/curve_points.csv`（120 行）｜ `analysis/pareto_frontier_combined.csv`（120 行）｜ `analysis/budget_matches_joint_learned_gate_vs_{full_two_layer,joint_new_verify_stop}.csv` ｜ `analysis/accuracy_questions_curves.png` ｜ `config/run_learned_gate_sweep.sh` ｜ `config/analysis_script.py` ｜ 本报告。
