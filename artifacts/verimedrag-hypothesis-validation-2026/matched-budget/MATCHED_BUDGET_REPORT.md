# VeriMedRAG Matched-Budget Pareto + 识别/纠错瓶颈分解报告（prompt #6）

日期：2026-08-30 ｜ HEAD：e661311 ｜ 阶段：6 冻结停止阈值下的 matched-budget 风险—问题数曲线 + joint→oracle 缺口分解

---

## 0. 一句话结论

**在相同问题预算（`total_atomic_questions`）下，可部署的 `joint_new_verify_stop` 在全部 4 个噪声档、6 个冻结阈值上均不被 `full_two_layer` 支配（24/24 单元 `joint_dominated_by_full=0`），且 matched-budget 精度优势随噪声递增：+1.4 / +2.1 / +4.2 / +3.9 pp（噪声 0.0/0.1/0.2/0.3）→ 命中预注册 `pareto_requirement`（「多数冻结停止阈值下 better_or_non_dominated」，实际 6/6 全过）。** 但 joint 距完美上界仍有缺口，其瓶颈**明确在「可靠性识别（检测）」而非「纠错」**：缺口精确可加分解为 识别缺口 = `oracle_select − joint`、纠错缺口 = `oracle_verify − oracle_select`，识别贡献 68–89%（噪声 0.3→0.0），纠错仅 11–32%。**一个必须透明披露的观察**：joint/oracle 两类策略在 6 个阈值下精度**完全平坦**（STOP 由可靠性 gate + 效用决定，而非 posterior 阈值），故「多数阈值」判定实为同一单点重复 6 次。

---

## 1. 运行快照与完整性

| 检查项 | 结果 |
|---|---|
| matched-budget/ 目录冲突 | 运行前不存在，全新创建；**未覆盖 formal/** |
| git HEAD / 分支 | e661311 / main（6 个未提交修改：cli.py、decision.py、reliability_experiment.py、test_decision.py、test_reliability_experiment.py） |
| Python / numpy | 3.13.13 / 2.4.6 |
| **test split** | **未访问**（输入仅 `release_validate_patients.zip`） |
| 本阶段代码修改 | ① 修复阈值传播 bug（非 joint 分支 StopRule 曾硬编码 `posterior_threshold=0.85`）；② `oracle_verification` 布尔拆为 `oracle_selection`/`oracle_correction`；③ 新增 `oracle_select_same_channel` 上界 |

### 结果完整性（全部通过）
- reliability 病例级记录：**282,240**（=980×3 seeds×4 noise×4 策略×6 阈值），每个 (策略, noise, 阈值) 组恰 2,940，无缺失。
- retro 病例级记录：**70,560**（=980×3 seeds×4 noise×6 阈值），每个 (noise, 阈值) 组恰 2,940，无缺失。
- 统一 `case_outcomes.csv`：**352,800** 行；reliability 与 retro 的 980 个 case_id 集合**完全一致**。
- 18 个 reliability 进程 + 3 个 retro 进程全部 exit 0，无 Traceback。
- 分析脚本一处 bug 已修复（retro 行的 `resolved_wrong_reports`/`unnecessary_verifications` 为 NA，汇总时误 `int("")`），**不影响 runner 产物**，仅影响分析层。

---

## 2. 冻结阈值风险—问题曲线（3 seeds × 980 例，逐例 seed 平均）

口径：`total_atomic_questions = new_questions + verify_questions`；retro 的 `verify_questions`=clarification 数。曲线图见 `analysis/accuracy_questions_curves.png`。

| strategy | 阈值 | Top-1 | Brier | 总问题 |
|---|---:|---:|---:|---:|
| full_two_layer | 0.60 | 0.7395 | 0.4030 | 6.26 |
| full_two_layer | 0.70 | 0.7663 | 0.3562 | 6.99 |
| full_two_layer | 0.80 | 0.7823 | 0.3262 | 7.64 |
| full_two_layer | 0.85 | 0.7912 | 0.3056 | 8.11 |
| full_two_layer | 0.90 | 0.8007 | 0.2851 | 8.66 |
| full_two_layer | 0.95 | 0.8088 | 0.2641 | 9.39 |
| joint_new_verify_stop | 0.60–0.95 | **0.8378（平坦）** | 0.2310 | 8.32–8.38 |
| oracle_select_same_channel | 0.60–0.95 | **0.8935（平坦）** | 0.1515 | 8.11–8.16 |
| oracle_verify | 0.60–0.95 | **0.9109（平坦）** | 0.1287 | 7.55–7.59 |
| retro_utility_u050_b1 | 0.60 | 0.7568 | 0.3747 | 5.89 |
| retro_utility_u050_b1 | 0.95 | 0.8276 | 0.2537 | 8.08 |

上表为噪声 0.2（完整 4 噪声 × 6 阈值见 `curve_points.csv`）。

### ⚠️ 关键观察：joint / oracle 的曲线退化为单点
`joint_new_verify_stop`、`oracle_select_same_channel`、`oracle_verify` 三条策略在 6 个冻结阈值下 **Top-1 完全不变**、总问题数几乎不变（±0.06）。原因是这三者的 STOP 由「可靠性 gate 就绪 + 动作效用 ≤ minimum_action_utility」驱动，而非 posterior 阈值；posterior 阈值在这些配置下几乎不绑定停止条件。只有 `full_two_layer` 与 `retro_utility_u050_b1` 真正随阈值扫出曲线。

**含义**：预注册 `pareto_requirement` 要求「多数冻结停止阈值下 better_or_non_dominated」，对 joint 而言是**同一个点被重复计 6 次**。技术上 6/6 满足，但「阈值鲁棒性」这一解读在此配置下是平凡满足，须在报告中如实披露，不能包装成「joint 在多阈值下稳健」。

---

## 3. Matched-budget 比较（`total_atomic_questions` 对齐，线性插值）

`compare_at_equal_budget`：在 joint 每个阈值点的实际问题预算处，线性插值 `full_two_layer` 的精度，比较 `reference − candidate`（负 = candidate 更优）。

| 对比（候选 vs 参照） | noise | 参照精度优势均值 | 被参照支配 |
|---|---|---:|---:|
| joint vs full_two_layer | 0.0 | **−0.0137**（joint 更优） | 0/6 |
| joint vs full_two_layer | 0.1 | **−0.0210** | 0/6 |
| joint vs full_two_layer | 0.2 | **−0.0422** | 0/6 |
| joint vs full_two_layer | 0.3 | **−0.0392** | 0/6 |
| oracle_select vs full_two_layer | 0.2 | −0.1017 | 0/6 |
| oracle_verify vs full_two_layer | 0.2 | −0.1301 | 0/6 |

**解读**：joint 在同预算下比 full 高 +1.4→+4.2 pp，且噪声越大优势越明显（full 在噪声下精度被拖低更多）。`oracle_verify` 作为完美上界在同预算下比 full 高 +13.0 pp（噪声 0.2）。matched-budget 明细见 `budget_matches_*.csv`。

> 注：joint↔oracle_select、oracle_select↔oracle_verify 两组 matched-budget 因两者曲线均为平坦单点、且问题区间不重叠，`_bracket` 无法插值 → 输出为空（仅表头）。这两组关系由第 5 节「同阈值直接分解」覆盖，而非 matched-budget。

---

## 4. Pareto 判定（预注册 `pareto_requirement`）

`pointwise_pareto_joint_vs_full.csv`（同冻结阈值，joint vs primary_reference = full_two_layer）：

- **`joint_dominated_by_full = 0` 于全部 24 个（4 noise × 6 阈值）单元。**
- 每个单元 joint 的 Top-1 均 **高于** full（accuracy_delta 全部为正，+0.001 ~ +0.100 pp）。
- 问题数在低阈值段 joint 略多（+0.98 ~ +2.63），高阈值段 joint 反而更少（−1.49 ~ −0.04，因为 full 在 0.9/0.95 阈值下问题数远超 joint）。

| noise | joint_dominated_by_full=1 的单元数 | 判定 |
|---:|---:|---|
| 0.0 | 0 / 6 | 通过 |
| 0.1 | 0 / 6 | 通过 |
| 0.2 | 0 / 6 | 通过 |
| 0.3 | 0 / 6 | 通过 |

**结论：`pareto_requirement`（多数冻结阈值下 better_or_non_dominated）—— 通过（6/6，实际全过，超出「多数」要求）。** 附带第 2 节的平坦性披露：该通过是对同一单点的重复计数。

**但要区分第二层**：在**全体策略**（含 oracle 上界）的 Pareto 前沿上，joint 在噪声 0.2/0.3 下**被 `oracle_verify` 支配**（oracle_verify 精度更高且问题更少），故 `pareto_frontier.csv` 中 joint 的 `on_pareto_frontier=0`。这是**预期且正确**的 —— oracle_verify 是不可部署的完美上界（读取 latent_states），它不是可比的部署目标；joint 相对它的缺口正是第 5 节分解的对象。预注册的 Pareto 门槛只针对 primary_reference（full_two_layer），**不与 oracle 上界混同**。

---

## 5. joint → oracle 缺口分解（识别 vs 纠错）

三者仅在机制上不同（同 frozen 阈值、同 `maximum_verifications=1` 预算）：

- `identification_gap = oracle_select_same_channel − joint`（把 noisy 检测换成完美检测，纠错通道不变）
- `correction_gap = oracle_verify − oracle_select_same_channel`（检测不变，把 normal re-ask 换成完美纠错）
- `total_gap = oracle_verify − joint = identification_gap + correction_gap`（残差 = 0.00e+00，严格可加）

| noise | joint | oracle_select | oracle_verify | 识别缺口 | 纠错缺口 | 总缺口 | 识别占比 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.9609 | 0.9724 | 0.9738 | 0.0116 | 0.0014 | 0.0129 | 89.5% |
| 0.1 | 0.9037 | 0.9408 | 0.9497 | 0.0371 | 0.0088 | 0.0459 | 80.7% |
| 0.2 | 0.8378 | 0.8935 | 0.9109 | 0.0558 | 0.0173 | 0.0731 | 76.3% |
| 0.3 | 0.7605 | 0.8316 | 0.8650 | 0.0711 | 0.0333 | 0.1044 | 68.1% |

### 机制读数（噪声 0.2，`summary_by_strategy_noise_threshold.csv`）
| 指标 | joint | oracle_select | oracle_verify |
|---|---:|---:|---:|
| unnecessary_verification_rate | 0.688 | **0.000** | **0.000** |
| conflict_resolution_rate | 0.294 | 0.939 | **1.000** |
| avg verification_questions | 0.333 | 0.361 | 0.361 |

**瓶颈在「识别」**：joint 的核验动作 **68.8% 打在正确报告上（浪费）**，仅 29.4% 命中真错误；换成完美识别（oracle_select）后，不必要核验归零、冲突解决率跳到 93.9%，精度 +5.58 pp。而即便识别完美，normal re-ask 通道也做不到 100% 纠错（93.9% vs 100%），残差 6.1% 构成纠错缺口 +1.73 pp。

**噪声外推**：识别缺口随噪声 1.16→7.11 pp（≈6×），纠错缺口 0.14→3.33 pp（≈24×，增速更快）；纠错占比从 11% 升到 32%，但**所有噪声档下识别仍是绝对主导瓶颈**。

---

## 6. 冻结成功标准核对（不修改阈值）

引用 `configs/preregistered_success_criteria.json`（primary_reference = full_two_layer）。逐条报告，含失败项。

| 预注册门槛 | 本阶段判定 | 依据 |
|---|---|---|
| `maximum_top1_drop ≤ 0.005` | ✅ 通过 | joint 相对 full 无下降、反升（accuracy_delta 全为正） |
| `maximum_relative_brier_degradation ≤ 0.05` | ✅ 通过 | joint Brier 相对 full 无退化（0.231 vs 0.306，噪声 0.2） |
| `minimum_average_total_questions_saved ≥ 1.0` | ❌ 未通过 | joint 在同阈值下比 full **多花**问题（低噪声段 +0.98~+2.63）；joint 是「同预算换精度」，不是「省问题」 |
| `pareto_requirement`（多数冻结阈值 better_or_non_dominated） | ✅ 通过（6/6，见第 4 节） | 本阶段正式评估，是 prompt #6 主交付 |
| `gate_requirement`（AUROC/AUPRC/Brier/ECE） | ⏳ 本阶段未评估 | 需 gate 分析，超出 matched-budget 范围 |

**整体结论：预注册门槛仍未全部满足**（questions_saved 一项失败），不可宣称整体通过；但 **`pareto_requirement` 这一此前未评估的门槛在本阶段明确通过**。

---

## 7. 回答本阶段核心问题

1. **joint 在 matched-budget 下是否 Pareto 优于 full_two_layer？** 是。同预算精度 +1.4/+2.1/+4.2/+3.9 pp（噪声 0.0/0.1/0.2/0.3），24/24 单元不被 full 支配，命中 `pareto_requirement`。
2. **joint→oracle 的缺口由「识别」还是「纠错」贡献？** 由**识别（检测）**贡献绝大部分：68–89% 依噪声档；纠错仅 11–32%。
3. **retro 澄清基线处于什么位置？** 同预算下 retro（PAMIS 风格 retro 效用）精度介于 full 与 joint 之间、最省问题（噪声 0.2：0.828 @ 8.08 vs joint 0.838 @ 8.38）；是「低预算、中精度」基线，不改变本阶段关于 joint/oracle 的结论。
4. **阈值平坦性是否影响结论？** 影响解读：joint/oracle 的 STOP 由可靠性 gate 驱动，6 阈值曲线退化为单点，故 Pareto「多数阈值」是平凡满足。结论方向不变，但「阈值鲁棒性」不作强声明。

---

## 8. Go / No-Go

- **Go**：joint 的 matched-budget Pareto 判定已通过；识别缺口已量化（68–89% 由检测贡献）。下一步可聚焦**改进可靠性「识别」**（降低 68.8% 的不必要核验率、提高 29.4% 的冲突命中率），并单独评估 `gate_requirement`。
- **No-Go（本阶段不做）**：不访问 test split、不改模型、不 SFT/RL、不加 RAG/LLM、不调预注册阈值、不把 oracle 上界描述为可部署方法、不把平坦曲线包装成「多阈值稳健」、不 commit。

## 9. 产物清单

`config/CONFIG.json` ｜ `config/run_reliability_sweep.sh` ｜ `config/run_retro_sweep.sh` ｜ `config/analysis_script.py` ｜ `config/git_head.txt` ｜ `config/git_diff.txt` ｜ `config/ENVIRONMENT.txt` ｜ `config/DATA_HASHES.txt` ｜ `reliability-sweep/t0XX/seed-YYYY/ddxplus_reliability_outcomes.csv`（18 文件）｜ `retro-sweep/seed-YYYY/ddxplus_clarification_case_outcomes.csv`（3 文件）｜ `analysis/case_outcomes.csv`（352,800 行）｜ `analysis/summary_by_strategy_noise_threshold.csv`（120 行）｜ `analysis/curve_points.csv`（120 行）｜ `analysis/pareto_frontier.csv`（120 行）｜ `analysis/pointwise_pareto_joint_vs_full.csv`（24 行）｜ `analysis/oracle_gap_decomposition.csv`（24 行）｜ `analysis/budget_matches_*.csv`（5 文件）｜ `analysis/accuracy_questions_curves.png` ｜ 本报告。
