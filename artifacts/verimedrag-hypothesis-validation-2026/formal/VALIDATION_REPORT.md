# VeriMedRAG 正式 N=20 Validation 报告（prompt #5）

日期：2026-08-30 ｜ HEAD：e661311 ｜ 阶段：max-total-turns=15 正式筛选（非 matched-budget sweep）

---

## 0. 一句话结论

**oracle_verify（完美发现 + 完美纠错上界）在噪声 0.2/0.3 下同时显著提高 Top-1（+12.0 / +15.4 pp）并略微减少总问题数（−0.51 / −0.26），Brier 近乎减半，配对 95% bootstrap CI 均不跨 0 → 命中「情况 C」，VerifyOld 上界存在明确价值，可进入 matched-budget sweep；未触发停止条件。** 但可部署的 joint_new_verify_stop 虽也显著降错误，却靠多花问题（+0.27 / +0.67），且与 oracle 相差 7.3~10.4 pp —— 瓶颈在**可靠性识别（检测）而非纠错机制**。

---

## 1. 运行快照与完整性

| 检查项 | 结果 |
|---|---|
| formal/ 目录冲突 | 运行前不存在，全新创建 |
| git HEAD / 分支 | e661311 / main（4 个未提交 oracle_verify 修改已复核，限定预期文件） |
| Python / numpy | 3.13.13 / 2.4.6 |
| 数据 / 模型 / manifest SHA256 | 见 `config/DATA_HASHES.txt`，与 prompt #3 冻结一致 |
| **test split** | **未访问**（输入仅 `release_validate_patients.zip`） |

### 结果完整性（全部通过）
- reliability 病例级记录：**58,800**（=980×3 seeds×4 noise×5 策略），每个 (策略, noise, seed) 组恰 980，无缺失。
- retro 病例级记录：**11,760**（=980×3 seeds×4 noise），每个 seed 3,920，无缺失。
- 两个 runner 的 case_id 集合完全一致，且与 manifest 的 980 个 case_id 完全一致（`reliability == retro == manifest: True`）。
- oracle 路径与非 oracle 路径隔离（`oracle_verification` 仅在 `strategy=="oracle_verify"` 时启用）。
- 无 runner 失败；唯一问题是分析脚本一处拼装 bug（已修复，见 FAILURE_LOG.md），**未影响 runner 产物**。

---

## 2. 统一汇总表（3 seeds × 980 例 = 2,940 条/组）

口径：`total_atomic_questions = new_questions + verify_questions`；retro 的 `verify_questions`=clarification 数，`new_questions`=questions−clarification，`total`=questions。

| strategy | noise | Top-1 | Top-3 | Brier | new | verify | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary_eig_reliable | 0.0 | 0.940 | 0.974 | 0.104 | 5.42 | 0.00 | 5.42 |
| ordinary_eig_reliable | 0.1 | 0.867 | 0.928 | 0.224 | 5.91 | 0.00 | 5.91 |
| ordinary_eig_reliable | 0.2 | 0.784 | 0.874 | 0.353 | 6.42 | 0.00 | 6.42 |
| ordinary_eig_reliable | 0.3 | 0.697 | 0.805 | 0.498 | 6.89 | 0.00 | 6.89 |
| full_two_layer | 0.0 | 0.951 | 0.980 | 0.082 | 6.72 | 0.00 | 6.72 |
| full_two_layer | 0.1 | 0.884 | 0.945 | 0.177 | 7.43 | 0.00 | 7.43 |
| full_two_layer | 0.2 | 0.791 | 0.885 | 0.306 | 8.11 | 0.00 | 8.11 |
| full_two_layer | 0.3 | 0.711 | 0.827 | 0.426 | 8.67 | 0.00 | 8.67 |
| adaptive_history | 0.0 | 0.941 | 0.974 | 0.103 | 5.46 | 0.00 | 5.46 |
| adaptive_history | 0.1 | 0.877 | 0.935 | 0.202 | 6.19 | 0.00 | 6.19 |
| adaptive_history | 0.2 | 0.807 | 0.883 | 0.302 | 7.05 | 0.00 | 7.05 |
| adaptive_history | 0.3 | 0.727 | 0.836 | 0.409 | 7.82 | 0.00 | 7.82 |
| joint_new_verify_stop | 0.0 | 0.961 | 0.984 | 0.060 | 6.34 | 0.03 | 6.37 |
| joint_new_verify_stop | 0.1 | 0.904 | 0.953 | 0.146 | 7.17 | 0.21 | 7.39 |
| joint_new_verify_stop | 0.2 | 0.838 | 0.907 | 0.231 | 8.04 | 0.33 | 8.38 |
| joint_new_verify_stop | 0.3 | 0.760 | 0.861 | 0.337 | 8.92 | 0.42 | 9.34 |
| **oracle_verify** | 0.0 | 0.974 | 0.991 | 0.044 | 6.02 | 0.05 | 6.07 |
| **oracle_verify** | 0.1 | 0.950 | 0.979 | 0.074 | 6.61 | 0.22 | 6.83 |
| **oracle_verify** | 0.2 | **0.911** | 0.959 | **0.129** | 7.23 | 0.36 | **7.59** |
| **oracle_verify** | 0.3 | **0.865** | 0.930 | **0.195** | 7.91 | 0.50 | **8.41** |
| retro_utility_u050_b1 | 0.0 | 0.946 | NA | 0.091 | 5.46 | 0.36 | 5.82 |
| retro_utility_u050_b1 | 0.1 | 0.889 | NA | 0.177 | 6.13 | 0.41 | 6.54 |
| retro_utility_u050_b1 | 0.2 | 0.816 | NA | 0.280 | 6.84 | 0.43 | 7.27 |
| retro_utility_u050_b1 | 0.3 | 0.736 | NA | 0.407 | 7.40 | 0.47 | 7.87 |

可靠性预算说明：`maximum_verifications = 1`，故 joint / oracle / retro 三者都在**相同「每例至多 1 次核验」预算**下比较；oracle 只改变「发现 + 纠错」机制，不改预算。

---

## 3. 正式比较（配对 95% bootstrap CI，n=980 例，逐例 seed 平均后对病例 bootstrap）

A − B，正=更优。**重点噪声 0.2 / 0.3**。

| 对比（A−B） | noise | Top-1 差（CI） | Brier 差（CI） | 总问题差（CI） |
|---|---|---|---|---|
| oracle − full_two_layer | 0.2 | **+0.120** [0.106, 0.134] | **−0.177** [−0.196, −0.158] | **−0.51** [−0.66, −0.36] |
| oracle − full_two_layer | 0.3 | **+0.154** [0.137, 0.170] | **−0.231** [−0.254, −0.208] | **−0.26** [−0.42, −0.10] |
| oracle − joint_new_verify_stop | 0.2 | +0.073 [0.062, 0.085] | −0.102 [−0.118, −0.088] | −0.79 [−0.89, −0.68] |
| oracle − joint_new_verify_stop | 0.3 | +0.104 [0.090, 0.118] | −0.142 [−0.160, −0.123] | −0.93 [−1.06, −0.80] |
| joint − full_two_layer | 0.2 | +0.047 [0.034, 0.059] | −0.075 [−0.090, −0.059] | **+0.27** [0.15, 0.41] |
| joint − full_two_layer | 0.3 | +0.049 [0.036, 0.063] | −0.089 [−0.106, −0.072] | **+0.67** [0.54, 0.79] |
| adaptive_history − full_two_layer | 0.2 | +0.016 [0.004, 0.027] | −0.003 [−0.020, 0.013] | −1.06 [−1.19, −0.93] |
| adaptive_history − full_two_layer | 0.3 | +0.016 [0.004, 0.029] | −0.017 [−0.035, 0.000] | −0.86 [−0.99, −0.72] |
| retro − joint_new_verify_stop | 0.2 | −0.022 [−0.033, −0.012] | +0.049 [0.034, 0.065] | −1.11 [−1.22, −0.99] |
| retro − joint_new_verify_stop | 0.3 | −0.025 [−0.037, −0.013] | +0.070 [0.053, 0.087] | −1.47 [−1.60, −1.36] |
| ordinary_eig − full_two_layer | 0.2 | −0.007 [−0.018, 0.005] | +0.048 [0.031, 0.064] | −1.69 [−1.81, −1.57] |
| ordinary_eig − full_two_layer | 0.3 | −0.014 [−0.027, −0.001] | +0.072 [0.052, 0.091] | −1.79 [−1.92, −1.66] |

（`paired_comparisons.csv` 含全部 15 对 × 4 噪声 × 6 指标的均值/标准差/CI。）

### 关键机制读数（噪声 0.2 / 0.3）
| 指标 | oracle | joint | retro |
|---|---|---|---|
| unnecessary_verification_rate | **0.00 / 0.00** | 0.688 / 0.578 | — |
| conflict_resolution_rate | **1.00 / 1.00** | 0.294 / 0.390 | — |
| 检测 precision / recall（retro） | — | — | 0.241/0.302, 0.327/0.269 |
| 缓解率 mitigation_rate（retro） | — | — | 0.928 / 0.911 |
| premature_stop_rate | 0.018 / 0.023 | 0.034 / 0.050 | NA |
| uncertain_output_rate | 0.014 / 0.013 | 0.017 / 0.015 | NA |

oracle 的 `unnecessary_verification_rate=0`、`conflict_resolution_rate=1` 确认其为「完美发现 + 完美纠错」绝对上界。

---

## 4. 结果解释（对照第 10 节规则）

**命中「情况 C」**：oracle 在噪声 0.2/0.3 下 **Top-1 改善（+12.0 / +15.4 pp，CI 不跨 0）且 Brier 改善（−0.177 / −0.231），且总问题数不增反减（−0.51 / −0.26）**。因此：

- **情况 A（触发停止）不成立** —— 不停止，不实现 RAG / 自然语言控制器。
- **情况 B（改善但多花问题）不成立** —— oracle 不需要更多问题。
- **情况 C 成立** —— VerifyOld 上界有明确价值，**可进入 matched-budget sweep**。
- 对情况 C 的两项后续判断：
  1. **joint 是否接近 oracle？否**。oracle 比 joint 高 7.3 / 10.4 pp，且问题更少（−0.79 / −0.93）。gap 很大。
  2. **瓶颈在识别还是纠错？在可靠性识别（检测）**。joint 的核验动作 ~58–69% 打在正确报告上（浪费），仅 ~29–39% 命中真错误；纠错本身（命中后）可靠。retro 同病：检测 recall 仅 ~27–30%，缓解率却 ~91–93%。即「发现」而非「纠正」是短板。
- **情况 D 提醒（joint）**：joint 相对 full 的精度增益与「多花问题」绑定（+0.27 / +0.67 总问题），故**单点 max-turns=15 下不能宣称 joint 是 Pareto 改善**，须在相同问题预算下重比。

---

## 5. 冻结成功标准核对（不修改阈值）

引用 `configs/preregistered_success_criteria.json`（primary_reference = full_two_layer）。**明确区分三件事，不混为一谈：**

1. **预注册门槛（本阶段可查部分）**
   - `maximum_top1_drop ≤ 0.005`：joint 相对 full 无下降、反升 → 该条**通过**。
   - `maximum_relative_brier_degradation ≤ 0.05`：joint Brier 相对 full 无退化 → 该条**通过**。
   - `minimum_average_total_questions_saved ≥ 1.0`：joint 在噪声 0.2/0.3 比 full **多花** +0.27/+0.67 问题 → 该条**未通过**。
   - `pareto_requirement`（多冻结停止阈值）与 `gate_requirement`（AUROC/AUPRC/Brier/ECE）**本阶段未评估**（需多阈值曲线 + gate 分析，超出单点筛选）。
   - **结论：预注册门槛尚未整体满足，不可声称通过。**
2. **风险—问题数 Pareto 是否改善**：本阶段**未评估**（只有单点 max-turns=15）。
3. **单点筛选**：本报告所有数字只是 max-turns=15 的**筛选结果**，不是 Pareto 证明。

---

## 6. 重复病例敏感性（第 4 节）

- 规范化精确匹配（`PATHOLOGY|AGE|SEX|sorted(DDX)|sorted(EVIDENCES)|INITIAL_EVIDENCE`，SHA256）发现 validate 中 **1,643** 行与 train 精确重复；980 例 manifest 中 **31 例（3.2%）** 被标记 `is_exact_train_duplicate=1`。
- `case_id` = ZIP 内文件行号，**从 0 开始**（0 = 表头后第一行数据）。
- 主分析用全部 980 例；剔除 31 例后敏感性：

| 关键 delta（噪声 0.2 / 0.3） | 全量 980 | 剔除重复 949 |
|---|---|---|
| oracle − full 的 Top-1 | +0.1197 / +0.1537 | +0.1215 / +0.1535 |
| joint − full 的 Top-1 | +0.0466 / +0.0493 | +0.0471 / +0.0492 |

**结论不受影响**：各项 delta 剔除重复后基本不变（oracle−full 甚至略升）。不重训模型、不改主实验病例。

---

## 7. 回答 7 个核心问题

1. **oracle VerifyOld 是否存在收益？** 是，明确且显著。噪声 0.2/0.3 下 Top-1 +12.0/+15.4 pp、Brier 近乎减半、总问题数略减，配对 CI 全部不跨 0。
2. **joint policy 是否减少错误而非单纯增加问题？** 减少了错误（+4.7/+4.9 pp，显著），但**伴随**更多总问题（+0.27/+0.67）。单点下属于「花问题换精度」，且距 oracle 尚远（−7.3/−10.4 pp），不能宣称 matched-budget 优势。
3. **retro 澄清策略处于什么位置？** 精度介于 full_two_layer 与 joint 之间（0.816/0.736），但**最省问题**（7.27/7.87，比 full、joint 都少）。检测 recall 低（~27–30%）限制其上界；是「低预算、中精度」的澄清基线。
4. **结论是否受 train/validate 精确重复影响？** 否，见第 6 节，delta 稳健。
5. **是否触发停止条件？** 否（情况 A 不成立）。
6. **是否值得进入 matched-budget sweep？** 是（情况 C）。
7. **当前结果能否支持继续 VeriMedRAG？** 有条件支持：VerifyOld 上界价值明确，但**下一步应聚焦可靠性「识别」瓶颈 + matched-budget 曲线**，而非加入 RAG / NL 控制器。

---

## 8. Go / No-Go

- **Go**：进入 matched-budget 问题预算曲线实验（多冻结停止阈值下的风险—问题数 Pareto），并量化 joint 距 oracle 的缺口由「识别」贡献几何。
- **No-Go（本阶段不做）**：不访问 test split、不改模型、不 SFT/RL、不加 RAG/LLM、不调预注册阈值、不把单点结果包装成 Pareto 曲线、不 commit。

## 9. 产物清单

`config/CONFIG.json` ｜ `config/ENVIRONMENT.txt` ｜ `config/DATA_HASHES.txt` ｜ `config/validation_case_manifest_with_duplicate_flag.csv` ｜ `FAILURE_LOG.md` ｜ `merged/validation_outcomes_normalized.csv`（70,560 行）｜ `merged/summary_by_strategy_noise.csv`（24 行）｜ `merged/paired_comparisons.csv`（340 行）｜ `merged/duplicate_sensitivity.csv`（48 行）｜ 本报告。
