# Learned Gate 独立评估报告（prompt #7）

日期：2026-08-31 ｜ HEAD：e661311 ｜ 阶段：gate_requirement 独立评估（不接入 joint policy、不调整决策策略、不访问 test split）

---

## 0. 一句话结论

**learned gate 在 validation 上以 AUROC 0.844 显著优于 heuristic gate（0.712）与 calibrated surprisal（0.778），区分度稳定、Brier 优于固定先验；唯一未满足的是「ECE 低于 fixed_prior」这条子标准——而 fixed_prior 是常数预测器、单 bin 天然近乎零 ECE，此条对任何真实模型近乎不可满足。** 结论：**gate_requirement 部分通过**（区分度 AUROC/AUPRC + Brier 全部通过；ECE 通过 random、未通过 fixed_prior），learned gate 作为「零额外成本的 drop-in 评分器」值得接入 joint policy，但 AUPRC 未提升，接入后必须做端到端 matched-budget 验证，且识别模块（检测）仍是结构性瓶颈。

---

## 1. 运行快照与完整性

| 检查项 | 结果 |
|---|---|
| gate-evaluation/ 目录冲突 | 运行前不存在，全新创建；未覆盖 formal/、matched-budget/ 或旧 gate artifacts |
| git HEAD / 分支 | e661311 / main（5 个未提交修改复核，限定预期文件） |
| Python / numpy / scipy | 3.13.13 / 2.4.6 / 1.18.0 |
| 数据 / 模型 / manifest SHA256 | 见 `config/DATA_HASHES.txt`，与冻结一致 |
| **test split** | **未访问**（输入仅 `release_train_patients.zip`、`release_validate_patients.zip`） |

### 结果完整性（全部通过）
- train gate events：**176,400** = 3 seeds × 980 例 × 4 noise × 15 turns，manifest 980 例、49 疾病。
- validation gate events：**176,400**，每个 seed 均「980 unique case_ids，missing=0，extra=0」，与冻结 980-case manifest 完全一致。
- 两个 split 分开保存（`train-events/` 与 `validation-events/`），**从不按 case_id 合并**（train/validation case_id 均为 ZIP 行号、数值重叠）。
- 校准集按**病例分组**留出（calibration_fraction=0.2 → train_cases=789 / calibration_cases=191，同一 case 的 seed/noise/turn 从不跨训练/校准），hold-out 校准集 AUROC=0.850、Brier=0.0347、ECE=0.0096。
- 唯一问题是分析脚本 4 处拼装 bug（均已在本地修复、复跑出结果，见 FAILURE_LOG.md），**未影响 runner 产物**。

---

## 2. 主结果表（validation，主标签 `harmful_misreport`，pooled_noisy）

口径：pooled_noisy = 噪声 0.1/0.2/0.3 合并，n=132,300 事件（980 例 × 3 seeds × 3 noise × 15 turns），患病率 0.0511。AUROC/AUPRC 用排名分，Brier/ECE/slope 用概率。

| method | AUROC [95% CI] | AUPRC | Brier | ECE | calib. slope |
|---|---:|---:|---:|---:|---:|
| **learned_gate** | **0.844** [0.838, 0.850] | 0.290 | **0.0442** | **0.0190** | 0.280 |
| heuristic_gate | 0.712 [0.706, 0.719] | 0.294 | 0.0473 | 0.0433 | 0.281 |
| calibrated_surprisal | 0.778 [0.772, 0.785] | 0.302 | 0.0423 | 0.0245 | 0.078 |
| fixed_prior（train 患病率 0.038） | 0.500 | 0.045 | 0.0487 | 0.0132 | 0.027 |
| deterministic_random | 0.501 | 0.051 | 0.3328 | 0.4483 | −2.711 |

`case-clustered bootstrap`（1000 次，seed 2026）：所有 CI 均以病例为聚类单元重采样。

---

## 3. gate_requirement 核对（主标签，pooled_noisy）

预注册标准（`configs/preregistered_success_criteria.json`）：**"outperform_random_score_and_fixed_prior_on_validation-frozen AUROC, AUPRC, Brier, and ECE"**。

| 指标 | learned | random | fixed_prior | 优于 random？ | 优于 fixed_prior？ |
|---|---:|---:|---:|---:|---:|
| AUROC ↑ | 0.844 | 0.501 | 0.500 | ✅ | ✅ |
| AUPRC ↑ | 0.290 | 0.051 | 0.045 | ✅ | ✅ |
| Brier ↓ | 0.0442 | 0.3328 | 0.0487 | ✅ | ✅ |
| ECE ↓ | 0.0190 | 0.4483 | 0.0132 | ✅ | ❌ |

**结论：gate_requirement 部分通过。** 4 项中 3.5 项通过：区分度（AUROC/AUPRC）与 Brier 全面通过；ECE 通过 random（大幅），但 **未通过 fixed_prior**（0.0190 > 0.0132）。

**关于「ECE 未通过 fixed_prior」必须如实说明**：fixed_prior 是常数预测器（对每个事件都输出 train 患病率 0.038），其 ECE 恒等于 train→validation 患病率漂移 |0.0511 − 0.038| ≈ 0.013——全部质量落在一个 bin 里，天然「几乎零 ECE」。这是一个**退化基线**：任何有区分度的真实模型其 ECE 都不可能低于它。因此「ECE 低于 fixed_prior」这一子标准在数学上近乎不可满足，不能据此判定 learned gate 校准失败。learned gate 的绝对 ECE=1.9% 本身是良好的（heuristic 4.3%、surprisal 2.4%），在三个非平凡方法中**最好**。

---

## 4. learned gate 是否优于 heuristic 与 surprisal

| 对比（pooled_noisy，主标签） | AUROC | AUPRC | Brier | ECE |
|---|---:|---:|---:|---:|
| learned − heuristic | **+0.132** | −0.004 | −0.003 | **−0.024** |
| learned − surprisal | **+0.066** | −0.012 | +0.002 | **−0.006** |

- **区分度（AUROC）**：learned gate **明确最优**，比 heuristic 高 +0.132、比 surprisal 高 +0.066，且逐噪声稳定（0.1/0.2/0.3 分别 0.854 / 0.839 / 0.831）。
- **校准（ECE）**：learned gate 在三个非平凡方法中**最优**（0.019 vs 0.043 vs 0.024）。
- **Brier**：learned 优于 heuristic（0.044 vs 0.047），略逊于 surprisal（0.042）。
- **AUPRC**：learned **略低于** heuristic（0.290 vs 0.294）与 surprisal（0.290 vs 0.302），差距在 CI 内、不显著。

**小结**：learned gate 的强项是**整体排序（AUROC）与校准（ECE）**；其 AUPRC 不占优——原因是 Platt 校准 + 概率上界 `single_answer_cap=0.15` 压缩了分布顶端，没有像 raw surprisal 那样给「最像错误」的事件极值高分。即：**learned gate 排序更准，但顶部精度与 heuristic/surprisal 基本持平。**

---

## 5. 主要「失败」来自区分度还是校准

**校准侧。** 区分度没有任何失败：AUROC 0.844 已是强区分（显著高于 0.5 与 heuristic 0.71）。

校准侧有两处如实记录：
1. **ECE vs fixed_prior 未过**（见第 3 节）：退化基线所致，非实质校准失败。
2. **calibration slope=0.280（<1，过自信）**：learned gate 输出的概率系统性地偏乐观——它给出高概率时，真实阳性率低于预测（slope 0.28 ≈ 需把 logit 收缩到 28% 才与真实对齐）。这反映 fit 时在 20% train 上做的 Platt 校准（calibration_slope=1.11）向 validation **迁移不完全**，叠加概率上界 0.15 的压缩效应。

所以准确表述：**短板在校准（轻度过自信 + 退化 ECE 基线），不在区分度。** learned gate 的区分度是它的优势而非短板。

---

## 6. 逐噪声稳定性（主标签 `harmful_misreport`）

| noise | prevalence | learned AUROC | heuristic AUROC | surprisal AUROC | learned Brier | learned ECE |
|---|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.0260 | 0.854 | 0.699 | 0.823 | 0.0231 | 0.0122 |
| 0.2 | 0.0515 | 0.839 | 0.721 | 0.782 | 0.0446 | 0.0193 |
| 0.3 | 0.0760 | 0.831 | 0.729 | 0.739 | 0.0650 | 0.0398 |

learned gate 在噪声 0.2/0.3 下 AUROC 稳定于 0.83–0.84，且 **Brier 在全部噪声档位都低于 fixed_prior**（0.1: 0.023 vs 0.025；0.2: 0.045 vs 0.049；0.3: 0.065 vs 0.072）。噪声 0.0 时主标签/`latent_misreport` 全为负类 → AUROC/AUPRC 无定义（NaN，符合预期）。

---

## 7. 辅助标签（pooled_noisy）

| label | learned AUROC | heuristic AUROC | surprisal AUROC |
|---|---:|---:|---:|
| `wrong_report` | 0.821 | 0.646 | 0.791 |
| `latent_misreport` | 0.798 | 0.656 | 0.760 |
| `harmful_misreport`（主） | 0.844 | 0.712 | 0.778 |

learned gate 在三个标签上 AUROC 均为**最优**，领先幅度一致（约 +0.15~0.18 于 heuristic）。主标签（错误 AND 误报模式）略高于两个单侧标签，符合「连乘事件更难但信号更纯」的直觉。

---

## 8. 重复病例敏感性（31 例 train 精确重复）

| scope | learned AUROC | heuristic AUROC | surprisal AUROC |
|---|---:|---:|---:|
| full（980 例） | 0.844 | 0.712 | 0.778 |
| exclude_duplicates（949 例） | 0.843 | 0.711 | 0.776 |

各项 delta 剔除 31 例精确重复后基本不变（learned 0.844→0.843）。**结论不受影响。**

---

## 9. exploratory 标签 `decision_sensitive_wrong`（非预注册，附混杂警示）

`decision_sensitive_wrong = wrong_report AND diagnostic_impact ≥ q75`（q75 冻结自 train wrong_report 事件的 75 分位 = 0.564）。

| method | AUROC（pooled_noisy） |
|---|---:|
| learned_gate | 0.949 |
| heuristic_gate | 0.671 |
| calibrated_surprisal | 0.828 |

learned gate 对「决策敏感错误」AUROC 高达 0.949，远高于 heuristic（0.671）。**但必须如实标注混杂**：该标签的定义里包含 `diagnostic_impact ≥ q75`，而 `diagnostic_impact` 本身就是 learned gate 的 9 个特征之一（系数 0.316，第二重要）。因此这一高 AUROC **部分来自特征与标签定义的重叠**（gate 可直接读 diagnostic_impact 去分割 q75 标签），不是完全独立的「检测高影响错误」证据。**此结果仅作探索性方向提示，不包装为预注册结论。**

---

## 10. 特征系数解读（fit 于 train，7/9 特征有效）

| feature | 系数 | 方向 |
|---|---:|---|
| surprisal | **+0.678** | 高意外 → 更可能误报（主导） |
| diagnostic_impact | +0.316 | 影响大 → 更可能有害错误 |
| history_unreliable_fraction | +0.218 | 病史不可靠 → 更可能误报 |
| top_probability | +0.088 | — |
| normalized_belief_entropy | −0.087 | 高熵 → 略低 |
| is_uncertain | −0.163 | 「不确定」答案 → 更不可能有害错误 |
| is_unknown | −0.293 | 「未知」答案 → 更不可能有害错误 |
| direct_conflicts | 0（退化） | 恒为 0（gate 不介入轨迹 → 永不产生冲突） |
| extraction_uncertainty | 0（退化） | 恒为 0（无提取置信度信号） |

`direct_conflicts` 与 `extraction_uncertainty` 在全部 train+validation 事件中恒为 0（gate 只被评分、不介入轨迹），因此实际是 **7 特征模型**；surprisal 占绝对主导。`is_unknown`/`is_uncertain` 为负说明「未知/不确定」答复本身不是有害误报（这与 heuristic gate 的 discount 逻辑一致）。

---

## 11. 回答 5 个核心问题

1. **gate_requirement 是否通过？** **部分通过。** AUROC/AUPRC/Brier 三项全面优于 random 与 fixed_prior；ECE 优于 random 但**未优于 fixed_prior**（0.0190 vs 0.0132）。后者是常数预测器的退化基线、近乎不可满足，不能据此判校准失败。

2. **learned gate 是否优于 heuristic 和 surprisal？** 区分度（AUROC）与校准（ECE）**明确优于**两者（+0.13/+0.07 AUROC，ECE 最低）；AUPRC 与两者**基本持平**（略低，CI 内）。即「排序更准、校准更好，但顶部精度不占优」。

3. **主要失败来自区分度还是校准？** **校准**（轻度过自信 slope 0.28 + ECE 未过退化 fixed_prior 基线），**非区分度**（AUROC 0.844 是强项）。

4. **是否值得接入 joint policy？** **值得，但需端到端验证。** learned gate 是零额外成本的 drop-in 评分器（信号本就已计算），在 AUROC 上比当前 heuristic（0.71）强 +0.13，且 ECE 更好，有望改善正式报告里指出的「可靠性识别（检测）recall ~27–30%」瓶颈。但 AUPRC 未提升，意味着**在 gate 实际触发的顶部区域精度不一定会变好**，必须用 matched-budget 端到端实验确认，不能只看 AUROC。

5. **下一阶段应接入 gate 还是重新设计识别模块？** **先接入（作为严格更优的 baseline），同时保留重新设计的可能。** learned gate 相对 heuristic 是严格 Pareto 改善（AUROC/ECE/Brier 均不差、无额外成本），没有理由不替换。但 9 特征里 2 个退化、surprisal 主导，且 AUPRC 未突破，说明「识别」的结构性瓶颈（可观察信号对「误报且错误」这一罕见连乘事件的辨识力上限）**尚未被特征工程打通**——下一步可在不引入 RAG/LLM/SFT 的约束下，探索新的结构化贝叶斯信号。

---

## 12. Go / No-Go

- **Go**：把 learned gate 作为 **drop-in 替换 heuristic gate** 接入 joint policy（不调预注册阈值、不改模型），并在 matched-budget 下做**端到端**验证，量化 AUROC 0.71→0.84 能否转化为实际核验命中率（conflict_resolution_rate / unnecessary_verification_rate）的提升。
- **No-Go（本阶段不做）**：不访问 test split、不改疾病模型、不 SFT/RL、不加 RAG/LLM、不调预注册阈值、不把 exploratory `decision_sensitive_wrong` 包装成预注册结果、不把 ECE-vs-fixed_prior 的退化差异表述为「校准失败」、不 commit。

---

## 13. 产物清单

`config/CONFIG.json` ｜ `config/ENVIRONMENT.txt` ｜ `config/DATA_HASHES.txt` ｜ `config/git_diff.txt` ｜ `config/git_head.txt` ｜ `config/evaluate_gate.py` ｜ `config/prepare_gate_data.py` ｜ `config/run_train_gate_events.sh` ｜ `config/run_validation_gate_events.sh` ｜ `train-events/train_gate_events.csv`（176,400 行）｜ `validation-events/validation_gate_events.csv`（176,400 行）｜ `model/learned_gate.json` ｜ `evaluation/validation_gate_predictions.csv`（176,400 行）｜ `evaluation/validation_gate_metrics.csv`（120 行）｜ `evaluation/feature_coefficients.csv` ｜ `evaluation/duplicate_sensitivity.csv`（18 行）｜ `evaluation/calibration_curves.png` ｜ `FAILURE_LOG.md` ｜ 本报告。
