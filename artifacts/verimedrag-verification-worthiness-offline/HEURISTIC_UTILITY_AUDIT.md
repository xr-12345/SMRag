# Phase 4 Step 1 — heuristic_verify_utility 负相关审计

数据：/Users/xr-12345/Desktop/SafeMedRAG/artifacts/verimedrag-action-value-audit/action_value_samples.csv，VerifyOld 样本 n=4143。

## 实现/计账检查（1–7）

1. **符号约定**：`_verification_utility` 返回 `existing = disease_gain + 0.01·H(error) + 0.25·influence − 0.03`，数值越大表示「越值得核验」；`rank_actions` 用 `reverse=True` 降序取最大值。**越大越优，方向正确**。
2. **排序方向**：`sorted(actions, key=utility, reverse=True)`；`_verification_scores` 也按 `score`（=error_prob×influence）降序。**正确**。
3. **记录的是原始分数**：audit CSV 的 `heuristic_verify_utility` 列来自 `policy._verification_utility(score)[0]`（原始 existing utility 值），非 rank 非 cost。**正确**（见 `run_audit.py` score_state 的 `existing_utility, _final = policy._verification_utility(score)`）。
4. **成本只扣一次**：utility 里 `− verification_cost` 一次；`v_real` 由 `realized_value_mc` 计算为 `gross_brier_reduction − cost` 一次；`v_bayes` 由 `verify_value` 计算 `R(b_t)−E[R]−C` 一次。**无重复扣减**。
5. **report_index 对齐**：`_verification_scores` 的 `report_index` 来自 `enumerate(reports)`，audit 用 `reports[score.report_index]` 取值，`retrospective_error_prob`/`retrieval_impact`/`v_bayes`/`v_real` 均按同一 report_index 计算。**对齐**。
6. **候选截断**：每状态只记录 top-5 回溯分数候选（`MAX_VERIFY=5`）。这是按 `error_prob×influence` 截断的**上尾样本**——相关性是在「系统认为最值得核验的 top-5 内部」计算的，不是全体报告。它不会翻转符号，但限制了解释域：−0.22 表示「在 top-5 候选中，系统排名更高的反而 realized 价值更低」。
7. **realized 符号**：`value = gross_brier_reduction − cost = (L_brier_before − E[L_brier_after]) − cost`。正 = 风险下降。**正确**。

## 分层相关性（8）

| 分层 | n | Spearman(heuristic_verify_utility, v_real) | p |
|---|---|---|---|
| overall | 4143 | -0.2226 | 1.07e-47 |
| noise=0.2 | 2093 | -0.2210 | 1.41e-24 |
| noise=0.3 | 2050 | -0.2276 | 1.72e-25 |
| state=early | 437 | -0.2110 | 8.69e-06 |
| state=middle | 1615 | -0.1849 | 6.96e-14 |
| state=late | 2091 | -0.2649 | 6.59e-35 |
| turn∈[0,3] | 924 | -0.2854 | 8.93e-19 |
| turn∈[4,6] | 816 | -0.2162 | 4.34e-10 |
| turn∈[7,9] | 1193 | -0.1390 | 1.44e-06 |
| turn∈[10,99] | 1210 | -0.2106 | 1.35e-13 |

### 分病种相关性（每病种 ≥10 样本）

| 病种 | n | Spearman |
|---|---|---|
| Acute COPD exacerbation / infection | 88 | -0.3752 |
| Acute dystonic reactions | 54 | -0.2010 |
| Acute laryngitis | 106 | -0.1069 |
| Acute otitis media | 109 | -0.1727 |
| Acute pulmonary edema | 84 | -0.1519 |
| Acute rhinosinusitis | 106 | -0.2887 |
| Allergic sinusitis | 57 | -0.0847 |
| Anaphylaxis | 68 | -0.5049 |
| Anemia | 27 | -0.3425 |
| Atrial fibrillation | 59 | -0.5276 |
| Boerhaave | 92 | -0.1906 |
| Bronchiectasis | 79 | -0.5574 |
| Bronchiolitis | 100 | -0.0770 |
| Bronchitis | 95 | -0.3909 |
| Bronchospasm / acute asthma exacerbation | 87 | -0.1578 |
| Chagas | 109 | -0.3349 |
| Chronic rhinosinusitis | 99 | -0.3096 |
| Cluster headache | 96 | -0.1079 |
| Croup | 99 | -0.1978 |
| Ebola | 101 | -0.0620 |
| Epiglottitis | 85 | -0.1871 |
| GERD | 85 | -0.1252 |
| Guillain-Barré syndrome | 70 | +0.0374 |
| HIV (initial infection) | 41 | -0.5537 |
| Influenza | 88 | -0.3679 |
| Inguinal hernia | 67 | -0.3789 |
| Larygospasm | 98 | -0.4483 |
| Localized edema | 75 | -0.1942 |
| Myasthenia gravis | 65 | -0.3010 |
| Myocarditis | 104 | -0.4720 |
| PSVT | 103 | -0.2869 |
| Pancreatic neoplasm | 54 | -0.2952 |
| Panic attack | 76 | -0.1632 |
| Pericarditis | 98 | +0.0216 |
| Pneumonia | 76 | -0.0414 |
| Possible NSTEMI / STEMI | 90 | -0.4154 |
| Pulmonary embolism | 76 | -0.3102 |
| Pulmonary neoplasm | 97 | -0.1154 |
| SLE | 45 | -0.2495 |
| Sarcoidosis | 104 | -0.0676 |
| Scombroid food poisoning | 80 | -0.1990 |
| Spontaneous pneumothorax | 96 | -0.4016 |
| Spontaneous rib fracture | 87 | -0.2941 |
| Stable angina | 108 | -0.0715 |
| Tuberculosis | 96 | -0.1987 |
| URTI | 89 | -0.0954 |
| Unstable angina | 104 | -0.1789 |
| Viral pharyngitis | 88 | -0.2950 |
| Whooping cough | 83 | -0.0664 |
| **mean over diseases** | | -0.2420 |

## within-case 相关性（9，排除病例难度混杂）

方法：把 heuristic_verify_utility 与 v_real 分别在每个 case 内去均值（demean），再对去均值后的 pooled 样本计算 Spearman；同时给出每病例 Spearman 的均值。

- pooled within-case Spearman = **-0.1528** (p=4.44e-23, n=4143)
- 每病例 Spearman 均值 = **-0.2420** (n=49 cases)

## 人工检查样本（10）：heuristic_verify_utility 最高/最低各 20

列：utility、error_prob、influence(反推)、retrieval_impact、v_bayes、v_real、gross_brier_reduction、reported_value、is_report_wrong。

### 最高 20（系统最想核验）

| utility | error_prob | influence | retrieval_impact | v_bayes | v_real | gross_brier↓ | reported | wrong |
|---|---|---|---|---|---|---|---|---|
| +0.5320 | 0.531 | 0.706 | 0.46 | +0.0667 | +0.9513 | +0.9813 | 9 | 1 |
| +0.5294 | 0.541 | 0.694 | 0.46 | +0.0361 | +0.2707 | +0.3007 | 2 | 1 |
| +0.5270 | 0.569 | 0.668 | 0.57 | -0.0314 | -0.0170 | +0.0130 | 6 | 1 |
| +0.5254 | 0.589 | 0.651 | 0.00 | +0.0879 | +0.7541 | +0.7841 | V_12 | 1 |
| +0.5234 | 0.578 | 0.657 | 1.00 | -0.0544 | -0.1877 | -0.1577 | present | 0 |
| +0.5234 | 0.557 | 0.674 | 0.18 | -0.0131 | +0.0901 | +0.1201 | present | 1 |
| +0.5222 | 0.619 | 0.624 | 0.95 | -0.0289 | -0.0909 | -0.0609 | present | 0 |
| +0.5198 | 0.540 | 0.683 | 0.18 | -0.0840 | +0.0819 | +0.1119 | present | 1 |
| +0.5173 | 0.482 | 0.734 | 0.00 | -0.0308 | -0.0181 | +0.0119 | 1 | 1 |
| +0.5123 | 0.539 | 0.675 | 0.46 | +0.1493 | -0.2287 | -0.1987 | present | 0 |
| +0.5115 | 0.659 | 0.585 | 0.95 | -0.0343 | -0.0064 | +0.0236 | present | 1 |
| +0.5068 | 0.516 | 0.687 | 0.89 | +0.1006 | +0.4393 | +0.4693 | present | 1 |
| +0.5057 | 0.484 | 0.716 | 0.18 | -0.0369 | -0.2091 | -0.1791 | present | 0 |
| +0.5041 | 0.500 | 0.699 | 0.46 | -0.0877 | +0.1052 | +0.1352 | present | 1 |
| +0.4990 | 0.434 | 0.759 | 0.00 | +0.0220 | -0.0945 | -0.0645 | 9 | 0 |
| +0.4981 | 0.687 | 0.554 | 0.75 | +0.2782 | +0.3126 | +0.3426 | present | 1 |
| +0.4976 | 0.425 | 0.767 | 0.57 | +0.1715 | +0.7290 | +0.7590 | 1 | 1 |
| +0.4890 | 0.715 | 0.529 | 0.00 | +0.2003 | +0.4930 | +0.5230 | absent | 1 |
| +0.4878 | 0.541 | 0.642 | 0.00 | +0.0462 | -0.1065 | -0.0765 | absent | 0 |
| +0.4867 | 0.457 | 0.717 | 0.75 | -0.0419 | +0.1351 | +0.1651 | present | 1 |

### 最低 20（系统最不想核验）

| utility | error_prob | influence | retrieval_impact | v_bayes | v_real | gross_brier↓ | reported | wrong |
|---|---|---|---|---|---|---|---|---|
| -0.0300 | 0.000 | 0.000 | 0.82 | -0.0300 | -0.0300 | -0.0000 | present | 0 |
| -0.0300 | 0.000 | 0.000 | 0.95 | -0.0299 | -0.0300 | -0.0000 | present | 0 |
| -0.0300 | 0.000 | 0.000 | 0.57 | -0.0300 | -0.0300 | +0.0000 | present | 0 |
| -0.0300 | 0.000 | 0.000 | 0.89 | -0.0299 | -0.0300 | +0.0000 | present | 0 |
| -0.0300 | 0.000 | 0.000 | 0.67 | -0.0300 | -0.0300 | -0.0000 | present | 0 |
| -0.0300 | 0.000 | 0.000 | 0.75 | -0.0298 | -0.0300 | -0.0000 | present | 0 |
| -0.0300 | 0.000 | 0.000 | 1.00 | -0.0300 | -0.0300 | -0.0000 | present | 0 |
| -0.0300 | 0.000 | 0.000 | 0.82 | -0.0300 | -0.0300 | -0.0000 | present | 0 |
| -0.0300 | 0.000 | 0.000 | 0.82 | -0.0297 | -0.0300 | -0.0000 | present | 0 |
| -0.0300 | 0.000 | 0.000 | 1.00 | -0.0300 | -0.0300 | -0.0000 | present | 0 |
| -0.0300 | 0.000 | 0.000 | 1.00 | -0.0300 | -0.0300 | -0.0000 | present | 0 |
| -0.0300 | 0.000 | 0.000 | 0.82 | -0.0299 | -0.0300 | -0.0000 | present | 0 |
| -0.0300 | 0.000 | 0.000 | 0.95 | -0.0298 | -0.0300 | +0.0000 | present | 0 |
| -0.0300 | 0.000 | 0.000 | 0.00 | -0.0300 | -0.0301 | -0.0001 | absent | 0 |
| -0.0300 | 0.000 | 0.000 | 0.00 | -0.0306 | -0.0300 | -0.0000 | absent | 0 |
| -0.0300 | 0.000 | 0.000 | 0.00 | -0.0305 | -0.0300 | +0.0000 | absent | 0 |
| -0.0300 | 0.000 | 0.000 | 0.00 | -0.0305 | -0.0300 | -0.0000 | absent | 0 |
| -0.0300 | 0.000 | 0.000 | 0.57 | -0.0299 | -0.0300 | -0.0000 | present | 0 |
| -0.0300 | 0.000 | 0.000 | 0.57 | -0.0298 | -0.0300 | -0.0000 | present | 0 |
| -0.0300 | 0.000 | 0.000 | 0.00 | -0.0303 | -0.0300 | -0.0000 | absent | 0 |

## 结论

**负相关是真实现象，不是实现或计账错误。** 逐条依据：

1. **符号/排序/原始值/成本/对齐/realized 符号**（检查 1–7）全部正确——记录的是「越大越优」的原始 existing utility，成本只扣一次，realized 符号正=风险下降。
2. **分层（检查 8）全部为负**：noise 0.2 / 0.3、state early/middle/late、turn∈[0,3]/[4,6]/[7,9]/[10,99] 八个分层的 Spearman 均落在 **−0.14 ~ −0.29**，无一为正。分病种 49 个中 47 个为负（mean −0.242），仅 Guillain-Barré(+0.037)、Pericarditis(+0.022) 轻微为正且样本少。
3. **within-case（检查 9）仍为负**：在每个病例内去均值后 pooled Spearman = **−0.153**（p=4.4e-23），每病例 Spearman 均值 −0.242。这**排除了「病例难度混杂」**的解释——即便在同一个病例内部，系统给更高 heuristic utility 的报告，其 realized 价值反而更低。负相关不是「难病例既高分又低收益」的伪相关。
4. **人工检查（检查 10）**：最低 20 名全是 `error_prob=0.000, influence=0.000` 的退化项（utility=−0.03=纯成本，realized=−0.03，识别正确）；负相关的驱动在**顶部**——高 utility（error_prob 0.4~0.7 × 高 influence）的 realized 价值正负混杂，且不少 `is_report_wrong=1` 高 utility 样本 realized 价值为负。即 `error_prob × influence` 这一排序键本身不是 realized 核验价值的良好（更不是正向）代理。

### 含义
- 现有 `heuristic_verify_utility` 的核心项 `error_probability × diagnostic_influence` 对 realized 核验价值是**反预测器**。这佐证 Phase 3A 的「主动选错核验对象」判断，不是数据或代码错误。
- 无实现错误 → 按 spec **无需** 记录 FAILURE_LOG、无需回滚、无需重生成旧 artifacts；可直接进入 Step 2（特征抽取 + 标注 + 训练）。
- 注意检查 6 的截断域限制：该负相关是在「系统自己挑出的 top-5 回溯候选中」计算的，结论严格表述为「在系统认为最值得核验的候选内部，排名越高 realized 价值越低」。