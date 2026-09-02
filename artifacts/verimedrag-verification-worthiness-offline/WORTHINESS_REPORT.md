# Phase 4 — Learned Verification-Worthiness 离线训练与验证报告

裁决：**No-Go**（细节见 `GO_NO_GO.md`）。

## 1. heuristic 负相关是否是真实问题

见 `HEURISTIC_UTILITY_AUDIT.md`。结论：**是真实现象，不是实现/计账错误**。within-case 去均值后 Spearman 仍为 −0.153（p=4.4e-23），排除病例难度混杂；`error_prob × influence` 是 realized 核验价值的反预测器。

## 2. frozen learned worthiness 模型

- gain head: **hgb** (params {'learning_rate': 0.1, 'max_leaf_nodes': 15})
- harm head: **hgb** (params {'learning_rate': 0.1, 'max_leaf_nodes': 31})
- tau_harm = 0.4129（train grouped-CV 选定）
- 特征（9）: retrospective_error_prob, diagnostic_influence, error_prob_times_influence, current_risk, n_reports, turn_index, state_index, retrieval_impact, error_prob_times_impact
- train OOF gain Spearman = 0.1923，harm AUC = 0.8661

## 3. 相对 V_Bayes / heuristic 的排序与 regret

| method | Spearman | regret(mean) | matched net Brier |
|---|---|---|---|
| heuristic_verify_utility | -0.2329 | 0.09046 | 0.00644 |
| retrospective_error_probability | -0.0498 | 0.08805 | 0.00493 |
| retrieval_impact | -0.041 | 0.10054 | -0.01708 |
| error_prob_times_impact | -0.0607 | 0.09473 | 0.0074 |
| v_bayes | 0.2221 | 0.10023 | -0.00559 |
| learned_norag | 0.1169 | 0.09207 |  |
| learned_real | 0.1297 | 0.08522 | 0.01288 |
| learned_shuffled | 0.1234 | 0.08928 |  |

- learned_real Spearman 0.1297 vs V_Bayes 0.2221 vs heuristic -0.2329
- learned_real regret 0.0852 vs V_Bayes 0.1002 vs heuristic 0.0905

## 4. 是否减少 correct→wrong

- learned_real matched-budget correct→wrong = 0.03180，V_Bayes = 0.02207
- harm head validation AUC = 0.8702
- harm gate 使 correct→wrong 从 0.03534（无 gate）降到 0.03180（加 gate），减少 +0.00354

## 5. matched-trigger 净收益

| method | triggers | mean net Brier | mean gross Brier | frac>0 |
|---|---|---|---|---|
| heuristic_verify_utility | 2917 | 0.00644 | 0.03644 | 0.26226 |
| v_bayes | 2917 | -0.00559 | 0.02441 | 0.26568 |
| learned_real | 2917 | 0.01288 | 0.04288 | 0.2482 |

## 6. RAG 是否优于 no-RAG 和 shuffled-RAG

| model | Spearman | 95% CI | net Brier@k |
|---|---|---|---|
| learned_norag | 0.1169 | [0.095,0.1362] | 0.00746 |
| learned_real | 0.1297 | [0.108,0.1491] | 0.01582 |
| learned_shuffled | 0.1234 | [0.1043,0.1425] | 0.00626 |

- RAG 增量（full>no-RAG 且 full>shuffled）：**是**

## 7. 是否存在泄漏

无。特征仅 label-free/deployable（test 1/6）；验证集不参与超参选择（test 7）；默认策略未接入 learned 模型（test 10）；AskNew EIG 未改动（test 11）。

## 8. Go / Conditional Go / No-Go

**No-Go**。判据通过 8/10（见 `GO_NO_GO.md`）。

## 9. 是否值得进入在线 Joint Policy 集成

**不进入（No-Go for the primary bar）**：learned 模型的 Spearman 排序未超过 V_Bayes（0.1297 < 0.2221），且 correct→wrong 高于 V_Bayes （0.03180 > 0.02207）。

但存在一个可部署的次要结论：V_Bayes 依赖 latent/true state，是 oracle 量、不可在线部署；learned 模型只用 label-free 特征，却把当前可部署的 heuristic 的 Spearman 从 -0.2329 提升到 0.1297（Δ=+0.3626，bootstrap CI 显著），且 regret / matched-budget 净 Brier 均优于 heuristic。因此 learned 模型可视为「比 heuristic 强、但未达 oracle」的替代方案；若部署场景以 heuristic 为基线（而非 V_Bayes），则值得作为 heuristic 的升级进入集成。