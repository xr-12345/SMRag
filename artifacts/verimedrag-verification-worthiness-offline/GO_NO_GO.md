# Phase 4 — Go / No-Go 裁决

## 裁决：**No-Go**

判据通过 8/10。

| # | 判据 | 结果 | 值 |
|---|---|---|---|
| 1 | learned Spearman > V_Bayes | ❌ | 0.1297 vs 0.2221 |
| 2 | regret < V_Bayes 且 < heuristic | ✅ | 0.0852 vs 0.1002/0.0905 |
| 3 | matched net Brier > V_Bayes/heuristic | ✅ | 0.0129 vs -0.0056/0.0064 |
| 4 | correct→wrong ≤ V_Bayes | ❌ | 0.03180 vs 0.02207 |
| 5 | harm head 减少有害核验 | ✅ | Δctw=+0.00354 |
| 6 | full RAG > no-RAG 且 > shuffled | ✅ | 0.1297 vs 0.1169/0.1234 |
| 7 | 95% CI 支持 ≥1 主要优势 | ✅ | 2 个显著差异 |

## 结论

**No-Go**（通过 8/10）。

learned 模型在 Spearman 排序上未超过 V_Bayes（oracle 基线），且 correct→wrong 高于 V_Bayes，故**不进入**在线 Joint Policy 集成。

次要结论（可部署性）：V_Bayes 依赖 latent/true state、不可在线部署；learned 只用 label-free 特征，将可部署 heuristic 的 Spearman 从 -0.2329 提升到 0.1297（显著），regret 与 matched-budget 净 Brier 亦优于 heuristic。若基线是 heuristic 而非 V_Bayes，则可作为其升级进入集成。