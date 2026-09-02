# Phase 5 — Go / No-Go 裁决

## 裁决：**No-Go**

判据通过 3/8。

| # | 判据 | 结果 | 值 |
|---|---|---|---|
| 1 | learned brier < heuristic | ❌ | Δ=+0.2884 CI[+0.2756,+0.3009] |
| 2 | learned top1 不显著更差 | ❌ | Δ=-0.2272 CI[-0.2395,-0.2148] |
| 3 | learned correct→wrong ≤ heuristic | ✅ | 0.0429 vs 0.0379 |
| 4 | learned 问题数 ≤ heuristic | ✅ | Δ=-6.27 |
| 5 | learned ~ V_Bayes（brier 不显著更差） | ✅ | Δ=-0.0191 |
| 6 | harm gate 降低 correct→wrong 率 | ❌ | 0.0429 vs 0.0379 |
| 7 | full-RAG brier ≤ no-RAG | ❌ | Δ=+0.0008 |

## 结论

**No-Go**（通过 3/8）。

learned 模型作为可部署的 VerifyOld 排序升级（比 heuristic 强、但未达 oracle V_Bayes）的证据已在 Phase 4 离线建立；本次在线集成在统一 Brier 单位下使 AskNew 过早停止，问诊问题数显著少于 heuristic，导致 realized Brier/精度未超过旧基线。