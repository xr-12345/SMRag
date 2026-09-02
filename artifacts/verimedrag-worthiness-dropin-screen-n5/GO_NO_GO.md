# Phase 6 — GO / NO-GO

裁决：**No-Go**（判据通过 4/6）。

| # | 判据 | 结果 | 说明 |
|---|---|---|---|
| C1 | 隔离不变量：extra_stop==0，learned 不改 AskNew/Stop、不自行打开 VerifyOld | PASS | extra_stop 总数 = 0 |
| C2 | 问题预算：|Δ total_q| ≤ 1.0（无 Prompt #14 塌缩） | PASS | Δq = {'learned_full_rerank': '-0.032', 'learned_full_filter': '-0.067', 'learned_norag_filter': '-0.084'} |
| C3 | rerank 不显著伤精度（Δtop1 CI 不全<0、Δbrier CI 不全>0） | PASS | |
| C4 | rerank 更准地选中真错报告（selected 误报率 ≥ heuristic） | FAIL | heuristic 0.416 vs rerank 0.321 |
| C5 | filter 降低无谓核验且不伤精度 | FAIL | 无谓率 0.584 → 0.627 |
| C6 | full-RAG 不显著差于 no-RAG | PASS | |

结论：机制能跑（rerank 改了 297 次核验对象、filter 替换了 305 次核验），但无诊断增益：rerank 选中报告的真实误报率更低（0.321 vs heuristic 0.416）、NLL 显著变差、无谓核验显著增加、纠错显著减少。
