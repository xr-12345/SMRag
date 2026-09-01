# Phase 2C 冻结配置 N=20 确认实验 —— Go / No-Go 判定

**判定：No-Go**

**理由：Brier 改善未达统计显著（95% CI 含 0，跨 seed 不稳定），Phase-2B 的 Brier 优势未在确认集复现。**


判断对象：735 例主确认子集（排除 245 例筛选病例，overlap=0），`joint_gate_w1.0` vs 基线 `rank_only_w0.0`，pooled noise 0.2/0.3。


## 七项判据逐条结论

| # | 判据 | 结果 | 证据 |
|---|------|------|------|
| 1 | Brier paired Δ < 0 且 95% CI 上界 < 0 | ❌ 不通过 | Brier Δ=-0.00306 95% CI [-0.00990, 0.00395] |
| 2 | Top-1 非劣：paired Δ 95% CI 下界 ≥ −0.005 | ✅ 通过 | Top-1 Δ=0.00317 95% CI lower=-0.00227 |
| 3 | 平均总原子问题数增加 ≤ 0.25 | ✅ 通过 | total atomic questions Δ=0.072 |
| 4 | 不必要核验率方向 = 下降 | ✅ 通过 | unnecessary-verification rate Δ=-0.0422 |
| 5 | 冲突解决率方向 = 上升 | ✅ 通过 | conflict-resolution rate Δ=0.0374 |
| 6 | 触发核验精度明显高于候选误报基准率 | ✅ 通过 | triggered precision 61.7% (Wilson [57.6%, 65.7%]) vs base rate 7.7%; enrichment 8.0228x; n_triggered=554 |
| 7 | ≥1 项主要临床/安全终点改善，且非仅靠更多问题 | ✅ 通过 | improving endpoints: ['Top-1', 'Brier', 'Top-3', 'NLL', 'ECE', 'premature-stop', 'unnecessary-verification rate', 'conflict-resolution rate'] |

## 红线合规确认

- ✅ 仅 validate 分支；未访问 DDXPlus test。
- ✅ 未重新调权重/阈值（配置冻结，见 CONFIG.json）。
- ✅ 未加入 LLM / dense retriever / SFT / RL。
- ✅ 未修改既有正式结果；未覆盖 Phase 2A/2B artifacts。
- ✅ 未修改 preregistered success criteria。
- ✅ full N=20 仅作为补充，未描述为完全独立确认集。
- ✅ 评估真值仅用于 eval 侧统计，未进入动作决策路径。
- ✅ 未 git commit。
- ✅ 未根据结果改变 Go/No-Go 门槛。

## 最终决定

**No-Go。** Brier 改善未达统计显著（95% CI 含 0，跨 seed 不稳定），Phase-2B 的 Brier 优势未在确认集复现
