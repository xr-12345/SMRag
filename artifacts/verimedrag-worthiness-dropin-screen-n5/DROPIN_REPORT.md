# Phase 6 — VerifyOld Drop-in 隔离筛选报告 (N=5)

裁决：**No-Go**（判据通过 4/6，见 `GO_NO_GO.md`）。

本阶段把 Phase 4 冻结的 learned verification-worthiness 模型只作为 VerifyOld 的**重排器/过滤器**接入，**完全不改动 AskNew 与 Stop**（AskNew 的 EIG 排序、Stop 的 posterior/margin/min-utility 阈值均原样）。控制器（旧 heuristic）先决定动作类型；learned 仅在 controller=VerifyOld 时，在同一批候选里 re-rank（harm gate + max net_value）或把无价值核验替换为 AskNew（never Stop）。在 245 例 × 2 噪声 × 3 种子 × 4 策略 = 5,880 条轨迹上做 N=5 筛选。

## 0. 四策略总览（pooled over noise×seed）

| strategy | top1 | top3 | brier | nll | new | verify | total q | c→w/verify | premature stop |
|---|---|---|---|---|---|---|---|---|---|
| heuristic_baseline | 0.772 | 0.866 | 0.310 | 0.911 | 8.941 | 0.611 | 9.552 | 0.038 | 0.041 |
| learned_full_rerank | 0.766 | 0.858 | 0.319 | 0.953 | 8.910 | 0.611 | 9.520 | 0.030 | 0.046 |
| learned_full_filter | 0.769 | 0.856 | 0.316 | 0.946 | 8.922 | 0.563 | 9.485 | 0.031 | 0.046 |
| learned_norag_filter | 0.773 | 0.854 | 0.310 | 0.955 | 9.032 | 0.436 | 9.468 | 0.039 | 0.044 |

## 1. 提前停止混杂是否消除

heuristic 平均问题数 = 9.552，learned 三策略 = 9.520 / 9.485 / 9.468。Δ问题数 = {'learned_full_rerank': '-0.032', 'learned_full_filter': '-0.067', 'learned_norag_filter': '-0.084'}，不再出现 Prompt #14 的 ~9.5 → ~3.3 塌缩。

**是，混杂已消除**：AskNew/Stop 完全由旧 heuristic 决定，learned 只作用于 VerifyOld 的目标选择，未使用统一 Brier 值给 AskNew 定价。

## 2. Rerank 是否优于 heuristic

| metric | heuristic | rerank | Δ | 95% CI | p |
|---|---|---|---|---|---|
| Brier | 0.310 | 0.319 | 0.009 | [-0.002, 0.021] | 0.117 |
| Top-1 acc | 0.772 | 0.766 | -0.006 | [-0.016, 0.004] | 0.225 |
| Top-3 acc | 0.866 | 0.858 | -0.008 | [-0.017, 0.000] | 0.0641 |
| NLL | 0.911 | 0.953 | 0.042 | [0.007, 0.078] | 0.0181 |
| total atomic questions | 9.552 | 9.520 | -0.032 | [-0.082, 0.018] | 0.211 |
| VerifyOld count | 0.611 | 0.611 | 0.000 | [0.000, 0.000] | 1 |

rerank 在 898 次核验机会中改变了 297 次 report_index（33.073%）。selected 报告真实误报率：heuristic = 0.416，rerank = 0.321（rerank 反而选中更安全、更少真错的报告）。
显著性：ΔNLL +0.042 (p=0.018)、Δ无谓核验 +0.059 (p<1e-13)、Δ纠错 -0.056 (p<1e-13)、Δwrong→correct -0.010 (p=0.007)，均对 learned 不利；Δc→w -0.005 (p=0.090) 不显著。

## 3. Filter 是否优于 heuristic

| metric | heuristic | filter | Δ | 95% CI | p |
|---|---|---|---|---|---|
| Brier | 0.310 | 0.316 | 0.006 | [-0.006, 0.018] | 0.319 |
| Top-1 acc | 0.772 | 0.769 | -0.003 | [-0.012, 0.007] | 0.579 |
| Top-3 acc | 0.866 | 0.856 | -0.010 | [-0.018, -0.001] | 0.0268 |
| NLL | 0.911 | 0.946 | 0.035 | [-0.002, 0.072] | 0.0609 |
| total atomic questions | 9.552 | 9.485 | -0.067 | [-0.118, -0.016] | 0.00965 |
| VerifyOld count | 0.611 | 0.563 | -0.048 | [-0.059, -0.037] | 2.57e-17 |

filter 在 1133 次核验机会中把 305 次核验替换为 AskNew（26.920%），从未因此 Stop。
显著性：Δtop3 -0.010 (p=0.027)、Δ纠错 -0.040 (p<1e-8) 对 learned 不利；Δ无谓核验 -0.003 (p=0.704) 不显著——即 filter 降低了核验次数，但剩余核验的无谓率并未改善。

## 4. 是否降低无谓或有害核验

| strategy | verify 机会 | 执行核验 | 替换为AskNew | 无谓率 | 纠错率 | c→w/verify | selected误报率 |
|---|---|---|---|---|---|---|---|
| heuristic_baseline | 898 | 898 | 0 | 0.584 | 0.381 | 0.038 | 0.416 |
| learned_full_rerank | 898 | 898 | 0 | 0.679 | 0.290 | 0.030 | 0.321 |
| learned_full_filter | 1133 | 828 | 305 | 0.627 | 0.342 | 0.031 | 0.373 |
| learned_norag_filter | 845 | 641 | 204 | 0.651 | 0.326 | 0.039 | 0.349 |

## 5. 是否保持相同问题预算

heuristic: 8.941 AskNew + 0.611 VerifyOld = 9.552 问题。learned 三策略 total q = 9.520 / 9.485 / 9.468，预算保持一致（filter 用 AskNew 替换核验，因此 verify 略降、new 略升）。

## 6. Full-RAG 是否优于 No-RAG

| metric | no-RAG filter | full-RAG filter | Δ | 95% CI | p |
|---|---|---|---|---|---|
| Brier | 0.310 | 0.316 | 0.005 | [-0.009, 0.020] | 0.456 |
| Top-1 acc | 0.773 | 0.769 | -0.003 | [-0.015, 0.008] | 0.569 |
| Top-3 acc | 0.854 | 0.856 | 0.003 | [-0.007, 0.012] | 0.579 |
| NLL | 0.955 | 0.946 | -0.009 | [-0.051, 0.033] | 0.684 |
| total atomic questions | 9.468 | 9.485 | 0.017 | [-0.036, 0.070] | 0.526 |
| VerifyOld count | 0.436 | 0.563 | 0.127 | [0.109, 0.146] | 1.04e-39 |

## 7. 是否存在泄漏

无。预测路径不读 true disease / latent state / true wrongness / noise label / oracle correction（测试 11）；N=5 只取 validate 前 5 例，不触 test split；learned 模型为 Phase 4 冻结产物，本阶段不重训练、不调 tau_harm；默认策略仍为 heuristic_baseline（测试 13）；extra_stop 总数 = 0（learned 从未自行打开 Stop）。

## 8. Go / Conditional Go / No-Go

**No-Go**（判据通过 4/6）。机制能跑（rerank 改了 297 次核验对象、filter 替换了 305 次核验），但无诊断增益：rerank 选中报告的真实误报率更低（0.321 vs heuristic 0.416）、NLL 显著变差、无谓核验显著增加、纠错显著减少。

## 9. 是否值得进入 N=20

**否**。机制能跑（rerank 改了 297 次核验对象、filter 替换了 305 次核验），但无诊断增益：rerank 选中报告的真实误报率更低（0.321 vs heuristic 0.416）、NLL 显著变差、无谓核验显著增加、纠错显著减少。
