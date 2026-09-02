# Phase 5 — Learned Verification-Worthiness 在线集成与 N=5 筛选报告

裁决：**No-Go**（判据通过 3/8，细节见 `GO_NO_GO.md`）。

本阶段把 Phase 4 冻结的 learned verification-worthiness 模型（hgb gain + hgb harm，tau_harm=0.4129）接入完整 AskNew/VerifyOld/Stop 问诊策略，在 245 例 × 2 噪声 × 3 种子 × 4 策略 = 5,880 条真实问诊轨迹上做 N=5 筛选。四策略：heuristic_verify（旧基线）、model_based_vbayes_verify（V_Bayes 参考）、learned_worthiness_full_rag、learned_worthiness_no_rag。

## 1. Learned 是否优于 heuristic

| strategy | top1 | top3 | brier | new | verify | total q | c→w/verify | wall(s) |
|---|---|---|---|---|---|---|---|---|
| heuristic | 0.7721 | 0.8660 | 0.3098 | 8.9415 | 0.6109 | 9.5524 | 0.0379 | 9.1208 |
| v_bayes | 0.5286 | 0.6204 | 0.6173 | 2.9905 | 0.1306 | 3.1211 | 0.0052 | 0.6979 |
| learned_full | 0.5449 | 0.6374 | 0.5982 | 3.0177 | 0.2694 | 3.2871 | 0.0429 | 0.9178 |
| learned_norag | 0.5435 | 0.6374 | 0.5974 | 3.0014 | 0.3075 | 3.3089 | 0.0442 | 0.5532 |

- learned_full vs heuristic 配对 Δbrier = +0.2884 (95% CI [+0.2756, +0.3009], p=0.000)（显著）
- learned_full vs heuristic 配对 Δtop1 = -0.2272 (95% CI [-0.2395, -0.2148], p=0.000)（显著）
- learned_full vs heuristic 配对 Δ问题数 = -6.27 (95% CI [-6.41, -6.12])
- learned_full vs heuristic 配对 Δcorrect→wrong = -0.0116 (95% CI [-0.0153, -0.0080])

结论：**否**（learned 未在 realized Brier/精度上超过 heuristic）。

原因：learned 与 v_bayes 均采用统一的 Brier 单位计价 AskNew（V_Bayes_new = R(b_t) − E[R(b_{t+1})] − C_new，C_new=0.03），在扩散先验下 V_Bayes_new 很快 ≤0，策略过早停止（~3.3 问 vs heuristic ~9.6 问）。该精度损失来自**共享的 AskNew 计价方式**，而非 learned 核验模型本身：learned_full 与 v_bayes 的问题数/精度几乎一致（3.29 vs 3.12 问，0.545 vs 0.529 top1）。

## 2. Learned 与 V_Bayes 效果与速度差距

- 效果：learned_full brier = 0.5982、learned_norag = 0.5974，V_Bayes = 0.6173（learned 略优于 V_Bayes，Δ ≈ -0.0191）。
- 速度：learned_norag = 0.55s vs V_Bayes = 0.70s（同为 NO_RAG，learned 快 ≈ 1.3×）；learned_full = 0.92s 额外含 DYNAMIC_RAG 检索成本（比 NO_RAG 多 ~0.36s），并非 worthiness 模型本身更慢。
- learned_full vs V_Bayes 配对 Δbrier = -0.0191 (95% CI [-0.0253, -0.0131], p=0.000)（显著）

## 3. harm gate 是否降低 correct→wrong

| strategy | correct→wrong / verification |
|---|---|
| heuristic | 0.0379 |
| v_bayes | 0.0052 |
| learned_full | 0.0429 |
| learned_norag | 0.0442 |

- learned_full: gain 通过 935，harm gate 额外拦截 90 个（mean harm_hat 0.65636），实际选择 396 个，selected 中 realized correct→wrong 17 个（率 0.042929）
- learned_norag: gain 通过 1015，harm gate 额外拦截 14 个（mean harm_hat 0.841582），实际选择 452 个，selected 中 realized correct→wrong 20 个（率 0.044248）

结论：**否**。harm gate 机械性地拦截了高 harm_hat 候选（learned_full 额外拦截 90 个，mean harm_hat ≈ 0.66），因此总 correct→wrong 计数下降（learned_full 17 vs heuristic 34）；但**每次核验**的 c→w 率 learned_full = 0.0429 并不低于 heuristic = 0.0379。total c→w 的下降主要由核验次数本身变少（0.269 vs 0.611 次）驱动，而非 harm gate 把「会把对的改错」的核验筛选掉了——单次核验的风险率并没有改善。

## 4. 是否增加问题数

- heuristic: 平均 8.9415 新问题 + 0.6109 核验 = 9.5524 问题
- v_bayes: 平均 2.9905 新问题 + 0.1306 核验 = 3.1211 问题
- learned_full: 平均 3.0177 新问题 + 0.2694 核验 = 3.2871 问题
- learned_norag: 平均 3.0014 新问题 + 0.3075 核验 = 3.3089 问题

## 5. Full-RAG 是否优于 No-RAG

| metric | full-RAG | no-RAG | Δ |
|---|---|---|---|
| correct_top1 | 0.544898 | 0.543537 | 0.001361 |
| correct_top3 | 0.637415 | 0.637415 | 0.0 |
| brier_score | 0.598215 | 0.597382 | 0.000833 |
| new_questions | 3.017687 | 3.001361 | 0.016327 |
| verification_questions | 0.269388 | 0.307483 | -0.038095 |
| total_atomic_questions | 3.287075 | 3.308844 | -0.021769 |
| correct_to_wrong_flips | 0.011565 | 0.013605 | -0.002041 |
| wrong_to_correct_flips | 0.022449 | 0.022449 | 0.0 |
| unnecessary_verifications | 0.182313 | 0.211565 | -0.029252 |
| resolved_wrong_reports | 0.07551 | 0.084354 | -0.008844 |

## 6. 是否存在泄漏

无。预测路径不读 true disease / latent state / true wrongness / noise label / oracle correction（测试 08、10）；N=5 只取 validate 前 5 例，不触 test split；learned 模型为 Phase 4 冻结产物，本阶段不重新训练、不调整 tau_harm；默认策略仍为 heuristic（测试 13）。

## 7. Go / Conditional Go / No-Go

**No-Go**（判据通过 3/8，见 `GO_NO_GO.md`）。

## 8. 是否值得进入 N=20

**不进入 N=20**：learned 在 N=5 上未在 realized Brier/精度上超过 heuristic（主要原因：统一 Brier 单位下 AskNew 的 V_Bayes_new 在 C_new=0.03 时过早停止，问诊问题数显著少于 heuristic），作为可部署的 VerifyOld 排序升级的证据已在 Phase 4 建立，但完整在线集成不构成对旧 heuristic 的改进。