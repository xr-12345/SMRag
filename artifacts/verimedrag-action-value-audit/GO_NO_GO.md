# Phase 3A — Go / No-Go 裁决

## 裁决：**Conditional Go**

进入 learned verification-worthiness 阶段 **可以**，但**范围必须收窄到 VerifyOld 核验效用**；不建议对 AskNew 侧做「统一价值」改造（EIG 已等价）。

---

## 七项 Go 判据逐条核对

| # | 判据 | 结果 | 证据 |
|---|---|---|---|
| 1 | V_Bayes 与 V_real Spearman 显著 > 0 | ✅ 通过 | combined 0.247 [0.202, 0.298]；AskNew 0.234；VerifyOld 0.210，CI 均 >0 |
| 2 | V_Bayes 相关性高于现有 heuristic | ✅ 通过（VerifyOld）/ ⚠️ 平手（AskNew） | VerifyOld: V_Bayes 0.210 vs heuristic −0.223；AskNew: 0.234 vs EIG 0.241 |
| 3 | V_Bayes top-action regret 低于现有策略 | ✅ 通过（边际） | mean 0.112 vs 0.115；median 0.019 vs 0.030 |
| 4 | V_Bayes 所选动作 realized Brier 收益不低于现状 | ✅ 通过（边际） | net 0.0387 vs 0.0358；gross Brier 0.0687 vs 0.0658 |
| 5 | V_Bayes 正值组 realized 收益明显高于负值组 | ✅ 通过（强） | >0 组 mean V_real +0.037 vs <0 组 −0.024（Δ=0.061）；符号一致率 73.2% |
| 6 | deployable V_Bayes 无 latent-state 泄漏 | ✅ 通过 | 代码审查 + 测试 6/7/10 |
| 7 | 方向在 C∈{0.00,0.03,0.06} 稳定 | ✅ 通过 | 单调下降，V_Bayes/V_real 各档 mean 接近 |

---

## 为什么不是干净的「Go」

判据 3/4 虽通过，但绝对值改善 < 0.005 Brier 单位、相对 ~1–3%，接近 MC 噪声量级；判据 2 在 AskNew 侧为平手。本审计是 **train-only、in-distribution**（model-full.json 在 train 上拟合），存在乐观偏差，按红线不视为正式泛化结果。

## 为什么不是「No-Go」

判据 1（正相关、CI 排除 0）与判据 5（正/负组干净分离 Δ=0.061）都稳健成立；且审计暴露了一个**实质缺陷**：现有 `heuristic_verify_utility` 与 realized 价值**反相关（−0.22）**。V_Bayes 是首个对 VerifyOld 呈正相关的 label-free 信号。这否定了「弱相关即 No-Go」的条件——问题不在 V_Bayes 太弱，而在现有 VerifyOld 启发式是**负**的。

---

## 附条件进入的建议（进入 learned verification-worthiness 时的约束）

1. **目标窄化**：learned worthiness 的回归目标 = realized 核验价值（或 V_Bayes），用于**替换** `_verification_action` 中的 `existing_verification_utility`；不重做 AskNew 排序。
2. **学习「不做」**：VerifyOld 平均 realized 价值为负（−0.024，仅 10.9% 为正），learned 模型应学会在高成本下抑制核验。
3. **不要**把本 train 审计的相关性绝对值当作泛化预期；下一步须在 test split（若后续阶段允许）或新的 held-out 上重验。
4. **红线不变**：不访问 test、不训练最终 value model 于本阶段之外、不修改正式策略默认行为、不 commit。
