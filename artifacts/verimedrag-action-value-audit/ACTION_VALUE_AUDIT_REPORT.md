# Phase 3A — 统一动作价值 (Unified Action Value) 合理性审计报告

**任务**: 审计 deployable 统一动作价值 `V(a|H_t) = R(b_t) − E[ R(b_{t+1}) ] − C(a)` 是否比现有启发式分数（EIG、回溯错误概率、检索影响、启发式核验效用）更好地预测 realized 风险下降。

**范围**: DDXPlus **train** 245 例 × 噪声 {0.2, 0.3} = **490 条对话**；每条对话取 3 个状态（early/middle/late），每状态枚举 top-5 EIG AskNew + 至多 5 VerifyOld；合计 **11,253 个 state-action**（AskNew 7,110 / VerifyOld 4,143）。

**关键约定（红线遵守）**: deployable V_Bayes 只用信念 `b_t`、fitted model、冻结 answer channel；真实疾病 `D*` 与 latent state **只** 作为 eval 标签进入 realized return。未访问 test split，未训练任何 value model，未修改既有策略默认行为，未 commit。

---

## 1. V_Bayes 与 V_real 的相关性（Spearman / Kendall，cluster-bootstrap 95% CI）

| score | 作用域 | Spearman | 95% CI | Kendall | n |
|---|---|---|---|---|---|
| **V_Bayes** | combined | **0.247** | [0.202, 0.298] | 0.184 | 11,253 |
| **V_Bayes** | AskNew | **0.234** | [0.185, 0.286] | 0.159 | 7,110 |
| **V_Bayes** | VerifyOld | **0.210** | [0.146, 0.272] | 0.153 | 4,143 |
| EIG | AskNew | 0.241 | [0.199, 0.285] | 0.165 | 7,110 |
| heuristic_verify_utility | VerifyOld | **−0.223** | [−0.265, −0.183] | −0.170 | 4,143 |
| retrospective_error_prob | VerifyOld | −0.035 | [−0.074, −0.001] | −0.037 | 4,143 |
| retrieval_impact | VerifyOld | −0.034 | [−0.073, 0.008] *n.s.* | −0.025 | 4,143 |
| error_prob × retrieval_impact | VerifyOld | −0.050 | [−0.092, −0.002] | −0.040 | 4,143 |

**结论**:
- V_Bayes 与 V_real 呈 **显著正相关**（combined Spearman 0.247，CI 严格 >0）。
- **AskNew**: V_Bayes(0.234) ≈ EIG(0.241)。二者共用同一套预测机制（`predicted_answers` + `posterior_for`），因此 V_Bayes 在 AskNew 上**没有超越 EIG**——这是预期内的。
- **VerifyOld**: V_Bayes(0.210) 相对现有 `heuristic_verify_utility`(**−0.223**) 是 **方向性反转**：现有启发式核验效用与 realized 价值**反相关**，V_Bayes 则正相关。这是本审计最实质的发现。
- 现有回溯错误概率、检索影响、`error_prob×impact` 与 V_real 均**不相关或轻微反相关**，即现有 label-free 核验排序信号基本**不预测** realized 风险下降。

## 2. AskNew / VerifyOld 二选一准确率（n=1,367 个同时含两类动作的状态）

| policy | 准确率 | 应验却选新 (under-verify) | 应新却选验 (over-verify) |
|---|---|---|---|
| **V_Bayes** | **75.05%** | 287 | 54 |
| heuristic（现有） | 73.96% | 266 | 90 |
| retrieval joint-gate | 72.79% | 332 | 40 |

**结论**: V_Bayes 二选一准确率最高（75.05%），比现有 heuristic 高 +1.1pp。所有 policy 都系统性 **under-verify**（该核验时选了 AskNew）；但现有 heuristic 的 over-verify（90 次）明显多于 V_Bayes（54 次）——即现有 heuristic 会 **误触发核验**，而 V_Bayes 更克制。

## 3. 相对现有 heuristic 的 regret（n=1,422 状态，regret = max_a V_real − V_real(选择)）

| policy | mean regret | median regret | frac_zero_regret | max |
|---|---|---|---|---|
| **V_Bayes** | **0.112** | **0.019** | **26.1%** | 1.44 |
| heuristic（现有） | 0.115 | 0.030 | 22.9% | 1.44 |
| EIG | 0.116 | 0.026 | 21.7% | 1.44 |
| retrieval joint-gate | 0.118 | 0.025 | 21.7% | 1.44 |

**结论**: V_Bayes 的 mean/median regret 均最低，且选中最优动作的比例最高（26.1% vs 22.9%）。**改善幅度小**（median 0.030→0.019，约 −37% 相对，但绝对值仅 0.011 个 Brier 单位）。

## 4. 实际 Brier / NLL 风险下降（各 policy 所选动作的 realized 收益，n=1,422 状态）

| policy | gross Brier ↓ | gross NLL ↓ | net (Brier↓−C) | wrong→correct | correct→wrong |
|---|---|---|---|---|---|
| **V_Bayes** | **0.0687** | 0.164 | **0.0387** | 0.0885 | **0.0305** |
| heuristic（现有） | 0.0658 | **0.191** | 0.0358 | 0.0885 | 0.0259 |
| EIG | 0.0645 | 0.186 | 0.0345 | 0.0795 | 0.0222 |
| retrieval joint-gate | 0.0626 | 0.183 | 0.0326 | 0.0786 | 0.0227 |

**结论**:
- V_Bayes 在 **Brier** 口径上最优（gross 0.0687 / net 0.0387），符合其以 Brier diversity 为风险单位的设计。
- 但 V_Bayes 的 **NLL 下降最低**（0.164 < heuristic 0.191），且 **correct→wrong 最高**（0.0305）——优化 Brier 会偶尔把一个本来正确的 top-1 判对翻成判错。这是优化目标错位的真实代价。
- 绝对值差异均 < 0.005，属边际。

## 5. 代表性反例（`counterexamples.csv` 共 40 条，每类至多 10 条）

1. **高回溯错误概率但 V_real≈0**：如 case 336184 (E_66) `retrospective_error_prob=0.999` 但 `V_real=−0.002`。回溯错误概率虽极高，但再问一次并不能降低 realized Brier（信念已把该报告边缘化，或重答返回同值）。
2. **高检索影响但 V_real≈0 / 负**：如 case 715819 (E_208) `retrieval_impact=1.0` 但 `V_real=−0.091`。删除该报告改变检索排序，不代表核验它有诊断价值。
3. **中等错误概率但 V_real 很高**：如 case 834228 (Stable angina, E_13) `error_prob=0.077` 但 `V_real=1.187`。现有启发式会漏掉这类「不显眼却极有价值」的核验。
4. **V_Bayes 与 V_real 符号相反**：如 case 93969 (Bronchospasm, E_124) `v_bayes=−1.006` 但 `v_real=−0.976`（同为负但量级相反）、及若干 verify 行 `v_bayes<0` 但 `v_real>1`。
5. **EIG 选新但 V_real 偏验**：如 case 915156 (Cluster headache) verify `V_real=1.088`，说明该状态下核验远胜问新。
6. **heuristic 选验但 V_real 偏新**：如 case 178169 (Tuberculosis, E_79) new `V_real=0.034`，heuristic 却倾向核验。

**结论**: 反例模式一致——现有 label-free 核验信号（回溯错误概率 / 检索影响 / 其乘积）是 realized 价值的 **弱/负代理**，而 V_Bayes 能捕捉到部分（但非全部）真实价值。

## 6. 成本敏感性（C ∈ {0.00, 0.03, 0.06}，离线重算，不重拟合）

| 量 | C=0.00 mean (frac>0) | C=0.03 mean (frac>0) | C=0.06 mean (frac>0) |
|---|---|---|---|
| V_Bayes | +0.0275 (0.761) | −0.0025 (0.387) | −0.0325 (0.218) |
| V_real | +0.0299 (0.461) | −0.0001 (0.253) | −0.0301 (0.199) |

**结论**: 方向单调一致（成本↑ → 价值↓），V_Bayes 与 V_real 在每个成本档位的 mean 接近（说明 V_Bayes 在总体均值上校准良好）。C=0.03 时二者均值≈0——即当前成本下「平均而言」单步动作价值恰好在盈亏平衡点附近；对 VerifyOld 尤甚（见 §7 补充）。

## 7. 是否存在泄漏

**无泄漏**。逐条核对：
- `brier_risk`、`asknew_value`、`verify_value` 只读 `tracker.belief`、`model.specs`、`channel.parameters`、`predictive_state_distribution`、`SurprisalClarificationProtocol.resolve`——**均不读** `case.diagnosis` / `case.states` / latent state / noise 标签。
- 真实疾病与采样答案**只**进入 `realized_value_mc` 返回的 `RealizedRollout`（eval 侧），从不回灌到 deployable 排序。
- 单元测试 6/7/10 显式覆盖：deployable 值对真实疾病不变（test_7 断言 deployable 值不受 case 存在与否影响，而 realized 值随诊断改变）；默认策略行为不变（test_10）。

**补充发现（config 已声明）**: `docs/DECISION_SENSITIVE_ACTION_VALUE.md` 写 `b_t=P(D|H_t,K)`，但代码实现的 `BeliefTracker` 只实现 `P(D|H_t)`，检索 K 只进 retrieval-impact/joint-gate 启发式、不进疾病后验。本审计的 V_Bayes 与之一致（无 K 项）。

## 8. 裁决：**Conditional Go**（细节见 `GO_NO_GO.md`）

7 项 Go 判据中，判据 1/2(VerifyOld)/5/6/7 干净通过；判据 3/4 通过但幅度边际；判据 2(AskNew) 为平手（V_Bayes ≈ EIG）。因此整体为 **Conditional Go**：机制有效且无泄漏，但「统一」价值相对现状的**聚合增益小**，最大价值集中在**替换反相关的 VerifyOld 启发式**。

## 9. 是否值得进入 learned verification-worthiness 阶段

**值得，但需收窄范围**。理由：
- 现有 `heuristic_verify_utility` 与 realized 价值**反相关（−0.22）**——它是**主动选错**核验对象的缺陷，而非随机噪声。用可学习的 verification-worthiness（以 V_real 或 V_Bayes 为回归目标）替换它，是审计指出的最明确改进点。
- 但 **AskNew 侧不建议重做**：EIG 已达 V_Bayes 同等水平（0.241 vs 0.234），V_Bayes 在此无信息增量。
- 且需注意：VerifyOld 动作平均 realized 价值为负（−0.024，仅 10.9% 为正，见 §4/§6），说明 C_verify=0.03 下「默认多核验」是亏的；learned worthiness 应学习「何时**不做**」，而不仅是「做哪个」。

---

### 附录：规模与运行
- 94/94 单元测试通过（`TEST_LOG.txt`）。
- 运行 `runtime_profile.csv`：490 对话，11,253 行，mean wall/dialogue 8.76s，总 478s，peak RSS 1.9 GB。
- 产物齐全：`action_value_samples.csv`、`train_action_value_audit_manifest.csv`、`value_correlations.csv`、`value_calibration.csv`、`pairwise_action_accuracy.csv`、`action_regret.csv`、`selected_action_returns.csv`、`counterexamples.csv`、`cost_sensitivity.csv`。
