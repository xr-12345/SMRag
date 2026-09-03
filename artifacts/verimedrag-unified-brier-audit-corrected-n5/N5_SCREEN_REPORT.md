# Prompt #18 — N=5 验证筛选报告

范围：DDXPlus validate split，245 例（5/病）× 2 噪声 {0.2, 0.3} × 3 seed
{2026, 2027, 2028} × 5 策略 = **7350 轨迹**。冻结参数 τ_V = τ_A = 0.03。

## 1. 结论

**No-Go（不可部署）。** 但结构修正**正确且已验证**：强制核验被消除、
基线回归 byte-identical、无泄漏、unnecessary-verify 相对 forced 下降 3×。
修正后的 unified-Brier 策略相对启发式基线仍损失 top1 −4.0pp、Brier 相对 +17.6%，
未达到部署判据。

## 2. 主表（noise 0.2，主验证点）

| 策略 | top1 | top3 | Brier | NLL | new_q | verify_q | total_q | unnec_verify | unnec_rate | uncertain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| heuristic_baseline | 0.7932 | 0.8816 | 0.2825 | 0.8136 | 8.603 | 0.559 | 9.162 | 0.340 | 0.608 | 0.010 |
| model_based_v_bayes | 0.5592 | 0.6422 | 0.5774 | 1.9671 | 3.004 | 0.135 | 3.139 | 0.103 | 0.768 | 0.576 |
| forced_verify (ablation) | 0.7510 | 0.8218 | 0.3475 | 1.1132 | 7.935 | 0.173 | 8.107 | 0.102 | 0.591 | 0.023 |
| **corrected** | **0.7578** | **0.8259** | **0.3321** | **1.0645** | **8.248** | **0.084** | **8.332** | **0.033** | **0.387** | **0.000** |
| corrected_dynamic_rag | 0.7578 | 0.8259 | 0.3321 | 1.0645 | 8.248 | 0.084 | 8.332 | 0.033 | 0.387 | 0.000 |

（`corrected_dynamic_rag` 与 `corrected` 逐轨迹 byte-identical：RAG 不进 Brier 价值。）

## 3. 机制修正验证（核心消融 corrected vs forced）

| 指标 | Δ (corrected − forced) |
| --- | ---: |
| top1 | **+0.0034**（无精度损失） |
| Brier | **−0.0086**（略优） |
| verification_questions | **−0.0966**（核验减半） |
| unnecessary_verifications | **−0.0762**（3× 下降） |
| total_atomic_questions | +0.2109（预算转向 AskNew） |

**结论**：`G_t^verify > τ_V` 不再强制 VerifyOld。修正后策略在 gross-gain 触发时，
把原本的「强制核验」转向「与 AskNew 统一比较」，于是 verify 次数减半、
unnecessary-verify 3× 下降，且 top1/Brier 相对 forced **没有损失（微增）**。
`stop_blocked_count`（均值 0.079）表明 Stop 审计仍会触发并阻止 Stop，但不再直接核验。

## 4. 部署判据（corrected vs heuristic，主验证点）

| 指标 | heuristic | corrected | Δ | 判据 | 结果 |
| --- | ---: | ---: | ---: | --- | --- |
| top1 | 0.7932 | 0.7578 | **−0.0354** | \|Δ\| ≤ 0.005 | **FAIL** |
| Brier | 0.2825 | 0.3321 | **+17.6%** rel | ≤ 5% | **FAIL** |
| total_q | 9.162 | 8.332 | −0.83 | 不塌缩(>1.0) | clear |

启发式基线用更多问题与核验换取更高 top1 与更低 Brier。unified-Brier 价值模型
（即便修正后）在 DDXPlus 诊断任务上仍**不优于启发式**。

## 5. Go / No-Go 判据汇总

**Go（6 项）**：forced-verify 消除 PASS、基线回归 PASS、无泄漏 PASS、
top1-drop ≤0.005 **FAIL**、Brier 相对 ≤5% **FAIL**、减少 unnecessary-verify **PASS**。
→ 6 中 4 过、2 不过。

**No-Go（3 项）**：问题塌缩 clear、被启发式支配 clear（q 更低故不构成严格支配）、
RAG 无增益 **TRIGGERED**（0/1470 轨迹变化）。

## 6. 基线回归

`heuristic_baseline` 与冻结 drop-in 启发式（Phase 15）逐轨迹比较：
**0 不匹配 / 0 缺失**（byte-identical，7 列）。确认无回归、无泄漏。

## 7. 红线合规

* 无 test split、无 retrain、无 threshold sweep、无特征/模型改动。
* 预测路径不读 latent / 真值疾病 / noise / 真错误 / 干净答案 / oracle（test_13）。
* RAG 保留但检索 Jaccard 永不进入 Brier 价值（test_14，0/1470 轨迹变化）。
* 默认策略不变（heuristic_verify byte-identical）；新策略 opt-in。
* 未 git commit。

## 8. 文件清单

`CORRECTION_REPORT.md`、`N5_SCREEN_REPORT.md`、`GO_NO_GO.md`、`CONFIG.json`、
`ENVIRONMENT.txt`、`TEST_LOG.txt`、`FAILURE_LOG.md`、`LEAKAGE_AUDIT.md`、
`baseline_regression_check.txt`、`case_outcomes.csv`(7350)、
`turn_action_values.csv`(25318)、`stop_audit_events.csv`(3672)、
`summary_by_strategy_noise.csv`、`paired_deltas.csv`(7350)、`git_diff.txt`。
