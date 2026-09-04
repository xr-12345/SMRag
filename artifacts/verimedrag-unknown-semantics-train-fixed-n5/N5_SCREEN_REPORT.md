# Phase 8E — N=5 筛选报告（UNKNOWN 语义修复 + protocol_fixed 先验）

## 0. 一句话结论

- **UNKNOWN 语义修复：GO** —— `p_wrong` 对 UNKNOWN 返回 `None`，校准不再混入 UNKNOWN，
  策略轨迹与 Phase 8C 逐位一致（1470/1470，0 mismatch），240 测试全绿。
- **protocol_fixed（train_fixed）先验：校准 GO，诊断部署 NO-GO** —— 明确回答上
  `p_wrong` ECE 从 0.0398 → 0.0054（−0.034，CI 不跨 0），`p_mode` ECE 0.0433 → 0.0048；
  但**总问题数 +4.56（+48%，远超 §七「≤0.25」红线）**，Top-1 不升反微降（−0.27pp，不显著）。
  这是一个「calibration fix」，不是诊断性能提升，应保留为可选配置、**不作为默认**。

## 1. 实验规模与冻结

| 项 | 值 |
|---|---|
| 配置 | `joint_legacy_prior` vs `joint_train_fixed_prior`（matched 0/0） |
| 病例 | 245（validate，N=20 manifest 每疾病前 5 例） |
| noise | {0.2, 0.3} |
| seeds | {2026, 2027, 2028} |
| 轨迹数 | 2940（245 × 2 noise × 3 seed × 2 config） |
| 可靠性行 | 32932 |
| 预算/阈值 | 与 Phase 8C 完全一致（`max_total_turns=15`，见 `CONFIG.json`） |
| 运行 | 16 workers，1030.9s，均值 5.57s/对话，峰值 148MB |

## 2. 轨迹回归（§三 硬性要求）

`baseline_regression_check.txt`：Phase 8E legacy 1470 行 vs Phase 8C joint matched-0/0 1470 行，
**0 mismatch，byte-identical**。UNKNOWN 语义修复不改变动作轨迹（原因：`choose_action` 只调
`p_mode_misreported`，从不调 `p_wrong`）。

## 3. 诊断与负担（paired，legacy → train_fixed）

| 指标 | legacy | train_fixed | Δ | Δ% | 95% CI（case-cluster bootstrap） |
|---|---|---|---|---|---|
| Top-1 | 0.78095 | 0.77823 | −0.00272 | −0.35% | [−0.0252, +0.0177] |
| Top-3 | 0.85918 | 0.84082 | −0.01837 | −2.14% | — |
| diagnostic Brier | 0.29145 | 0.27711 | −0.01434 | −4.92% | [−0.0386, +0.0102] |
| NLL | 0.87629 | 0.82372 | −0.05257 | −6.00% | — |
| **总原子问题数** | 9.452 | 14.010 | **+4.558** | **+48.2%** | **[+4.076, +5.048]** |
| AskNew 次数 | 8.960 | 13.443 | +4.483 | +50.0% | — |
| VerifyOld 次数 | 0.493 | 0.567 | +0.075 | +15.2% | — |
| 提前停止率 | 0.540 | 0.103 | −0.437 | −81.0% | — |
| 过早停止率 | 0.0374 | 0.0020 | −0.0354 | −94.5% | — |
| 不必要核验率 | 0.630 | 0.655 | +0.025 | +3.9% | — |
| 冲突解决率 | 0.325 | 0.315 | −0.009 | −2.8% | — |

**机制解读**：protocol_fixed 把 MISREPORTED 先验从 0.03 抬到 0.075，使 `p_mode`
系统性偏高 → Stop 审计更保守 → 提前停止率从 54% 崩到 10%，每对话多问 ~4.6 个问题。
代价是**没有换来精度**（Top-1 平、diagnostic Brier 的 −4.9% 主要来自「多问」而非「答得更准」）。
唯一正面安全信号：过早停止率下降 94.5%（0.037 → 0.002，更少「该继续问却停了」）。

## 4. 可靠度校准（核心结果）

| 信号 | 配置 | 子集 | n | 患病率 | Brier | ECE | 95% CI |
|---|---|---|---|---|---|---|---|
| p_wrong | legacy | explicit（排除 UNKNOWN） | 11584 | 0.0962 | 0.0660 | 0.0398 | [0.0358, 0.0447] |
| p_wrong | train_fixed | explicit（排除 UNKNOWN） | 17402 | 0.0950 | 0.0600 | **0.0054** | [0.0042, 0.0105] |
| p_mode | legacy | all_first | 13171 | 0.0777 | 0.0626 | 0.0433 | [0.0392, 0.0474] |
| p_mode | train_fixed | all_first | 19761 | 0.0759 | 0.0569 | **0.0048** | [0.0031, 0.0094] |
| is_nonresponse | legacy | all_first | 13171 | 0.1205 | — | — | UNKNOWN rate 单独统计 |
| is_nonresponse | train_fixed | all_first | 19761 | 0.1194 | — | — | UNKNOWN rate 单独统计 |

**配对 delta（bootstrap）**：
- `p_wrong` ECE：Δ = −0.0344，CI [−0.0384, −0.0275]（不跨 0，**显著**）
- `p_mode` ECE：Δ = −0.0385，CI [−0.0419, −0.0323]（不跨 0，**显著**）

**UNKNOWN rate 保持不变（~12%）**：这是环境属性（`from_noise_rate` 的 unknown 份额），
UNKNOWN 语义修复只改变日志/校准口径，不改变环境生成。两次配置的 UNKNOWN rate 一致
（12.05% vs 11.94%），与预期吻合。

## 5. 分层（p_wrong ECE）

| 分层 | legacy ECE | train_fixed ECE |
|---|---|---|
| value=PRESENT | 0.1066 | 0.0176 |
| value=ABSENT | 0.0142 | 0.0050 |
| value=categorical | 0.0425 | 0.0112 |
| noise=0.2 | 0.0285 | 0.0120 |
| noise=0.3 | 0.0521 | 0.0148 |

- **两个噪声方向一致**：noise=0.2 与 0.3 均显著改善（0.029→0.012；0.052→0.015）——满足 §七「方向一致」。
- **PRESENT 残差最大**：train_fixed 后 PRESENT 层 ECE 仍为 0.018（全层最高），因 PRESENT 是
  罕见阳性类（explicit 中患病率 ~0.20–0.27），其 `p_wrong` 天然偏高、更难校准。这是
  asymmetric-calibration 的已知模式，非 protocol_fixed 能完全消除，报告中如实标注。

## 6. Go/No-Go 判定摘要（详见 GO_NO_GO.md）

- **UNKNOWN 语义修复：GO**（4/4 条件全满足）。
- **protocol_fixed 先验：校准 GO / 诊断部署 NO-GO**（6 项中第 4 项「总问题数 ≤+0.25」
  以 +4.56 严重超标；Top-1 未改善）。定性为 **calibration fix**，保留为可选配置，
  不进入 N=20 作为默认部署。

## 7. 对 §十 七个问题的直接回答

见 `GO_NO_GO.md` 与最终汇报（终端输出）。
