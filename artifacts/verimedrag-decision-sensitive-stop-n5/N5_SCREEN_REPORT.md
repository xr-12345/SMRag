# Phase 22B — Decision-Sensitive Stop 四组 N=5 筛选报告

**结论（先给）**: `CONDITIONAL GO`。决策价值 Stop（P-V）显著把 protocol_fixed 的
问题数从 14.01 压回 8.88（−5.13 题，CI [4.63, 5.58]），**保留** protocol_fixed 的
校准（p_wrong ECE 0.009 vs 0.040），且无强制核验。但相对部署基线 L-H，P-V 的
Top-1 点估计 −1.22pp **超过 0.5pp 预注册阈值**（CI 跨 0，N=5 无法判定显著性），
需 N=20 确认。

---

## 1. 四组核心结果（all-noise 均值，N=1470/组）

| 策略 | 先验 | Stop 审计 | Top-1 | Top-3 | Brier | NLL | 总题数 | AskNew | VerifyOld | 早停率 | 早停错误率 | p_wrong ECE | p_mode ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **L-H** | legacy | hard gate | 0.7810 | 0.8592 | 0.2914 | 0.8763 | **9.45** | 8.96 | 0.493 | 0.540 | 0.0374 | 0.0398 | 0.0433 |
| **P-H** | protocol_fixed | hard gate | 0.7782 | 0.8408 | **0.2771** | **0.8237** | **14.01** | 13.44 | 0.567 | 0.103 | 0.0020 | 0.0054 | 0.0048 |
| **L-V** | legacy | value audit | 0.7707 | 0.8490 | 0.3179 | 0.9548 | **7.78** | 7.30 | 0.476 | 0.780 | 0.0748 | 0.0414 | 0.0446 |
| **P-V** | protocol_fixed | value audit | 0.7687 | 0.8347 | 0.3042 | 0.9221 | **8.88** | 8.36 | 0.522 | 0.717 | 0.0408 | 0.0092 | 0.0052 |

（早停错误率 = premature_stop_rate = 自信但预测错误的停止比例。）

## 2. 四个研究问题（§五）

### RQ1 — 多问 4.6 题是否由 hard gate 导致？**是。**

P-H（protocol_fixed 先验 + hard gate）总题数 14.01，比 L-H 的 9.45 **多 4.56 题**
——与 Phase 8E 冻结结果**逐字复现**（回归检查 0 不匹配）。换成 value audit 后，
P-V 总题数 8.88，比 P-H **少 5.13 题**（CI [4.63, 5.58]，显著）。即那 4.6 题的
膨胀完全来自 hard gate 的 `p_mode>τ_p` 阻断，value audit 将其释放。

### RQ2 — value-based Stop 是否对 prior 更稳健？**是（大幅）。**

- hard gate 的 prior 敏感度：L-H→P-H = **+4.56 题**（9.45→14.01）。
- value audit 的 prior 敏感度：L-V→P-V = **+1.10 题**（7.78→8.88）。
- 敏感度从 4.56 降到 1.10，**下降 76%**。value audit 用「决策价值」而非「误报概率」判定
  是否停止，因而对先验的 MISREPORTED 概率不再敏感。

### RQ3 — P-V 减少多少问题？

- 相对 P-H：**−5.13 题**（14.01→8.88，CI [4.63, 5.58]）。
- 相对 L-H：**−0.57 题**（9.45→8.88，CI [0.28, 0.85]）。
- 主减幅来自 AskNew（P-V 8.36 vs P-H 13.44，−5.08），VerifyOld 几乎不变（0.522 vs 0.567）。

### RQ4 — 是否牺牲 Top-1 / Brier / 停止安全？

| 指标 | P-V vs L-H | P-V vs P-H |
|---|---|---|
| Top-1 | −1.22pp（CI [−0.95, +3.6]，**跨 0**） | −0.95pp（CI [−1.56, −0.34]，**显著**） |
| Brier | +4.38%（0.3042 vs 0.2914，跨 0） | +9.78%（0.3042 vs 0.2771，显著） |
| 早停错误率 | +0.34pp（0.0408 vs 0.0374，跨 0） | +3.88pp（0.0408 vs 0.0020，显著） |

- 相对 L-H，Top-1/Brier/安全三者的点估计都**轻微恶化**，但**均不显著**（CI 跨 0）。
- Top-1 点估计 −1.22pp **超过 0.5pp 预注册阈值**，是唯一未达标项。

### RQ5 — 是否出现强制核验？**否。**

gross 增益阻断（G>τ_V）后，动作比较正常执行，**绝大多数选 AskNew 而非 VerifyOld**：

- P-V：6283 个 gross 阻断回合 → verify 767（12.2%）、new 5516（87.8%）。
- 全部四组一致：gross 阻断从不强制 VerifyOld，验证 Phase 22A 不变式
  「G>τ_V 只意味着不能停止」。

## 3. 校准（§七 附带）

- protocol_fixed 先验（P-H/P-V）把 p_wrong ECE 从 0.040 拉到 0.005~0.009、p_mode ECE
  从 0.043 拉到 0.005。
- value audit 本身**不**修复校准（L-V ≈ L-H 的 0.041/0.045），但 P-V **保留**了
  protocol_fixed 的校准收益（0.009/0.005）。
- 即「先验修校准、价值审计修题数」二者正交，P-V 同时拿到两者。

## 4. Top-1 配对迁移（bootstrap）

| 比较 | wrong→correct | correct→wrong | net（bootstrap CI） |
|---|---|---|---|
| P-V vs P-H | 16 | 2 | +14 [6, 23]（P-H 靠多问 5 题救回更多） |
| P-V vs L-H | 103 | 85 | +18 [−13, +52]（**跨 0**） |
| L-V vs L-H | 19 | 4 | +15 [6, 25] |

P-V 相对 L-H 的净迁移 +18 例（1.22pp）CI 跨 0，与 §2 的 Top-1 不显著一致。

## 5. 方法学

- 245 case × 2 noise {0.2,0.3} × 3 seed {2026,2027,2028} × 4 策略 = 5880 轨迹。
- matched 0/0（rho_env=rho_model=0），manifest 冻结自 N=20 清单前 5/病种。
- case 级 cluster bootstrap 1000 次，seed 2026。
- 基线回归：L-H ≡ joint_legacy_prior、P-H ≡ joint_train_fixed_prior，**逐字 0 不匹配**。
- 预测侧不读 true disease / latent / true wrongness / noise / clean（见 LEAKAGE_AUDIT.md）。

## 6. 待确认风险（N=20 需回答）

1. P-V 的 Top-1 点估计 −1.22pp 是否真实（当前 CI 跨 0，power 不足）。
2. P-V 的 Brier +4.38%、早停错误 +0.34pp 是否与 L-H 有统计差异。
3. value audit 释放的 6912 个 hard_gate_only 回合中，是否混入了「本应核验」的低价值阻断。

详细机制审计见 STOP_MECHANISM_AUDIT.md，Go/No-Go 判定见 GO_NO_GO.md。
