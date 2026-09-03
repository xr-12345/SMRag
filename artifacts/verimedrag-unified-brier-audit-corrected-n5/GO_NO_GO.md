# Prompt #18 -- N=5 screen Go / No-Go

Reference point: noise 0.2 (validate split, 245 cases x 3 seeds).

| metric | heuristic | corrected | delta |
|---|---:|---:|---:|
| top1 | 0.7932 | 0.7578 | -0.0354 |
| brier | 0.2825 | 0.3321 | +0.1756 |
| total questions | 9.16 | 8.33 | -0.83 |

## Go criteria
- [PASS] `go_forced_verify_eliminated`: corrected policy has no forced-VerifyOld path (gross gain only blocks Stop)
- [PASS] `go_baseline_regression`: heuristic_baseline byte-identical to frozen drop-in heuristic
- [PASS] `go_no_leakage`: prediction path reads no true state / latent / noise
- [FAIL] `go_top1_drop_le_0_005`: |top1 drop| = 0.0354 <= 0.005
- [FAIL] `go_brier_worsening_le_5pct`: Brier rel change = 0.1756 <= 0.05
- [PASS] `go_reduces_unnecessary_verify_vs_forced`: corrected unnecessary=0.033 vs forced=0.102

## No-Go criteria
- [clear] `nogo_question_collapse`: heuristic_q - corrected_q = 0.83 (>1.0 collapses)
- [clear] `nogo_dominated_by_heuristic`: corrected top1 0.7578 < heuristic 0.7932 AND q 8.33 >= 9.16
- [TRIGGERED] `nogo_rag_adds_nothing`: RAG changed 0/1470 trajectories

## Verdict

**No-Go (deployment).** 结构性修正**正确且已验证**：forced-verify 被消除
（`go_forced_verify_eliminated` PASS）、基线回归 byte-identical、无泄漏、
unnecessary-verify 相对 forced **下降 3×**（0.102 → 0.033），且相对 forced 无精度损失
（top1 +0.003、Brier −0.009）。但修正后的 unified-Brier 策略仍**逊于启发式基线**：
top1 **−4.0pp**（0.793 → 0.758）、Brier 相对恶化 **+17.6%**，两项 Go 判据 FAIL。

因此按预先注册判据 → **No-Go**（不可部署；机制修正有效但价值模型不优于启发式）。
RAG 增加 0/1470 轨迹变化（红线锁死，检索不进 Brier 价值）。

