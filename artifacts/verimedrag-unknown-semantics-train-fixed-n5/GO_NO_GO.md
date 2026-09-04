# Phase 8E — Go / No-Go（分开判断）

## A. UNKNOWN 语义修复 —— **GO**

| §七 条件 | 结果 | 证据 |
|---|---|---|
| UNKNOWN 不再输出数值型 `p_wrong` | ✅ | `p_wrong` 返回 `None`（非 1.0、非 0.0）；`is_nonresponse` 独立标记 |
| 校准评价不再混入 UNKNOWN | ✅ | wrong-report ECE/Brier 仅 explicit 子集；`calibration_metrics.csv` 标注「UNKNOWN excluded」 |
| 策略轨迹与原 joint 路径一致 | ✅ | `baseline_regression_check.txt`：1470/1470，0 mismatch |
| 无旧测试回归 | ✅ | 全量 240 tests OK（`TEST_LOG.txt`） |

**判定：GO。** 该修复是纯语义/日志/接口改动，不进入决策路径（`choose_action` 只调
`p_mode_misreported`），零轨迹风险，可直接采纳。

---

## B. protocol_fixed（train_fixed）先验 —— **校准 GO / 诊断部署 NO-GO**

| §七 条件 | 阈值 | 实测 | 判定 |
|---|---|---|---|
| p_wrong ECE 明确回答上优于 legacy | 优于 | 0.0398 → 0.0054（Δ −0.034，CI 不跨 0） | ✅ |
| Top-1 下降 ≤ 0.5pp | ≤ 0.5pp | −0.27pp（CI [−2.5, +1.8]pp，不显著） | ✅ |
| diagnostic Brier 相对恶化 ≤ 5% | ≤ +5% | **−4.9%**（改善，非恶化） | ✅ |
| 总问题数增加 ≤ 0.25 | ≤ +0.25 | **+4.56**（CI [+4.08, +5.05]） | ❌ **严重超标** |
| 两个噪声方向基本一致 | 一致 | noise=0.2 与 0.3 均改善 | ✅ |
| 无 test/latent/noise-type 泄漏 | 无 | `LEAKAGE_AUDIT.md` 全绿 | ✅ |

**判定：**
- **作为可靠度校准（calibration）：GO。** `p_wrong`/`p_mode` ECE 双双降到 <0.006，
  bootstrap CI 极窄且不跨 0，校准改善是真实、稳健、可复现的。
- **作为诊断默认配置：NO-GO。** 第 4 项「总问题数 ≤+0.25」以 +4.56（超 18 倍）失败，
  且 Top-1 未改善、diagnostic Brier 的微改善来自「多问」而非「答得更准」。按 §七 末句，
  如实表述为 **calibration fix**，**不声称提升诊断性能**。
- **处置：保留为可选配置**（用于校准/解释/审计/可靠性上报等上下文），**不设为默认**，
  **不进入 N=20** 作为默认部署候选。

---

## 最终结论

1. UNKNOWN 语义修复 → **采纳**（GO）。
2. protocol_fixed 先验 → **保留为可选配置，不默认**（校准 GO，诊断 NO-GO）。
3. 本阶段不运行 N=20（红线）。
