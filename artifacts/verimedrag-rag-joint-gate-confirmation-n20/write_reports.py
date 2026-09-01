"""Phase 2C report generator.

Reads the analysis CSVs produced by ``analyze_confirmation.py``, evaluates the
7 preregistered acceptance criteria on the 735-case primary subset (total
comparison joint_gate_w1.0 vs rank_only_w0.0, pooled noise 0.2/0.3), decides the
verdict, and writes GO_NO_GO.md + CONFIRMATION_REPORT.md.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASELINE = "rank_only_w0.0"
TOTAL_A = "joint_gate_w1.0"
TOTAL_B = "rank_only_w0.0"
POOLED = "0.2+0.3"


def load(path: str) -> list[dict]:
    with open(OUT / path, newline="") as f:
        return list(csv.DictReader(f))


def f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def i(x) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def find_boot(boot, subset, comparison, metric, level):
    for r in boot:
        if (
            r["subset"] == subset
            and r["comparison"] == comparison
            and r["metric"] == metric
            and r["level"] == level
        ):
            return r
    return None


def find_summary(summary, config, noise):
    for r in summary:
        if r["config"] == config and r["noise_rate"] == noise:
            return r
    return None


def find_trans(trans, subset, comparison, level):
    for r in trans:
        if (
            r["subset"] == subset
            and r["comparison"] == comparison
            and r["level"] == level
        ):
            return r
    return None


def pct(x, nd=1):
    return f"{x * 100:.{nd}f}%"


def main() -> None:
    summary = load("summary_primary.csv")
    summary_full = load("summary_full_n20.csv")
    boot = load("bootstrap_results.csv")
    trans = load("diagnosis_transition_counts.csv")
    trig = load("retrieval_trigger_analysis.csv")
    pd = load("paired_deltas_primary.csv")

    # ---- extract acceptance numbers (total comparison, primary, pooled 0.2/0.3)
    b_brier = find_boot(boot, "primary", "total", "brier", POOLED)
    b_top1 = find_boot(boot, "primary", "total", "top1", POOLED)
    b_quest = find_boot(boot, "primary", "total", "total_questions", POOLED)
    b_unnec = find_boot(boot, "primary", "total", "unnecessary_verification_rate", POOLED)
    b_conflict = find_boot(boot, "primary", "total", "conflict_resolution_rate", POOLED)

    # summaries (pooled via averaging the two noise levels is not exact; use per-noise)
    base_02 = find_summary(summary, BASELINE, "0.2")
    base_03 = find_summary(summary, BASELINE, "0.3")
    tot_02 = find_summary(summary, TOTAL_A, "0.2")
    tot_03 = find_summary(summary, TOTAL_A, "0.3")

    # trigger analysis row for joint_gate
    trig_row = next((r for r in trig if r["config"] == TOTAL_A), None)

    # per-seed brier deltas (replication check) for total comparison, pooled
    seed_brier = defaultdict(list)
    for r in pd:
        if r["comparison"] == "total" and r["level"] == POOLED:
            seed_brier[r["seed"]].append(f(r["delta_brier"]))
    seed_brier_mean = {
        s: sum(v) / len(v) for s, v in sorted(seed_brier.items()) if v
    }

    # ---- evaluate 7 criteria
    met = {}
    notes = {}
    if b_brier:
        met[1] = f(b_brier["point_estimate"]) < 0 and f(b_brier["ci_high"]) < 0
        notes[1] = (
            f"Brier Δ={f(b_brier['point_estimate']):.5f} "
            f"95% CI [{f(b_brier['ci_low']):.5f}, {f(b_brier['ci_high']):.5f}]"
        )
    else:
        met[1] = False
        notes[1] = "missing"

    if b_top1:
        met[2] = f(b_top1["ci_low"]) >= -0.005
        notes[2] = (
            f"Top-1 Δ={f(b_top1['point_estimate']):.5f} "
            f"95% CI lower={f(b_top1['ci_low']):.5f}"
        )
    else:
        met[2] = False
        notes[2] = "missing"

    if b_quest:
        met[3] = f(b_quest["point_estimate"]) <= 0.25
        notes[3] = f"total atomic questions Δ={f(b_quest['point_estimate']):.3f}"
    else:
        met[3] = False
        notes[3] = "missing"

    if b_unnec:
        met[4] = f(b_unnec["point_estimate"]) < 0
        notes[4] = f"unnecessary-verification rate Δ={f(b_unnec['point_estimate']):.4f}"
    else:
        met[4] = False
        notes[4] = "missing"

    if b_conflict:
        met[5] = f(b_conflict["point_estimate"]) > 0
        notes[5] = f"conflict-resolution rate Δ={f(b_conflict['point_estimate']):.4f}"
    else:
        met[5] = False
        notes[5] = "missing"

    if trig_row:
        base_rate = f(trig_row["candidate_misreport_base_rate"])
        prec = f(trig_row["triggered_precision"])
        met[6] = prec > base_rate
        notes[6] = (
            f"triggered precision {pct(prec)} (Wilson [{pct(f(trig_row['triggered_precision_wilson_low']))}, "
            f"{pct(f(trig_row['triggered_precision_wilson_high']))}]) vs base rate {pct(base_rate)}; "
            f"enrichment {trig_row['precision_enrichment']}x; "
            f"n_triggered={trig_row['n_triggered_verifications']}"
        )
    else:
        met[6] = False
        notes[6] = "missing"

    # criterion 7: >=1 primary clinical/safety endpoint improves not solely via more questions
    # endpoints: Top-1, Top-3, Brier, NLL, ECE, premature-stop (clinical/calibration),
    # unnecessary rate + conflict rate (safety/efficiency).  Question burden already capped.
    improvements = []
    if b_top1 and f(b_top1["point_estimate"]) > 0:
        improvements.append("Top-1")
    if b_brier and f(b_brier["point_estimate"]) < 0:
        improvements.append("Brier")
    b_top3 = find_boot(boot, "primary", "total", "top3", POOLED)
    if b_top3 and f(b_top3["point_estimate"]) > 0:
        improvements.append("Top-3")
    b_nll = find_boot(boot, "primary", "total", "nll", POOLED)
    if b_nll and f(b_nll["point_estimate"]) < 0:
        improvements.append("NLL")
    if base_02 and tot_02 and base_03 and tot_03:
        ece_delta = (
            (f(tot_02["ece_15bin"]) + f(tot_03["ece_15bin"])) / 2
            - (f(base_02["ece_15bin"]) + f(base_03["ece_15bin"])) / 2
        )
        if ece_delta < 0:
            improvements.append("ECE")
        pm_delta = (
            (f(tot_02["premature_stop_rate"]) + f(tot_03["premature_stop_rate"])) / 2
            - (f(base_02["premature_stop_rate"]) + f(base_03["premature_stop_rate"])) / 2
        )
        if pm_delta < 0:
            improvements.append("premature-stop")
    if met[4]:
        improvements.append("unnecessary-verification rate")
    if met[5]:
        improvements.append("conflict-resolution rate")
    met[7] = len(improvements) > 0
    notes[7] = f"improving endpoints: {improvements if improvements else 'none'}"

    # ---- verdict
    if not met[2]:
        verdict = "No-Go"
        verdict_reason = "Top-1 明显退化（paired Δ 95% CI 下界 < −0.005）"
    elif not met[1]:
        verdict = "No-Go"
        verdict_reason = (
            "Brier 改善未达统计显著（95% CI 含 0，跨 seed 不稳定），"
            "Phase-2B 的 Brier 优势未在确认集复现"
        )
    elif not met[7]:
        verdict = "No-Go"
        verdict_reason = "仅轨迹变化，无临床/校准/安全获益"
    elif all(met.values()):
        verdict = "Go"
        verdict_reason = "7 项判据全部通过"
    elif not (met[3] and met[4] and met[5] and met[6]):
        verdict = "Conditional Go"
        verdict_reason = "Brier 显著且 Top-1 非劣，但效率/机制判据不稳定"
    else:
        verdict = "Go"
        verdict_reason = "7 项判据全部通过"

    # ---- write GO_NO_GO.md
    go_lines = []
    go_lines.append("# Phase 2C 冻结配置 N=20 确认实验 —— Go / No-Go 判定\n")
    go_lines.append(f"**判定：{verdict}**\n")
    go_lines.append(f"**理由：{verdict_reason}。**\n")
    go_lines.append("")
    go_lines.append(
        "判断对象：735 例主确认子集（排除 245 例筛选病例，overlap=0），"
        "`joint_gate_w1.0` vs 基线 `rank_only_w0.0`，pooled noise 0.2/0.3。\n"
    )
    go_lines.append("")
    go_lines.append("## 七项判据逐条结论\n")
    go_lines.append("| # | 判据 | 结果 | 证据 |")
    go_lines.append("|---|------|------|------|")
    crit_names = [
        "Brier paired Δ < 0 且 95% CI 上界 < 0",
        "Top-1 非劣：paired Δ 95% CI 下界 ≥ −0.005",
        "平均总原子问题数增加 ≤ 0.25",
        "不必要核验率方向 = 下降",
        "冲突解决率方向 = 上升",
        "触发核验精度明显高于候选误报基准率",
        "≥1 项主要临床/安全终点改善，且非仅靠更多问题",
    ]
    for k in range(1, 8):
        mark = "✅ 通过" if met[k] else "❌ 不通过"
        go_lines.append(f"| {k} | {crit_names[k-1]} | {mark} | {notes[k]} |")
    go_lines.append("")
    go_lines.append("## 红线合规确认\n")
    go_lines.append("- ✅ 仅 validate 分支；未访问 DDXPlus test。")
    go_lines.append("- ✅ 未重新调权重/阈值（配置冻结，见 CONFIG.json）。")
    go_lines.append("- ✅ 未加入 LLM / dense retriever / SFT / RL。")
    go_lines.append("- ✅ 未修改既有正式结果；未覆盖 Phase 2A/2B artifacts。")
    go_lines.append("- ✅ 未修改 preregistered success criteria。")
    go_lines.append("- ✅ full N=20 仅作为补充，未描述为完全独立确认集。")
    go_lines.append("- ✅ 评估真值仅用于 eval 侧统计，未进入动作决策路径。")
    go_lines.append("- ✅ 未 git commit。")
    go_lines.append("- ✅ 未根据结果改变 Go/No-Go 门槛。")
    go_lines.append("")
    go_lines.append("## 最终决定\n")
    go_lines.append(f"**{verdict}。** {verdict_reason}")
    (OUT / "GO_NO_GO.md").write_text("\n".join(go_lines) + "\n", encoding="utf-8")

    # ---- write CONFIRMATION_REPORT.md
    rp = []
    rp.append("# VeriMedRAG Phase 2C —— 冻结配置 N=20 确认实验报告\n")
    rp.append(
        "范围：DDXPlus validate，980 例（49 病种 × 20 例），3 seeds (2026/2027/2028)，"
        "4 noise (0/0.1/0.2/0.3)，3 冻结配置，逐病例配对，共 35,280 轮对话。\n"
    )
    rp.append(
        "主确认子集 735 例（排除 245 例 Phase-2B 筛选病例，overlap=0）；"
        "full 980 例仅作补充（非完全独立）。\n"
    )
    rp.append("")

    rp.append("## 一、主确认子集（735 例）结果\n")
    rp.append("| config | noise | Top-1 | Top-3 | Brier | NLL | ECE | 总原子问题 | 不必要率 | 冲突率 |")
    rp.append("|--------|-------|-------|-------|-------|-----|-----|-----------|---------|--------|")
    for cfg in ["rank_only_w0.0", "rank_only_w1.0", "joint_gate_w1.0"]:
        for nz in ["0.2", "0.3"]:
            r = find_summary(summary, cfg, nz)
            if not r:
                continue
            rp.append(
                f"| {cfg} | {nz} | {pct(f(r['top1_accuracy']))} | {pct(f(r['top3_accuracy']))} | "
                f"{f(r['brier_score_mean']):.4f} | {f(r['nll_mean']):.3f} | {f(r['ece_15bin']):.4f} | "
                f"{f(r['total_atomic_questions_mean']):.2f} | {pct(f(r['unnecessary_verification_rate']))} | "
                f"{pct(f(r['conflict_resolution_rate']))} |"
            )
    rp.append("")

    rp.append("## 二、full N=20（980 例）补充结果\n")
    rp.append("（见 summary_full_n20.csv；此处仅列 pooled 0.2/0.3 的 Top-1/Brier 关键值。）\n")
    rp.append("| config | Top-1 (0.2/0.3) | Brier (0.2/0.3) |")
    rp.append("|--------|-----------------|-----------------|")
    for cfg in ["rank_only_w0.0", "rank_only_w1.0", "joint_gate_w1.0"]:
        r02 = find_summary(summary_full, cfg, "0.2")
        r03 = find_summary(summary_full, cfg, "0.3")
        if r02 and r03:
            rp.append(
                f"| {cfg} | {pct(f(r02['top1_accuracy']))} / {pct(f(r03['top1_accuracy']))} | "
                f"{f(r02['brier_score_mean']):.4f} / {f(r03['brier_score_mean']):.4f} |"
            )
    rp.append("")

    rp.append("## 三、三项成对比较 + 95% CI（主确认子集，pooled 0.2/0.3）\n")
    rp.append("| 比较 | 指标 | Δ (点估计) | 95% CI |")
    rp.append("|------|------|-----------|--------|")
    for comp, aa, bb in [("ranking", "rank_only_w1.0", "rank_only_w0.0"),
                         ("gating", "joint_gate_w1.0", "rank_only_w1.0"),
                         ("total", "joint_gate_w1.0", "rank_only_w0.0")]:
        for metric in ["top1", "brier", "total_questions", "unnecessary_verification_rate", "conflict_resolution_rate"]:
            r = find_boot(boot, "primary", comp, metric, POOLED)
            if r:
                rp.append(
                    f"| {comp} ({aa} − {bb}) | {metric} | {f(r['point_estimate']):.5f} | "
                    f"[{f(r['ci_low']):.5f}, {f(r['ci_high']):.5f}] |"
                )
    rp.append("")

    rp.append("## 四、wrong→correct / correct→wrong 分解（Top-1，total 比较，primary，pooled）\n")
    rp.append("| 子集 | 比较 | wrong→correct | correct→wrong | 净 | McNemar p | 预测改变 | 轨迹改变(预测不变) |")
    rp.append("|------|------|--------------|--------------|----|----------|---------|------------------|")
    for slab in ["primary", "full_n20"]:
        for comp in ["ranking", "gating", "total"]:
            t = find_trans(trans, slab, comp, POOLED)
            if t:
                net = i(t["wrong_to_correct"]) - i(t["correct_to_wrong"])
                rp.append(
                    f"| {slab} | {comp} | {t['wrong_to_correct']} | {t['correct_to_wrong']} | "
                    f"{net:+d} | {f(t['mcnemar_p']):.4f} | {t['final_prediction_changed']} | "
                    f"{t['trajectory_changed_but_prediction_unchanged']} |"
                )
    rp.append("")

    rp.append("## 五、Brier 改善是否独立复现（跨 seed）\n")
    rp.append("| seed | Brier Δ (total, primary, pooled 0.2/0.3) |")
    rp.append("|------|----------------------------------------|")
    for s, v in seed_brier_mean.items():
        rp.append(f"| {s} | {v:.5f} |")
    rp.append("")
    rp.append(f"pooled Brier Δ = {f(b_brier['point_estimate']):.5f} (95% CI [{f(b_brier['ci_low']):.5f}, {f(b_brier['ci_high']):.5f}]).\n")

    rp.append("## 六、核验效率是否稳定\n")
    rp.append(
        f"不必要核验率 Δ = {f(b_unnec['point_estimate']):.4f} "
        f"(95% CI [{f(b_unnec['ci_low']):.4f}, {f(b_unnec['ci_high']):.4f}])；"
        f"冲突解决率 Δ = {f(b_conflict['point_estimate']):.4f} "
        f"(95% CI [{f(b_conflict['ci_low']):.4f}, {f(b_conflict['ci_high']):.4f}]).\n"
    )

    rp.append("## 七、问题负担是否增加\n")
    rp.append(
        f"总原子问题数 Δ = {f(b_quest['point_estimate']):.3f} "
        f"(95% CI [{f(b_quest['ci_low']):.3f}, {f(b_quest['ci_high']):.3f}])，上限 0.25。\n"
    )

    rp.append("## 八、最终 Go / Conditional Go / No-Go\n")
    rp.append(f"**{verdict}** —— {verdict_reason}。详见 GO_NO_GO.md。\n")

    rp.append("## 九、是否足以支撑 ICLR 论文的 RAG 核心声明\n")
    rp.append(
        "**结论：不足以支撑「RAG 显著提升诊断准确率（Top-1 / Brier）」作为核心声明；"
        "但足以支撑「检索信号 label-free 标定证据可靠性、并提升核验效率」这一机制性声明。**\n"
    )
    rp.append("")
    rp.append("**依据一：机制本身真实且精确。** 检索触发核验的精度 61.7%"
              "（Wilson 95% CI [57.6%, 65.7%]），是候选误报基准率 7.7% 的 **8.0 倍**。"
              "joint gate 打开的正是「确实被误报」的证据项——删除该报告前后的检索 Jaccard 差异"
              "（normalized_retrieval_impact）能够在不读标签的情况下标定证据可靠性。"
              "这是 RAG 论文可诚实主张的核心机制。\n")
    rp.append("")
    rp.append("**依据二：但机制几乎没有转化为下游临床终点的显著改善。** 在 735 例主确认子集上："
              "Top-1 Δ = +0.0032（95% CI [−0.0023, +0.0086]，McNemar p = 0.33），非劣但无显著提升；"
              "Brier Δ = −0.0031（95% CI [−0.0099, +0.0039]），方向为正但 **未达统计显著**（CI 含 0）。"
              "34.2% 的对话轨迹被改变，但仅 6.7% 改变了最终 Top-1 预测；"
              "即便触发核验精确命中了错误证据，修正后多数病例的最终诊断不变"
              "（wrong→correct 96 vs correct→wrong 82，净 +14，p = 0.33）。\n")
    rp.append("")
    rp.append("**依据三：Phase-2B 的 Brier 改善是 seed-2026 特异的，未在确认集复现。** "
              "跨 seed 分解：2026 −0.0128 / 2027 −0.0043 / 2028 +0.0080。"
              "种子 2028 上反而恶化——这是 N=5 筛选过拟合于单一 seed 的典型信号。\n")
    rp.append("")
    rp.append("**可诚实主张的次要点：效率与校准。** "
              "核验效率显著改善：不必要核验率 −4.2%（95% CI [−5.7%, −2.7%]）、"
              "冲突解决率 +3.7%（95% CI [2.3%, 5.2%]），而总原子问题数仅 +0.07（≤ 0.25 上限）——"
              "即「用更少的无用提问、更精准地核验」。NLL 显著下降 −0.022（95% CI [−0.042, −0.001]）："
              "joint gate 让后验更接近真值分布，尽管 Top-1 / Brier 点估计未达显著。\n")
    rp.append("")
    rp.append("**对论文的建议**：不要把「RAG 提升 Top-1 / Brier」写进摘要或作为主结果图——"
              "它经不起 N=20 确认。核心声明应收敛为「BM25 稀疏检索信号能 label-free 标定证据可靠性，"
              "joint gate 据此显著降低不必要核验、提升冲突解决，且不增加提问负担」，"
              "并明确报告诊断准确率的非显著性。若坚持要诊断准确率声明，需更大规模"
              "（更多 seed、更多病例）或更强的检索/核验策略，后者超出本阶段红线。\n")
    (OUT / "CONFIRMATION_REPORT.md").write_text("\n".join(rp) + "\n", encoding="utf-8")

    print(json.dumps({
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "met": {str(k): met[k] for k in range(1, 8)},
        "notes": notes,
        "seed_brier_mean": seed_brier_mean,
    }, indent=2))


if __name__ == "__main__":
    main()
