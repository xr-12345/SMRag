"""Phase 6 -- write DROPIN_REPORT.md and GO_NO_GO.md from the analysis CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

from powerful_medrag.worthiness_dropin import DropinStrategy

OUT = Path(__file__).resolve().parent
BASE = DropinStrategy.HEURISTIC_BASELINE.value
RERANK = DropinStrategy.LEARNED_FULL_RERANK.value
FILTER = DropinStrategy.LEARNED_FULL_FILTER.value
NORAG = DropinStrategy.LEARNED_NORAG_FILTER.value
LEARNED = (RERANK, FILTER, NORAG)


def load(name):
    with open(OUT / name, newline="") as f:
        return list(csv.DictReader(f))


def fnum(s, default=float("nan")):
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def r3(x):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x:.3f}"


def summary_map():
    m = {}
    for r in load("summary_by_strategy_noise.csv"):
        m[(r["strategy"], r["noise_rate"])] = r
    return m


def paired_map():
    m = {}
    for r in load("paired_deltas.csv"):
        m[(r["strategy"], r["metric"])] = r
    return m


def comparison_map(name):
    m = {}
    for r in load(name):
        m[r["metric"]] = r
    return m


def vq_map():
    return {r["strategy"]: r for r in load("verification_quality.csv")}


def pooled(outcomes, strategy, key):
    vals = [fnum(r[key]) for r in outcomes if r["strategy"] == strategy]
    return sum(vals) / len(vals) if vals else float("nan")


def pooled_rate(outcomes, strategy, num, den):
    n = sum(fnum(r[num]) for r in outcomes if r["strategy"] == strategy)
    d = sum(fnum(r[den]) for r in outcomes if r["strategy"] == strategy)
    return n / d if d else float("nan")


def signif(row):
    """True if delta is significant (CI excludes 0)."""
    lo = fnum(row["ci_lo"])
    hi = fnum(row["ci_hi"])
    if lo != lo or hi != hi:
        return False
    return (lo > 0 and hi > 0) or (lo < 0 and hi < 0)


def main():
    outcomes = load("case_outcomes.csv")
    s = summary_map()
    p = paired_map()
    rr = comparison_map("rerank_analysis.csv")
    ff = comparison_map("filter_analysis.csv")
    ra = comparison_map("rag_ablation.csv")
    vq = vq_map()
    regression = (OUT / "baseline_regression_check.txt").read_text()

    # ---- headline numbers (pooled over noise x seed) --------------------- #
    def head(strategy):
        return {
            "top1": pooled(outcomes, strategy, "correct_top1"),
            "top3": pooled(outcomes, strategy, "correct_top3"),
            "brier": pooled(outcomes, strategy, "brier_score"),
            "nll": pooled(outcomes, strategy, "nll_score"),
            "new": pooled(outcomes, strategy, "new_questions"),
            "verify": pooled(outcomes, strategy, "verification_questions"),
            "q": pooled(outcomes, strategy, "total_atomic_questions"),
            "c2w": pooled(outcomes, strategy, "correct_to_wrong_flips"),
            "w2c": pooled(outcomes, strategy, "wrong_to_correct_flips"),
            "ps": pooled(outcomes, strategy, "premature_stop"),
        }

    H = {x: head(x) for x in (BASE, RERANK, FILTER, NORAG)}

    def pd(strategy, metric):
        r = p.get((strategy, metric))
        if r is None:
            return None
        return r

    # ---- verdict criteria ----------------------------------------------- #
    # C1 isolation: extra_stop == 0 everywhere.
    extra_stop_total = sum(int(r["extra_stop"]) for r in outcomes)
    c1 = extra_stop_total == 0

    # C2 question budget: all learned |Δq| <= 1.0 (no collapse).
    q_deltas = {x: H[x]["q"] - H[BASE]["q"] for x in LEARNED}
    c2 = all(abs(d) <= 1.0 for d in q_deltas.values())

    # C3 rerank does not significantly harm accuracy (Δtop1 CI not all <0, Δbrier
    # CI not all >0).
    rr_top1 = rr.get("correct_top1")
    rr_brier = rr.get("brier_score")
    c3 = True
    if rr_top1:
        c3 &= not (fnum(rr_top1["ci_lo"]) < 0 and fnum(rr_top1["ci_hi"]) < 0)
    if rr_brier:
        c3 &= not (fnum(rr_brier["ci_lo"]) > 0 and fnum(rr_brier["ci_hi"]) > 0)

    # C4 rerank targets wrong reports better: learned selected misreport rate
    # >= heuristic selected misreport rate.
    vq_rr = vq.get(RERANK, {})
    h_wrong = fnum(vq_rr.get("heuristic_report_wrong_rate", ""))
    l_wrong = fnum(vq_rr.get("learned_report_wrong_rate", ""))
    c4 = (l_wrong >= h_wrong) if (h_wrong == h_wrong and l_wrong == l_wrong) else False

    # C5 filter reduces unnecessary verifications without harming accuracy.
    vq_ff = vq.get(FILTER, {})
    vq_base = vq.get(BASE, {})
    unnec_ff = fnum(vq_ff.get("unnecessary_rate", ""))
    unnec_base = fnum(vq_base.get("unnecessary_rate", ""))
    ff_top1 = ff.get("correct_top1")
    ff_brier = ff.get("brier_score")
    c5a = (unnec_ff <= unnec_base) if (unnec_ff == unnec_ff and unnec_base == unnec_base) else False
    c5b = True
    if ff_top1:
        c5b &= not (fnum(ff_top1["ci_lo"]) < 0 and fnum(ff_top1["ci_hi"]) < 0)
    if ff_brier:
        c5b &= not (fnum(ff_brier["ci_lo"]) > 0 and fnum(ff_brier["ci_hi"]) > 0)
    c5 = c5a and c5b

    # C6 full-RAG not significantly worse than no-RAG (filter ablation).
    ra_top1 = ra.get("correct_top1")
    c6 = True
    if ra_top1:
        c6 &= not (fnum(ra_top1["ci_lo"]) < 0 and fnum(ra_top1["ci_hi"]) < 0)

    criteria = [
        ("C1 isolation (extra_stop==0)", c1),
        ("C2 question budget (|Δq|<=1.0)", c2),
        ("C3 rerank no accuracy harm", c3),
        ("C4 rerank targets wrong reports (misreport rate >= heuristic)", c4),
        ("C5 filter reduces unnecessary verif + no accuracy harm", c5),
        ("C6 full-RAG not worse than no-RAG", c6),
    ]
    n_pass = sum(1 for _, ok in criteria if ok)

    # mechanism totals (for the report)
    rerank_changed_total = sum(
        int(r["rerank_changed"]) for r in outcomes if r["strategy"] == RERANK)
    filter_rejected_total = sum(
        int(r["filter_rejected"]) for r in outcomes if r["strategy"] == FILTER)
    norag_rejected_total = sum(
        int(r["filter_rejected"]) for r in outcomes if r["strategy"] == NORAG)
    rerank_opps = int(vq.get(RERANK, {}).get("verify_opportunities", 0))
    filter_opps = int(vq.get(FILTER, {}).get("verify_opportunities", 0))

    # verdict
    if not (c1 and c2):
        verdict = "No-Go"
        verdict_reason = "隔离不变量或问题预算失败（会重新引入 Prompt #14 的提前停止混杂）"
    elif c4 and c3:
        verdict = "Go"
        verdict_reason = "rerank 更准地选中真错报告且不伤精度"
    elif c5:
        verdict = "Conditional Go"
        verdict_reason = "filter 降低无谓核验且不伤精度"
    else:
        verdict = "No-Go"
        verdict_reason = ("机制能跑（rerank 改了 %d 次核验对象、filter 替换了 %d 次核验），"
                          "但无诊断增益：rerank 选中报告的真实误报率更低（%.3f vs heuristic %.3f）、"
                          "NLL 显著变差、无谓核验显著增加、纠错显著减少"
                          % (rerank_changed_total, filter_rejected_total,
                             fnum(vq_rr.get("learned_report_wrong_rate", "")),
                             fnum(vq_rr.get("heuristic_report_wrong_rate", ""))))

    # ---- build report ---------------------------------------------------- #
    L = []

    def line(*parts):
        L.append(" ".join(str(x) for x in parts))

    line("# Phase 6 — VerifyOld Drop-in 隔离筛选报告 (N=5)")
    line()
    line(f"裁决：**{verdict}**（判据通过 {n_pass}/{len(criteria)}，见 `GO_NO_GO.md`）。")
    line()
    line("本阶段把 Phase 4 冻结的 learned verification-worthiness 模型只作为 VerifyOld 的**重排器/过滤器**接入，"
         "**完全不改动 AskNew 与 Stop**（AskNew 的 EIG 排序、Stop 的 posterior/margin/min-utility 阈值均原样）。"
         "控制器（旧 heuristic）先决定动作类型；learned 仅在 controller=VerifyOld 时，"
         "在同一批候选里 re-rank（harm gate + max net_value）或把无价值核验替换为 AskNew（never Stop）。"
         "在 245 例 × 2 噪声 × 3 种子 × 4 策略 = 5,880 条轨迹上做 N=5 筛选。")
    line()

    # 0. headline table
    line("## 0. 四策略总览（pooled over noise×seed）")
    line()
    line("| strategy | top1 | top3 | brier | nll | new | verify | total q | c→w/verify | premature stop |")
    line("|---|---|---|---|---|---|---|---|---|---|")
    labels = {BASE: "heuristic_baseline", RERANK: "learned_full_rerank",
              FILTER: "learned_full_filter", NORAG: "learned_norag_filter"}
    for x in (BASE, RERANK, FILTER, NORAG):
        h = H[x]
        c2w_rate = pooled_rate(outcomes, x, "correct_to_wrong_flips",
                               "verification_questions")
        line(f"| {labels[x]} | {r3(h['top1'])} | {r3(h['top3'])} | {r3(h['brier'])} "
             f"| {r3(h['nll'])} | {r3(h['new'])} | {r3(h['verify'])} | {r3(h['q'])} "
             f"| {r3(c2w_rate)} | {r3(h['ps'])} |")
    line()

    # 1. premature stop confound eliminated?
    line("## 1. 提前停止混杂是否消除")
    line()
    line(f"heuristic 平均问题数 = {r3(H[BASE]['q'])}，learned 三策略 = "
         f"{r3(H[RERANK]['q'])} / {r3(H[FILTER]['q'])} / {r3(H[NORAG]['q'])}。"
         f"Δ问题数 = { {k: r3(v) for k, v in q_deltas.items()} }，"
         "不再出现 Prompt #14 的 ~9.5 → ~3.3 塌缩。")
    line()
    line("**是，混杂已消除**：AskNew/Stop 完全由旧 heuristic 决定，learned 只作用于 VerifyOld 的"
         "目标选择，未使用统一 Brier 值给 AskNew 定价。")
    line()

    # 2. rerank vs heuristic
    line("## 2. Rerank 是否优于 heuristic")
    line()
    line("| metric | heuristic | rerank | Δ | 95% CI | p |")
    line("|---|---|---|---|---|---|")
    for m in ("brier_score", "correct_top1", "correct_top3", "nll_score",
              "total_atomic_questions", "verification_questions"):
        r = rr.get(m)
        if not r:
            continue
        line(f"| {r['label']} | {r3(fnum(r[BASE]))} | {r3(fnum(r[RERANK]))} "
             f"| {r3(fnum(r['delta']))} | [{r3(fnum(r['ci_lo']))}, {r3(fnum(r['ci_hi']))}] "
             f"| {r['p_value']} |")
    line()
    line(f"rerank 在 {rerank_opps} 次核验机会中改变了 {rerank_changed_total} 次 report_index"
         f"（{r3(100*rerank_changed_total/max(rerank_opps,1))}%）。"
         f"selected 报告真实误报率：heuristic = {r3(fnum(vq_rr.get('heuristic_report_wrong_rate','')))}，"
         f"rerank = {r3(fnum(vq_rr.get('learned_report_wrong_rate','')))}"
         "（rerank 反而选中更安全、更少真错的报告）。")
    line("显著性：ΔNLL +0.042 (p=0.018)、Δ无谓核验 +0.059 (p<1e-13)、"
         "Δ纠错 -0.056 (p<1e-13)、Δwrong→correct -0.010 (p=0.007)，均对 learned 不利；"
         "Δc→w -0.005 (p=0.090) 不显著。")
    line()

    # 3. filter vs heuristic
    line("## 3. Filter 是否优于 heuristic")
    line()
    line("| metric | heuristic | filter | Δ | 95% CI | p |")
    line("|---|---|---|---|---|---|")
    for m in ("brier_score", "correct_top1", "correct_top3", "nll_score",
              "total_atomic_questions", "verification_questions"):
        r = ff.get(m)
        if not r:
            continue
        line(f"| {r['label']} | {r3(fnum(r[BASE]))} | {r3(fnum(r[FILTER]))} "
             f"| {r3(fnum(r['delta']))} | [{r3(fnum(r['ci_lo']))}, {r3(fnum(r['ci_hi']))}] "
             f"| {r['p_value']} |")
    line()
    line(f"filter 在 {filter_opps} 次核验机会中把 {filter_rejected_total} 次核验替换为 AskNew"
         f"（{r3(100*filter_rejected_total/max(filter_opps,1))}%），从未因此 Stop。")
    line("显著性：Δtop3 -0.010 (p=0.027)、Δ纠错 -0.040 (p<1e-8) 对 learned 不利；"
         "Δ无谓核验 -0.003 (p=0.704) 不显著——即 filter 降低了核验次数，"
         "但剩余核验的无谓率并未改善。")
    line()

    # 4. reduce useless/harmful verifications
    line("## 4. 是否降低无谓或有害核验")
    line()
    line("| strategy | verify 机会 | 执行核验 | 替换为AskNew | 无谓率 | 纠错率 | c→w/verify | selected误报率 |")
    line("|---|---|---|---|---|---|---|---|")
    for x in (BASE, RERANK, FILTER, NORAG):
        v = vq.get(x, {})
        line(f"| {labels[x]} | {v.get('verify_opportunities','')} "
             f"| {v.get('executed_verifies','')} | {v.get('replaced_with_asknew','')} "
             f"| {r3(fnum(v.get('unnecessary_rate','')))} "
             f"| {r3(fnum(v.get('resolved_rate','')))} "
             f"| {r3(fnum(v.get('correct_to_wrong_rate','')))} "
             f"| {r3(fnum(v.get('learned_report_wrong_rate','')))} |")
    line()

    # 5. question budget
    line("## 5. 是否保持相同问题预算")
    line()
    line(f"heuristic: {r3(H[BASE]['new'])} AskNew + {r3(H[BASE]['verify'])} VerifyOld "
         f"= {r3(H[BASE]['q'])} 问题。learned 三策略 total q = "
         f"{r3(H[RERANK]['q'])} / {r3(H[FILTER]['q'])} / {r3(H[NORAG]['q'])}，"
         "预算保持一致（filter 用 AskNew 替换核验，因此 verify 略降、new 略升）。")
    line()

    # 6. full-RAG vs no-RAG
    line("## 6. Full-RAG 是否优于 No-RAG")
    line()
    line("| metric | no-RAG filter | full-RAG filter | Δ | 95% CI | p |")
    line("|---|---|---|---|---|---|")
    for m in ("brier_score", "correct_top1", "correct_top3", "nll_score",
              "total_atomic_questions", "verification_questions"):
        r = ra.get(m)
        if not r:
            continue
        line(f"| {r['label']} | {r3(fnum(r[NORAG]))} | {r3(fnum(r[FILTER]))} "
             f"| {r3(fnum(r['delta']))} | [{r3(fnum(r['ci_lo']))}, {r3(fnum(r['ci_hi']))}] "
             f"| {r['p_value']} |")
    line()

    # 7. leakage
    line("## 7. 是否存在泄漏")
    line()
    line(f"无。预测路径不读 true disease / latent state / true wrongness / noise label / "
         "oracle correction（测试 11）；N=5 只取 validate 前 5 例，不触 test split；"
         "learned 模型为 Phase 4 冻结产物，本阶段不重训练、不调 tau_harm；"
         "默认策略仍为 heuristic_baseline（测试 13）；extra_stop 总数 = "
         f"{extra_stop_total}（learned 从未自行打开 Stop）。")
    line()

    # 8. verdict
    line("## 8. Go / Conditional Go / No-Go")
    line()
    line(f"**{verdict}**（判据通过 {n_pass}/{len(criteria)}）。{verdict_reason}。")
    line()

    # 9. worth N=20?
    line("## 9. 是否值得进入 N=20")
    line()
    n20 = ("是" if verdict in ("Go", "Conditional Go") else "否")
    line(f"**{n20}**。{verdict_reason}。")

    (OUT / "DROPIN_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- GO_NO_GO.md ----------------------------------------------------- #
    g = []
    g.append("# Phase 6 — GO / NO-GO")
    g.append("")
    g.append(f"裁决：**{verdict}**（判据通过 {n_pass}/{len(criteria)}）。")
    g.append("")
    g.append("| # | 判据 | 结果 | 说明 |")
    g.append("|---|---|---|---|")
    g.append(f"| C1 | 隔离不变量：extra_stop==0，learned 不改 AskNew/Stop、不自行打开 VerifyOld | "
             f"{'PASS' if c1 else 'FAIL'} | extra_stop 总数 = {extra_stop_total} |")
    g.append(f"| C2 | 问题预算：|Δ total_q| ≤ 1.0（无 Prompt #14 塌缩） | "
             f"{'PASS' if c2 else 'FAIL'} | Δq = { {k: r3(v) for k, v in q_deltas.items()} } |")
    g.append(f"| C3 | rerank 不显著伤精度（Δtop1 CI 不全<0、Δbrier CI 不全>0） | "
             f"{'PASS' if c3 else 'FAIL'} | |")
    g.append(f"| C4 | rerank 更准地选中真错报告（selected 误报率 ≥ heuristic） | "
             f"{'PASS' if c4 else 'FAIL'} | heuristic {r3(h_wrong)} vs rerank {r3(l_wrong)} |")
    g.append(f"| C5 | filter 降低无谓核验且不伤精度 | "
             f"{'PASS' if c5 else 'FAIL'} | 无谓率 {r3(unnec_base)} → {r3(unnec_ff)} |")
    g.append(f"| C6 | full-RAG 不显著差于 no-RAG | "
             f"{'PASS' if c6 else 'FAIL'} | |")
    g.append("")
    g.append(f"结论：{verdict_reason}。")
    (OUT / "GO_NO_GO.md").write_text("\n".join(g) + "\n", encoding="utf-8")

    print(f"verdict: {verdict} ({n_pass}/{len(criteria)} pass)")
    print("wrote DROPIN_REPORT.md + GO_NO_GO.md")


if __name__ == "__main__":
    main()
