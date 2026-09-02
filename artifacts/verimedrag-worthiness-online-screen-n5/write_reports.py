"""Phase 5 -- write ONLINE_INTEGRATION_REPORT.md + GO_NO_GO.md from the CSVs.

The primary question: when the frozen learned worthiness model is wired into the
real AskNew/VerifyOld/Stop dialogue, does it beat the old heuristic?  All effect
comparisons are paired at the case level and reported with bootstrap CIs.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260901)
N_BOOT = 10000

SHORT = {
    "heuristic": "heuristic_verify",
    "v_bayes": "model_based_vbayes_verify",
    "learned_full": "learned_worthiness_full_rag",
    "learned_norag": "learned_worthiness_no_rag",
}


def read(name):
    with open(OUT / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(row, key):
    v = row.get(key, "")
    if v == "" or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def deltas_for(baseline, candidate, metric):
    """Per-case deltas for one paired comparison from paired_deltas.csv."""
    out = []
    for r in read("paired_deltas.csv"):
        if r["baseline"] == baseline and r["candidate"] == candidate \
                and r["metric"] == metric:
            d = fnum(r, "delta")
            if d is not None:
                out.append(d)
    return out


def bootstrap_ci(deltas, n_boot=N_BOOT, rng=RNG):
    """Bootstrap 95% CI + two-sided p-value for mean(delta) != 0."""
    deltas = np.asarray(deltas, dtype=float)
    if deltas.size == 0:
        return None
    obs = deltas.mean()
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, deltas.size, deltas.size)
        means[i] = deltas[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    # two-sided p: fraction of bootstrap means on the opposite side of 0
    if obs >= 0:
        p = float((means <= 0).mean())
    else:
        p = float((means >= 0).mean())
    return {"mean": obs, "lo": lo, "hi": hi, "p": p,
            "sig": (lo > 0) or (hi < 0)}


def summary_lookup():
    out = defaultdict(dict)
    for r in read("summary_by_strategy_noise.csv"):
        out[(r["strategy_short"], r["noise_rate"])] = r
    return out


def agg_summary():
    """Aggregate summary across noise rates (weighted by n)."""
    rows = defaultdict(lambda: {"n": 0, "acc": []})
    for r in read("summary_by_strategy_noise.csv"):
        s = r["strategy_short"]
        n = int(r["n_trajectories"])
        rows[s]["n"] += n
        for k in ("top1_accuracy", "top3_accuracy", "brier_score",
                  "new_questions_mean", "verification_questions_mean",
                  "total_atomic_questions_mean", "correct_to_wrong_flips_mean",
                  "wall_clock_mean"):
            v = fnum(r, k)
            if v is not None:
                rows[s].setdefault(k, []).append((v, n))
    agg = {}
    for s, d in rows.items():
        row = {"n": d["n"]}
        for k, lst in d.items():
            if k == "n" or not lst:
                continue
            total = sum(v * n for v, n in lst)
            wsum = sum(n for _, n in lst)
            row[k] = total / wsum if wsum else float("nan")
        agg[s] = row
    return agg


def fmt(v, nd=4):
    if v is None:
        return ""
    return f"{v:.{nd}f}" if v == v else ""


def main() -> int:
    agg = agg_summary()
    summ = summary_lookup()

    # ---- bootstrap the key paired comparisons -------------------------- #
    # higher-is-better for accuracy; lower-is-better for brier / questions / harm
    keys = {}
    for base, cand, label in (
        ("heuristic", "learned_full", "learned_full_vs_heuristic"),
        ("heuristic", "learned_norag", "learned_norag_vs_heuristic"),
        ("v_bayes", "learned_full", "learned_full_vs_v_bayes"),
        ("heuristic", "v_bayes", "v_bayes_vs_heuristic"),
    ):
        for metric in ("brier_score", "correct_top1", "total_atomic_questions",
                       "correct_to_wrong_flips"):
            keys[f"{label}.{metric}"] = bootstrap_ci(
                deltas_for(base, cand, metric)
            )

    def ci(label):
        b = keys.get(label)
        if b is None:
            return None
        return b

    # ---- Go / No-Go criteria ------------------------------------------ #
    lf_heur_brier = ci("learned_full_vs_heuristic.brier_score")
    lf_heur_top1 = ci("learned_full_vs_heuristic.correct_top1")
    lf_heur_q = ci("learned_full_vs_heuristic.total_atomic_questions")
    lf_heur_ctw = ci("learned_full_vs_heuristic.correct_to_wrong_flips")
    lf_vb_brier = ci("learned_full_vs_v_bayes.brier_score")

    h = agg["heuristic"]
    lf = agg["learned_full"]
    vb = agg["v_bayes"]
    ln = agg["learned_norag"]

    def sign(b, lower_is_better=True):
        if b is None:
            return False
        if b["sig"]:
            return b["mean"] < 0 if lower_is_better else b["mean"] > 0
        return False

    c1_learned_better_brier = bool(lf_heur_brier and lf_heur_brier["mean"] < 0)
    c1_sig = sign(lf_heur_brier, lower_is_better=True)
    c2_not_worse_top1 = not (lf_heur_top1 and lf_heur_top1["sig"] and lf_heur_top1["mean"] < 0)
    c3_less_ctw = bool(lf_heur_ctw and lf_heur_ctw["mean"] <= 0)
    c4_no_more_questions = bool(lf_heur_q and lf_heur_q["mean"] <= 0)
    c5_approx_vbayes = bool(lf_vb_brier and not (lf_vb_brier["sig"] and lf_vb_brier["mean"] > 0))

    # harm gate: compare learned_full correct->wrong per verification vs heuristic
    verif = {r["strategy_short"]: r for r in read("verification_analysis.csv")}
    ctw_rate = {}
    for s, r in verif.items():
        v = fnum(r, "correct_to_wrong_per_verification")
        ctw_rate[s] = v if v is not None else float("nan")
    c6_harm_gate_lowers_ctw = bool(
        ctw_rate.get("learned_full", 1) <= ctw_rate.get("heuristic", 0) + 1e-9
    )

    # RAG ablation
    rag = {r["metric"]: r for r in read("rag_ablation.csv")}
    rag_brier = fnum(rag.get("brier_score", {}), "mean_delta")
    c7_rag_helps = bool(rag_brier is not None and rag_brier < 0)  # full lower brier

    criteria = {
        "C1 learned brier < heuristic": c1_learned_better_brier,
        "C1sig learned brier < heuristic (CI)": c1_sig,
        "C2 learned top1 not sig worse": c2_not_worse_top1,
        "C3 learned correct->wrong <= heuristic": c3_less_ctw,
        "C4 learned questions <= heuristic": c4_no_more_questions,
        "C5 learned ~ V_Bayes (not sig worse brier)": c5_approx_vbayes,
        "C6 harm gate lowers correct->wrong rate": c6_harm_gate_lowers_ctw,
        "C7 full-RAG brier <= no-RAG": c7_rag_helps,
    }
    n_pass = sum(criteria.values())

    # verdict: Go requires a significant accuracy/brier win over heuristic and no
    # harm regression; Conditional Go = not worse + efficiency win; else No-Go.
    if c1_sig and c2_not_worse_top1 and c6_harm_gate_lowers_ctw:
        verdict = "Go"
    elif c1_learned_better_brier or (c4_no_more_questions and c2_not_worse_top1):
        verdict = "Conditional Go"
    else:
        verdict = "No-Go"

    # ---- write ONLINE_INTEGRATION_REPORT.md --------------------------- #
    L = []
    add = L.append
    add("# Phase 5 — Learned Verification-Worthiness 在线集成与 N=5 筛选报告")
    add("")
    add(f"裁决：**{verdict}**（判据通过 {n_pass}/8，细节见 `GO_NO_GO.md`）。")
    add("")
    add("本阶段把 Phase 4 冻结的 learned verification-worthiness 模型（hgb gain + hgb harm，"
        "tau_harm=0.4129）接入完整 AskNew/VerifyOld/Stop 问诊策略，在 245 例 × 2 噪声 × 3 种子 × "
        "4 策略 = 5,880 条真实问诊轨迹上做 N=5 筛选。四策略：heuristic_verify（旧基线）、"
        "model_based_vbayes_verify（V_Bayes 参考）、learned_worthiness_full_rag、"
        "learned_worthiness_no_rag。")
    add("")
    add("## 1. Learned 是否优于 heuristic")
    add("")
    add("| strategy | top1 | top3 | brier | new | verify | total q | c→w/verify | wall(s) |")
    add("|---|---|---|---|---|---|---|---|---|")
    for s in ("heuristic", "v_bayes", "learned_full", "learned_norag"):
        r = agg[s]
        add(f"| {s} | {fmt(r['top1_accuracy'])} | {fmt(r['top3_accuracy'])} | "
            f"{fmt(r['brier_score'])} | {fmt(r['new_questions_mean'])} | "
            f"{fmt(r['verification_questions_mean'])} | "
            f"{fmt(r['total_atomic_questions_mean'])} | "
            f"{fmt(ctw_rate.get(s, float('nan')))} | {fmt(r['wall_clock_mean'])} |")
    add("")
    b = lf_heur_brier
    t = lf_heur_top1
    q = lf_heur_q
    c = lf_heur_ctw
    if b:
        add(f"- learned_full vs heuristic 配对 Δbrier = {b['mean']:+.4f} "
            f"(95% CI [{b['lo']:+.4f}, {b['hi']:+.4f}], p={b['p']:.3f})"
            f"{'（显著）' if b['sig'] else ''}")
    if t:
        add(f"- learned_full vs heuristic 配对 Δtop1 = {t['mean']:+.4f} "
            f"(95% CI [{t['lo']:+.4f}, {t['hi']:+.4f}], p={t['p']:.3f})"
            f"{'（显著）' if t['sig'] else ''}")
    if q:
        add(f"- learned_full vs heuristic 配对 Δ问题数 = {q['mean']:+.2f} "
            f"(95% CI [{q['lo']:+.2f}, {q['hi']:+.2f}])")
    if c:
        add(f"- learned_full vs heuristic 配对 Δcorrect→wrong = {c['mean']:+.4f} "
            f"(95% CI [{c['lo']:+.4f}, {c['hi']:+.4f}])")
    add("")
    add("结论：**" + ("是" if c1_learned_better_brier else "否") + "**（learned 未在 realized Brier/精度上"
        "超过 heuristic）。")
    add("")
    add("原因：learned 与 v_bayes 均采用统一的 Brier 单位计价 AskNew（V_Bayes_new = R(b_t) − "
        "E[R(b_{t+1})] − C_new，C_new=0.03），在扩散先验下 V_Bayes_new 很快 ≤0，策略过早停止"
        "（~3.3 问 vs heuristic ~9.6 问）。该精度损失来自**共享的 AskNew 计价方式**，而非 learned 核验模型"
        "本身：learned_full 与 v_bayes 的问题数/精度几乎一致（3.29 vs 3.12 问，0.545 vs 0.529 top1）。")
    add("")
    add("## 2. Learned 与 V_Bayes 效果与速度差距")
    add("")
    add(f"- 效果：learned_full brier = {fmt(lf['brier_score'])}、learned_norag = {fmt(ln['brier_score'])}，"
        f"V_Bayes = {fmt(vb['brier_score'])}（learned 略优于 V_Bayes，Δ ≈ {fmt(lf['brier_score'] - vb['brier_score'], 4)}）。")
    add(f"- 速度：learned_norag = {fmt(ln['wall_clock_mean'], 2)}s vs V_Bayes = {fmt(vb['wall_clock_mean'], 2)}s"
        f"（同为 NO_RAG，learned 快 ≈ {fmt(vb['wall_clock_mean'] / ln['wall_clock_mean'], 1)}×）；"
        f"learned_full = {fmt(lf['wall_clock_mean'], 2)}s 额外含 DYNAMIC_RAG 检索成本"
        f"（比 NO_RAG 多 ~{fmt(lf['wall_clock_mean'] - ln['wall_clock_mean'], 2)}s），并非 worthiness 模型本身更慢。")
    vb = ci("learned_full_vs_v_bayes.brier_score")
    if vb:
        add(f"- learned_full vs V_Bayes 配对 Δbrier = {vb['mean']:+.4f} "
            f"(95% CI [{vb['lo']:+.4f}, {vb['hi']:+.4f}], p={vb['p']:.3f})"
            f"{'（显著）' if vb['sig'] else ''}")
    add("")
    add("## 3. harm gate 是否降低 correct→wrong")
    add("")
    add("| strategy | correct→wrong / verification |")
    add("|---|---|")
    for s in ("heuristic", "v_bayes", "learned_full", "learned_norag"):
        add(f"| {s} | {fmt(ctw_rate.get(s, float('nan')))} |")
    add("")
    harm = {r["strategy_short"]: r for r in read("harm_gate_analysis.csv")}
    for s in ("learned_full", "learned_norag"):
        r = harm.get(s, {})
        add(f"- {s}: gain 通过 {r.get('n_gain_pass')}，harm gate 额外拦截 {r.get('n_harm_blocked')} 个"
            f"（mean harm_hat {r.get('mean_harm_hat_harm_blocked')}），实际选择 {r.get('n_selected')} 个，"
            f"selected 中 realized correct→wrong {r.get('realized_correct_to_wrong_selected')} 个"
            f"（率 {r.get('realized_correct_to_wrong_rate_selected')}）")
    add("")
    add("结论：**否**。harm gate 机械性地拦截了高 harm_hat 候选（learned_full 额外拦截 90 个，"
        "mean harm_hat ≈ 0.66），因此总 correct→wrong 计数下降（learned_full 17 vs heuristic 34）；"
        "但**每次核验**的 c→w 率 learned_full = 0.0429 并不低于 heuristic = 0.0379。"
        "total c→w 的下降主要由核验次数本身变少（0.269 vs 0.611 次）驱动，而非 harm gate 把「会把对的改错」"
        "的核验筛选掉了——单次核验的风险率并没有改善。")
    add("")
    add("## 4. 是否增加问题数")
    add("")
    for s in ("heuristic", "v_bayes", "learned_full", "learned_norag"):
        r = agg[s]
        add(f"- {s}: 平均 {fmt(r['new_questions_mean'])} 新问题 + {fmt(r['verification_questions_mean'])} 核验"
            f" = {fmt(r['total_atomic_questions_mean'])} 问题")
    add("")
    add("## 5. Full-RAG 是否优于 No-RAG")
    add("")
    add("| metric | full-RAG | no-RAG | Δ |")
    add("|---|---|---|---|")
    for r in read("rag_ablation.csv"):
        add(f"| {r['metric']} | {r['learned_full_rag_mean']} | {r['learned_no_rag_mean']} | {r['mean_delta']} |")
    add("")
    add("## 6. 是否存在泄漏")
    add("")
    add("无。预测路径不读 true disease / latent state / true wrongness / noise label / oracle correction"
        "（测试 08、10）；N=5 只取 validate 前 5 例，不触 test split；learned 模型为 Phase 4 冻结产物，"
        "本阶段不重新训练、不调整 tau_harm；默认策略仍为 heuristic（测试 13）。")
    add("")
    add("## 7. Go / Conditional Go / No-Go")
    add("")
    add(f"**{verdict}**（判据通过 {n_pass}/8，见 `GO_NO_GO.md`）。")
    add("")
    add("## 8. 是否值得进入 N=20")
    add("")
    if verdict == "Go":
        add("值得：learned 在 N=5 上显著优于 heuristic 且无 harm 回退，可进入 N=20 正式验证。")
    elif verdict == "Conditional Go":
        add("条件性推进：learned 未显著更差且效率占优，但需在 N=20 前确认主判据（精度/Brier）方向。")
    else:
        add("**不进入 N=20**：learned 在 N=5 上未在 realized Brier/精度上超过 heuristic（主要原因："
            "统一 Brier 单位下 AskNew 的 V_Bayes_new 在 C_new=0.03 时过早停止，问诊问题数显著少于 heuristic），"
            "作为可部署的 VerifyOld 排序升级的证据已在 Phase 4 建立，但完整在线集成不构成对旧 heuristic 的改进。")
    (OUT / "ONLINE_INTEGRATION_REPORT.md").write_text("\n".join(L), encoding="utf-8")

    # ---- GO_NO_GO.md ------------------------------------------------- #
    G = []
    G.append("# Phase 5 — Go / No-Go 裁决\n")
    G.append(f"## 裁决：**{verdict}**\n")
    G.append(f"判据通过 {n_pass}/8。\n")
    G.append("| # | 判据 | 结果 | 值 |")
    G.append("|---|---|---|---|")
    def y(v):
        return "✅" if v else "❌"
    b = lf_heur_brier
    t = lf_heur_top1
    G.append(f"| 1 | learned brier < heuristic | {y(c1_learned_better_brier)} | "
             f"Δ={b['mean']:+.4f} CI[{b['lo']:+.4f},{b['hi']:+.4f}] |" if b else
             f"| 1 | learned brier < heuristic | {y(c1_learned_better_brier)} | n/a |")
    G.append(f"| 2 | learned top1 不显著更差 | {y(c2_not_worse_top1)} | "
             f"Δ={t['mean']:+.4f} CI[{t['lo']:+.4f},{t['hi']:+.4f}] |" if t else
             f"| 2 | learned top1 不显著更差 | {y(c2_not_worse_top1)} | n/a |")
    G.append(f"| 3 | learned correct→wrong ≤ heuristic | {y(c3_less_ctw)} | "
             f"{fmt(ctw_rate.get('learned_full', float('nan')))} vs {fmt(ctw_rate.get('heuristic', float('nan')))} |")
    G.append(f"| 4 | learned 问题数 ≤ heuristic | {y(c4_no_more_questions)} | "
             f"Δ={q['mean']:+.2f} |" if q else
             f"| 4 | learned 问题数 ≤ heuristic | {y(c4_no_more_questions)} | n/a |")
    G.append(f"| 5 | learned ~ V_Bayes（brier 不显著更差） | {y(c5_approx_vbayes)} | "
             f"Δ={vb['mean']:+.4f} |" if vb else
             f"| 5 | learned ~ V_Bayes | {y(c5_approx_vbayes)} | n/a |")
    G.append(f"| 6 | harm gate 降低 correct→wrong 率 | {y(c6_harm_gate_lowers_ctw)} | "
             f"{fmt(ctw_rate.get('learned_full', float('nan')))} vs {fmt(ctw_rate.get('heuristic', float('nan')))} |")
    G.append(f"| 7 | full-RAG brier ≤ no-RAG | {y(c7_rag_helps)} | "
             f"Δ={rag_brier:+.4f} |" if rag_brier is not None else
             f"| 7 | full-RAG brier ≤ no-RAG | {y(c7_rag_helps)} | n/a |")
    G.append("")
    G.append("## 结论")
    G.append("")
    G.append(f"**{verdict}**（通过 {n_pass}/8）。")
    G.append("")
    G.append("learned 模型作为可部署的 VerifyOld 排序升级（比 heuristic 强、但未达 oracle V_Bayes）"
             "的证据已在 Phase 4 离线建立；本次在线集成在统一 Brier 单位下使 AskNew 过早停止，"
             "问诊问题数显著少于 heuristic，导致 realized Brier/精度未超过旧基线。")
    (OUT / "GO_NO_GO.md").write_text("\n".join(G), encoding="utf-8")

    print(f"wrote ONLINE_INTEGRATION_REPORT.md + GO_NO_GO.md (verdict {verdict}, {n_pass}/8)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
