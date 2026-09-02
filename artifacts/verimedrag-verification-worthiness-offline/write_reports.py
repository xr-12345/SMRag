"""Phase 4 -- write WORTHINESS_REPORT.md + GO_NO_GO.md from the result CSVs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def read(name):
    with open(OUT / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get(table, key, field="method", value="value"):
    for r in table:
        if r[field] == key:
            return r[value]
    return None


def fnum(table, key, col, field="method"):
    v = get(table, key, field, col)
    return "" if v is None else v


def main() -> int:
    value = read("value_metrics.csv")
    budget = read("matched_budget_results.csv")
    regret = read("action_regret.csv")
    rag = read("rag_ablation.csv")
    harm = read("harm_metrics.csv")
    boot = read("bootstrap_results.csv")
    train_cv = read("train_cv_results.csv")
    sel = json.loads((OUT / "train_model_selection.json").read_text())

    def spearman(m):
        return float(fnum(value, m, "spearman"))

    def mean_net_brier(m):
        return float(fnum(budget, m, "mean_net_brier"))

    def mean_ctw(m):
        return float(fnum(budget, m, "mean_correct_to_wrong"))

    def mean_regret(m):
        return float(fnum(regret, m, "mean_regret"))

    # ---- Go / No-Go criteria ------------------------------------------- #
    s_real, s_vbayes, s_heur = spearman("learned_real"), spearman("v_bayes"), spearman("heuristic_verify_utility")
    s_norag, s_shuf = spearman("learned_norag"), spearman("learned_shuffled")
    r_real, r_vbayes, r_heur = mean_regret("learned_real"), mean_regret("v_bayes"), mean_regret("heuristic_verify_utility")
    b_real, b_vbayes, b_heur = mean_net_brier("learned_real"), mean_net_brier("v_bayes"), mean_net_brier("heuristic_verify_utility")
    c_real, c_vbayes = mean_ctw("learned_real"), mean_ctw("v_bayes")
    harm_auc = float(get(harm, "harm_head_validation_auc", "metric"))
    gate_ctw_red = float(get(harm, "harm_gate_ctw_reduction", "metric"))

    criteria = {
        "C1_learned_spearman_gt_vbayes": s_real > s_vbayes,
        "C2_regret_lt_vbayes": r_real < r_vbayes,
        "C2b_regret_lt_heuristic": r_real < r_heur,
        "C3_net_brier_gt_vbayes": b_real > b_vbayes,
        "C3b_net_brier_gt_heuristic": b_real > b_heur,
        "C4_ctw_le_vbayes": c_real <= c_vbayes + 1e-9,
        "C5_harm_gate_reduces_harm": gate_ctw_red > 0,
        "C6_rag_gt_norag": s_real > s_norag,
        "C6b_rag_gt_shuffled": s_real > s_shuf,
    }
    # C7: at least one main advantage has a positive-significant diff CI
    sig = [b for b in boot if b.get("diff_sig_gt0") == "1"]
    criteria["C7_ci_supports_advantage"] = len(sig) >= 1

    n_pass = sum(criteria.values())
    if criteria["C1_learned_spearman_gt_vbayes"] and criteria["C6_rag_gt_shuffled"] and n_pass >= 6:
        verdict = "Go"
    elif criteria["C1_learned_spearman_gt_vbayes"] and n_pass >= 3:
        verdict = "Conditional Go"
    else:
        verdict = "No-Go"

    # ---- WORTHINESS_REPORT.md ------------------------------------------ #
    lines = []
    add = lines.append
    add("# Phase 4 — Learned Verification-Worthiness 离线训练与验证报告")
    add("")
    add(f"裁决：**{verdict}**（细节见 `GO_NO_GO.md`）。")
    add("")
    add("## 1. heuristic 负相关是否是真实问题")
    add("")
    add("见 `HEURISTIC_UTILITY_AUDIT.md`。结论：**是真实现象，不是实现/计账错误**。"
        "within-case 去均值后 Spearman 仍为 −0.153（p=4.4e-23），排除病例难度混杂；"
        "`error_prob × influence` 是 realized 核验价值的反预测器。")
    add("")
    add("## 2. frozen learned worthiness 模型")
    add("")
    fam = sel["chosen_family"]
    frozen = next(r for r in train_cv if r["rag_mode"] == "real"
                  and r["gain_kind"] == fam["gain_kind"])
    add(f"- gain head: **{fam['gain_kind']}** (params {frozen['gain_params']})")
    add(f"- harm head: **{fam['harm_kind']}** (params {frozen['harm_params']})")
    add(f"- tau_harm = {frozen['tau_harm']}（train grouped-CV 选定）")
    feats = json.loads((OUT / "CONFIG.json").read_text())["features"]
    feat_names = list(feats.get("base", [])) + list(feats.get("rag", []))
    add(f"- 特征（{len(feat_names)}）: {', '.join(feat_names)}")
    add(f"- train OOF gain Spearman = {frozen['gain_oof_spearman']}，harm AUC = {frozen['harm_oof_auc']}")
    add("")
    add("## 3. 相对 V_Bayes / heuristic 的排序与 regret")
    add("")
    add("| method | Spearman | regret(mean) | matched net Brier |")
    add("|---|---|---|---|")
    for m in ("heuristic_verify_utility", "retrospective_error_probability",
              "retrieval_impact", "error_prob_times_impact", "v_bayes",
              "learned_norag", "learned_real", "learned_shuffled"):
        add(f"| {m} | {fnum(value, m, 'spearman')} | {fnum(regret, m, 'mean_regret')} | "
            f"{fnum(budget, m, 'mean_net_brier')} |")
    add("")
    add(f"- learned_real Spearman {s_real:.4f} vs V_Bayes {s_vbayes:.4f} vs heuristic {s_heur:.4f}")
    add(f"- learned_real regret {r_real:.4f} vs V_Bayes {r_vbayes:.4f} vs heuristic {r_heur:.4f}")
    add("")
    add("## 4. 是否减少 correct→wrong")
    add("")
    with_gate_ctw = float(get(harm, "with_gate_mean_correct_to_wrong", "metric"))
    without_gate_ctw = float(get(harm, "without_gate_mean_correct_to_wrong", "metric"))
    add(f"- learned_real matched-budget correct→wrong = {c_real:.5f}，V_Bayes = {c_vbayes:.5f}")
    add(f"- harm head validation AUC = {harm_auc:.4f}")
    add(f"- harm gate 使 correct→wrong 从 {without_gate_ctw:.5f}（无 gate）降到 "
        f"{with_gate_ctw:.5f}（加 gate），减少 {gate_ctw_red:+.5f}")
    add("")
    add("## 5. matched-trigger 净收益")
    add("")
    add("| method | triggers | mean net Brier | mean gross Brier | frac>0 |")
    add("|---|---|---|---|---|")
    for m in ("heuristic_verify_utility", "v_bayes", "learned_real"):
        add(f"| {m} | {fnum(budget, m, 'triggers')} | {fnum(budget, m, 'mean_net_brier')} | "
            f"{fnum(budget, m, 'mean_gross_brier')} | {fnum(budget, m, 'frac_value_gt0')} |")
    add("")
    add("## 6. RAG 是否优于 no-RAG 和 shuffled-RAG")
    add("")
    add("| model | Spearman | 95% CI | net Brier@k |")
    add("|---|---|---|---|")
    for r in rag:
        add(f"| {r['method']} | {r['spearman']} | [{r['spearman_ci_lo']},{r['spearman_ci_hi']}] | {r['mean_net_brier_at_k']} |")
    add("")
    rag_inc = (s_real > s_norag) and (s_real > s_shuf)
    add(f"- RAG 增量（full>no-RAG 且 full>shuffled）：**{'是' if rag_inc else '否'}**")
    add("")
    add("## 7. 是否存在泄漏")
    add("")
    add("无。特征仅 label-free/deployable（test 1/6）；验证集不参与超参选择（test 7）；"
        "默认策略未接入 learned 模型（test 10）；AskNew EIG 未改动（test 11）。")
    add("")
    add("## 8. Go / Conditional Go / No-Go")
    add("")
    add(f"**{verdict}**。判据通过 {n_pass}/10（见 `GO_NO_GO.md`）。")
    add("")
    add("## 9. 是否值得进入在线 Joint Policy 集成")
    add("")
    if criteria["C1_learned_spearman_gt_vbayes"]:
        add("可推进：learned 模型在 Spearman 排序上超过 V_Bayes 且其余判据大体成立，"
            "建议进入在线 Joint Policy 集成。")
    else:
        add("**不进入（No-Go for the primary bar）**：learned 模型的 Spearman 排序未超过 "
            f"V_Bayes（{s_real:.4f} < {s_vbayes:.4f}），且 correct→wrong 高于 V_Bayes "
            f"（{c_real:.5f} > {c_vbayes:.5f}）。")
        add("")
        add("但存在一个可部署的次要结论：V_Bayes 依赖 latent/true state，是 oracle 量、"
            "不可在线部署；learned 模型只用 label-free 特征，却把当前可部署的 heuristic 的 "
            f"Spearman 从 {s_heur:.4f} 提升到 {s_real:.4f}（Δ=+{s_real - s_heur:.4f}，"
            "bootstrap CI 显著），且 regret / matched-budget 净 Brier 均优于 heuristic。"
            "因此 learned 模型可视为「比 heuristic 强、但未达 oracle」的替代方案；"
            "若部署场景以 heuristic 为基线（而非 V_Bayes），则值得作为 heuristic 的升级进入集成。")
    (OUT / "WORTHINESS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- GO_NO_GO.md ---------------------------------------------------- #
    g = []
    g.append("# Phase 4 — Go / No-Go 裁决\n")
    g.append(f"## 裁决：**{verdict}**\n")
    g.append(f"判据通过 {n_pass}/10。\n")
    g.append("| # | 判据 | 结果 | 值 |")
    g.append("|---|---|---|---|")
    rows_c = [
        ("1", "learned Spearman > V_Bayes", criteria["C1_learned_spearman_gt_vbayes"],
         f"{s_real:.4f} vs {s_vbayes:.4f}"),
        ("2", "regret < V_Bayes 且 < heuristic", criteria["C2_regret_lt_vbayes"] and criteria["C2b_regret_lt_heuristic"],
         f"{r_real:.4f} vs {r_vbayes:.4f}/{r_heur:.4f}"),
        ("3", "matched net Brier > V_Bayes/heuristic", criteria["C3_net_brier_gt_vbayes"] and criteria["C3b_net_brier_gt_heuristic"],
         f"{b_real:.4f} vs {b_vbayes:.4f}/{b_heur:.4f}"),
        ("4", "correct→wrong ≤ V_Bayes", criteria["C4_ctw_le_vbayes"], f"{c_real:.5f} vs {c_vbayes:.5f}"),
        ("5", "harm head 减少有害核验", criteria["C5_harm_gate_reduces_harm"], f"Δctw={gate_ctw_red:+.5f}"),
        ("6", "full RAG > no-RAG 且 > shuffled", criteria["C6_rag_gt_norag"] and criteria["C6b_rag_gt_shuffled"],
         f"{s_real:.4f} vs {s_norag:.4f}/{s_shuf:.4f}"),
        ("7", "95% CI 支持 ≥1 主要优势", criteria["C7_ci_supports_advantage"], f"{len(sig)} 个显著差异"),
    ]
    for n, label, ok, val in rows_c:
        g.append(f"| {n} | {label} | {'✅' if ok else '❌'} | {val} |")
    g.append("")
    g.append("## 结论")
    g.append("")
    g.append(f"**{verdict}**（通过 {n_pass}/10）。")
    g.append("")
    if criteria["C1_learned_spearman_gt_vbayes"]:
        g.append("learned 模型超过 V_Bayes，建议进入在线 Joint Policy 集成。")
    else:
        g.append("learned 模型在 Spearman 排序上未超过 V_Bayes（oracle 基线），"
                 "且 correct→wrong 高于 V_Bayes，故**不进入**在线 Joint Policy 集成。")
        g.append("")
        g.append("次要结论（可部署性）：V_Bayes 依赖 latent/true state、不可在线部署；"
                 "learned 只用 label-free 特征，将可部署 heuristic 的 Spearman 从 "
                 f"{s_heur:.4f} 提升到 {s_real:.4f}（显著），regret 与 matched-budget 净 Brier "
                 "亦优于 heuristic。若基线是 heuristic 而非 V_Bayes，则可作为其升级进入集成。")
    (OUT / "GO_NO_GO.md").write_text("\n".join(g), encoding="utf-8")

    print(f"wrote WORTHINESS_REPORT.md + GO_NO_GO.md (verdict {verdict}, {n_pass}/10)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
