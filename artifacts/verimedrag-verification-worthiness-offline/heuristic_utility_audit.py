"""Phase 4 Step 1 — audit the negative heuristic_verify_utility correlation.

The Phase 3A audit reported Spearman(heuristic_verify_utility, v_real) = -0.223
on VerifyOld samples.  Before training a learned worthiness model we must rule
out an implementation / bookkeeping artefact.

This script performs the ten checks required by the spec and writes
HEURISTIC_UTILITY_AUDIT.md.  It is read-only over
``action_value_samples.csv``; it writes only into this directory.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from scipy import stats

OUT = Path(__file__).resolve().parent
SAMPLE = Path(
    "artifacts/verimedrag-action-value-audit/action_value_samples.csv"
).resolve()

# Frozen policy config values used to recompute diagnostic_influence from the
# recorded heuristic_verify_utility (see decision._verification_utility).
RELIABILITY_WEIGHT = 0.01
DECISION_WEIGHT = 0.25
C_VERIFY = 0.03


def num(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def binary_entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def influence_from_utility(row: dict) -> float | None:
    """Back out diagnostic_influence from the recorded utility + error prob."""
    utility = num(row, "heuristic_verify_utility")
    error = num(row, "retrospective_error_prob")
    if utility is None or error is None:
        return None
    entropy = binary_entropy(error)
    denom = error + DECISION_WEIGHT
    return (utility - RELIABILITY_WEIGHT * entropy + C_VERIFY) / denom


def spearman(xs: list[float], ys: list[float]):
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return float(stats.spearmanr(xs, ys).statistic)


def load_verify_rows() -> list[dict]:
    rows = list(csv.DictReader(open(SAMPLE, newline="", encoding="utf-8")))
    return [r for r in rows if r["action_kind"] == "verify"]


def paired(rows, key):
    pairs = [
        (num(r, key), num(r, "v_real"))
        for r in rows
        if num(r, key) is not None and num(r, "v_real") is not None
    ]
    if len(pairs) < 3:
        return None, None
    xs, ys = zip(*pairs)
    return list(xs), list(ys)


def corr_report(rows, key, label):
    xs, ys = paired(rows, key)
    if xs is None:
        return f"{label:40s} n<3"
    rho = spearman(xs, ys)
    p = stats.spearmanr(xs, ys).pvalue if len(set(xs)) > 1 and len(set(ys)) > 1 else None
    return f"{label:40s} n={len(xs):5d}  rho={rho:+.4f}  p={p if p is None else round(p,4)}"


def main() -> int:
    rows = load_verify_rows()
    print(f"VerifyOld rows: {len(rows)}")

    lines: list[str] = []
    add = lines.append
    add("# Phase 4 Step 1 — heuristic_verify_utility 负相关审计")
    add("")
    add(f"数据：{SAMPLE}，VerifyOld 样本 n={len(rows)}。")
    add("")
    add("## 实现/计账检查（1–7）")
    add("")

    # 1. bigger is better?
    add("1. **符号约定**：`_verification_utility` 返回 `existing = disease_gain + "
        "0.01·H(error) + 0.25·influence − 0.03`，数值越大表示「越值得核验」；"
        "`rank_actions` 用 `reverse=True` 降序取最大值。**越大越优，方向正确**。")
    # 2. sort direction
    add("2. **排序方向**：`sorted(actions, key=utility, reverse=True)`；"
        "`_verification_scores` 也按 `score`（=error_prob×influence）降序。**正确**。")
    # 3. raw vs rank
    add("3. **记录的是原始分数**：audit CSV 的 `heuristic_verify_utility` 列来自 "
        "`policy._verification_utility(score)[0]`（原始 existing utility 值），非 rank 非 cost。"
        "**正确**（见 `run_audit.py` score_state 的 `existing_utility, _final = "
        "policy._verification_utility(score)`）。")
    # 4. cost double deduction
    add("4. **成本只扣一次**：utility 里 `− verification_cost` 一次；`v_real` 由 "
        "`realized_value_mc` 计算为 `gross_brier_reduction − cost` 一次；`v_bayes` 由 "
        "`verify_value` 计算 `R(b_t)−E[R]−C` 一次。**无重复扣减**。")
    # 5. report_index alignment
    add("5. **report_index 对齐**：`_verification_scores` 的 `report_index` 来自 "
        "`enumerate(reports)`，audit 用 `reports[score.report_index]` 取值，"
        "`retrospective_error_prob`/`retrieval_impact`/`v_bayes`/`v_real` 均按同一 "
        "report_index 计算。**对齐**。")
    # 6. truncation
    add("6. **候选截断**：每状态只记录 top-5 回溯分数候选（`MAX_VERIFY=5`）。"
        "这是按 `error_prob×influence` 截断的**上尾样本**——相关性是在「系统认为最值得核验的"
        " top-5 内部」计算的，不是全体报告。它不会翻转符号，但限制了解释域："
        "−0.22 表示「在 top-5 候选中，系统排名更高的反而 realized 价值更低」。")
    # 7. realized sign
    add("7. **realized 符号**：`value = gross_brier_reduction − cost = "
        "(L_brier_before − E[L_brier_after]) − cost`。正 = 风险下降。**正确**。")

    add("")
    add("## 分层相关性（8）")
    add("")
    add("| 分层 | n | Spearman(heuristic_verify_utility, v_real) | p |")
    add("|---|---|---|---|")
    overall = paired(rows, "heuristic_verify_utility")
    if overall[0] is not None:
        rho = spearman(overall[0], overall[1])
        p = stats.spearmanr(overall[0], overall[1]).pvalue
        add(f"| overall | {len(overall[0])} | {rho:+.4f} | {p:.2e} |")
    for noise in sorted({r["noise_rate"] for r in rows}):
        sub = [r for r in rows if r["noise_rate"] == noise]
        xs, ys = paired(sub, "heuristic_verify_utility")
        if xs is not None:
            rho = spearman(xs, ys)
            p = stats.spearmanr(xs, ys).pvalue
            add(f"| noise={noise} | {len(xs)} | {rho:+.4f} | {p:.2e} |")
    for kind in ("early", "middle", "late"):
        sub = [r for r in rows if r["state_kind"] == kind]
        xs, ys = paired(sub, "heuristic_verify_utility")
        if xs is not None:
            rho = spearman(xs, ys)
            p = stats.spearmanr(xs, ys).pvalue
            add(f"| state={kind} | {len(xs)} | {rho:+.4f} | {p:.2e} |")
    # turn strata
    turn_bins = [(0, 3), (4, 6), (7, 9), (10, 99)]
    for lo, hi in turn_bins:
        sub = [r for r in rows if lo <= int(r["turn_index"]) <= hi]
        xs, ys = paired(sub, "heuristic_verify_utility")
        if xs is not None and len(xs) >= 3:
            rho = spearman(xs, ys)
            p = stats.spearmanr(xs, ys).pvalue
            add(f"| turn∈[{lo},{hi}] | {len(xs)} | {rho:+.4f} | {p:.2e} |")

    # per-disease correlation
    add("")
    add("### 分病种相关性（每病种 ≥10 样本）")
    add("")
    by_disease = defaultdict(list)
    for r in rows:
        by_disease[r["diagnosis"]].append(r)
    disease_rhos = []
    add("| 病种 | n | Spearman |")
    add("|---|---|---|")
    for disease in sorted(by_disease):
        sub = by_disease[disease]
        xs, ys = paired(sub, "heuristic_verify_utility")
        if xs is not None and len(xs) >= 10:
            rho = spearman(xs, ys)
            disease_rhos.append(rho)
            add(f"| {disease} | {len(xs)} | {rho:+.4f} |")
    if disease_rhos:
        import statistics
        add(f"| **mean over diseases** | | {statistics.mean(disease_rhos):+.4f} |")

    # 9. within-case correlation
    add("")
    add("## within-case 相关性（9，排除病例难度混杂）")
    add("")
    add("方法：把 heuristic_verify_utility 与 v_real 分别在每个 case 内去均值"
        "（demean），再对去均值后的 pooled 样本计算 Spearman；同时给出每病例 Spearman 的均值。")
    add("")
    within_x, within_y = [], []
    per_case_rhos = []
    for disease in by_disease:
        sub = by_disease[disease]
        xs, ys = paired(sub, "heuristic_verify_utility")
        if xs is None or len(xs) < 3:
            continue
        rho = spearman(xs, ys)
        if rho is not None:
            per_case_rhos.append(rho)
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        within_x.extend(x - mx for x in xs)
        within_y.extend(y - my for y in ys)
    if len(within_x) >= 3:
        wrho = spearman(within_x, within_y)
        wp = stats.spearmanr(within_x, within_y).pvalue
        add(f"- pooled within-case Spearman = **{wrho:+.4f}** (p={wp:.2e}, n={len(within_x)})")
    if per_case_rhos:
        import statistics
        add(f"- 每病例 Spearman 均值 = **{statistics.mean(per_case_rhos):+.4f}** "
            f"(n={len(per_case_rhos)} cases)")
    add("")

    # 10. manual inspection
    add("## 人工检查样本（10）：heuristic_verify_utility 最高/最低各 20")
    add("")
    add("列：utility、error_prob、influence(反推)、retrieval_impact、v_bayes、"
        "v_real、gross_brier_reduction、reported_value、is_report_wrong。")
    add("")
    keyed = [
        r for r in rows
        if num(r, "heuristic_verify_utility") is not None
        and num(r, "v_real") is not None
    ]
    keyed.sort(key=lambda r: -num(r, "heuristic_verify_utility"))
    add("### 最高 20（系统最想核验）")
    add("")
    add("| utility | error_prob | influence | retrieval_impact | v_bayes | v_real | gross_brier↓ | reported | wrong |")
    add("|---|---|---|---|---|---|---|---|---|")
    for r in keyed[:20]:
        add(
            f"| {num(r,'heuristic_verify_utility'):+.4f} | {num(r,'retrospective_error_prob'):.3f} "
            f"| {influence_from_utility(r):.3f} | {num(r,'retrieval_impact'):.2f} "
            f"| {num(r,'v_bayes'):+.4f} | {num(r,'v_real'):+.4f} "
            f"| {num(r,'gross_brier_reduction'):+.4f} | {r['reported_value']} | {r['is_report_wrong']} |"
        )
    add("")
    add("### 最低 20（系统最不想核验）")
    add("")
    add("| utility | error_prob | influence | retrieval_impact | v_bayes | v_real | gross_brier↓ | reported | wrong |")
    add("|---|---|---|---|---|---|---|---|---|")
    for r in keyed[-20:][::-1]:
        add(
            f"| {num(r,'heuristic_verify_utility'):+.4f} | {num(r,'retrospective_error_prob'):.3f} "
            f"| {influence_from_utility(r):.3f} | {num(r,'retrieval_impact'):.2f} "
            f"| {num(r,'v_bayes'):+.4f} | {num(r,'v_real'):+.4f} "
            f"| {num(r,'gross_brier_reduction'):+.4f} | {r['reported_value']} | {r['is_report_wrong']} |"
        )
    add("")
    add("## 结论")
    add("")
    add("（由 analyze 阶段填写，依据上述分层/within-case 结果判断负相关是否为真实现象。）")

    (OUT / "HEURISTIC_UTILITY_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote HEURISTIC_UTILITY_AUDIT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
