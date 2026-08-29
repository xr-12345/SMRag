"""Command-line entry points for fitting and exercising the prototype."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .data import load_jsonl_cases, load_specs
from .demo_data import generate_toy_cases
from .estimation import DiseaseStateModel
from .evaluation import evaluate_simulated_cohort
from .questioning import PrevalenceQuestionSelector, QuestionSelector
from .schema import CertaintyCue, FeatureKey, Observation
from .simulator import StopRule, StructuredPatientSimulator, run_dialogue


def _fit_command(args: argparse.Namespace) -> int:
    cases = load_jsonl_cases(args.cases)
    specs = load_specs(args.schema)
    model = DiseaseStateModel.fit(
        cases,
        specs,
        alpha=args.alpha,
        disease_prior_alpha=args.disease_prior_alpha,
        hierarchical_strength=args.hierarchical_strength,
    )
    model.save(args.output)
    print(
        f"已保存模型到 {args.output}：{len(model.diseases)} 个疾病，"
        f"{len(model.specs)} 个临床变量，{len(cases)} 个训练病例。"
    )
    return 0


def _inspect_command(args: argparse.Namespace) -> int:
    model = DiseaseStateModel.load(args.model)
    key = FeatureKey.from_token(args.feature)
    payload = {
        disease: model.state_distribution(disease, key)
        for disease in model.diseases
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _demo_command(args: argparse.Namespace) -> int:
    cases, specs = generate_toy_cases(seed=args.seed)
    model = DiseaseStateModel.fit(
        cases,
        specs,
        alpha=1.0,
        hierarchical_strength=2.0,
    )

    patient = StructuredPatientSimulator.sample_case(
        model,
        args.disease,
        seed=args.seed + 1,
    )
    initial = Observation(
        key=FeatureKey("fever"),
        value=args.initial_fever,
        certainty=CertaintyCue.CERTAIN,
    )
    result = run_dialogue(
        patient,
        initial_observations=(initial,),
        stop_rule=StopRule(
            max_questions=args.max_questions,
            posterior_threshold=args.posterior_threshold,
        ),
    )

    print("=== 单病例主动问诊 ===")
    print(f"真实疾病：{result.true_diagnosis}")
    for turn in result.turns:
        spec = model.specs[turn.question.key]
        print(
            f"第 {turn.index} 轮  {spec.question or turn.question.key.display_name()} "
            f"-> {turn.observation.value} "
            f"(EIG={turn.question.expected_information_gain:.3f}, "
            f"误报后验={turn.update.misreport_probability:.3f})"
        )
    print(
        f"停止：{result.stop_reason}；预测：{result.predicted_diagnosis}；"
        f"共 {len(result.turns)} 个追加问题"
    )
    print("疾病后验：")
    for disease, probability in sorted(
        result.belief.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"  {disease:20s} {probability:.3f}")

    if not args.skip_cohort:
        eig_metrics = evaluate_simulated_cohort(
            model,
            cases_per_disease=args.cohort_cases,
            stop_rule=StopRule(
                max_questions=args.max_questions,
                posterior_threshold=args.posterior_threshold,
            ),
            seed=args.seed + 100,
        )
        prevalence_metrics = evaluate_simulated_cohort(
            model,
            cases_per_disease=args.cohort_cases,
            selector_factory=PrevalenceQuestionSelector,
            stop_rule=StopRule(
                max_questions=args.max_questions,
                posterior_threshold=args.posterior_threshold,
            ),
            seed=args.seed + 100,
        )
        print("\n=== 同病例、同噪声的模拟对照 ===")
        print("策略                 准确率   平均问题数   Brier")
        print(
            f"EIG 自适应            {eig_metrics.accuracy:6.3f}   "
            f"{eig_metrics.average_questions:8.2f}   {eig_metrics.average_brier_score:6.3f}"
        )
        print(
            f"流行度/度代理         {prevalence_metrics.accuracy:6.3f}   "
            f"{prevalence_metrics.average_questions:8.2f}   "
            f"{prevalence_metrics.average_brier_score:6.3f}"
        )
        auroc = eig_metrics.misreport_detection_auroc
        print(
            f"EIG 不知道回答率：{eig_metrics.unknown_answer_rate:.3f}；"
            + (f"误报识别 AUROC：{auroc:.3f}" if auroc is not None else "误报识别 AUROC：N/A")
        )
    return 0


def _fit_ddxplus_command(args: argparse.Namespace) -> int:
    from .ddxplus import fit_ddxplus_model

    model = fit_ddxplus_model(
        args.patients,
        args.evidences,
        limit=args.limit,
        alpha=args.alpha,
        disease_prior_alpha=args.disease_prior_alpha,
        hierarchical_strength=args.hierarchical_strength,
    )
    model.save(args.output)
    print(
        f"已保存 DDXPlus 模型到 {args.output}：{len(model.diseases)} 个疾病，"
        f"{len(model.specs)} 个原子临床变量。"
    )
    return 0


def _benchmark_ddxplus_command(args: argparse.Namespace) -> int:
    from .benchmark import (
        average_curve_points,
        plot_accuracy_turns_noise,
        run_paired_curve_experiment,
        save_case_outcomes_csv,
        save_curve_csv,
        save_paired_comparisons_csv,
        summarize_paired_outcomes,
    )
    from .ddxplus import load_ddxplus_condition_graph, sample_balanced_ddxplus_cases

    model = DiseaseStateModel.load(args.model)
    askable_features = {
        key for key, spec in model.specs.items() if spec.askable
    }
    cases = sample_balanced_ddxplus_cases(
        args.patients,
        args.evidences,
        cases_per_disease=args.cases_per_disease,
        seed=args.sample_seed,
        available_features=askable_features,
    )
    sampled_counts = Counter(case.diagnosis for case in cases)
    underfilled = {
        disease: sampled_counts.get(disease, 0)
        for disease in model.diseases
        if sampled_counts.get(disease, 0) < args.cases_per_disease
    }
    print(
        f"测试病例：{len(cases)} 例，{len(sampled_counts)} 个疾病；"
        f"每病目标 {args.cases_per_disease} 例。",
        flush=True,
    )
    if underfilled:
        details = ", ".join(
            f"{disease}={count}" for disease, count in underfilled.items()
        )
        print(
            f"警告：测试集中的下列疾病不足目标数量，保留全部独立病例且不重复采样："
            f"{details}",
            flush=True,
        )
    condition_graph = (
        load_ddxplus_condition_graph(
            args.conditions,
            available_features=askable_features,
        )
        if args.conditions
        else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment_seeds = tuple(args.seeds) if args.seeds else (args.seed,)
    if len(set(experiment_seeds)) != len(experiment_seeds):
        raise ValueError("--seeds cannot contain duplicates")
    all_points = []
    all_outcomes = []
    for experiment_seed in experiment_seeds:
        run_dir = (
            args.output_dir
            if len(experiment_seeds) == 1
            else args.output_dir / f"seed-{experiment_seed}"
        )
        print(
            f"开始 seed={experiment_seed}；固定病例抽样 seed={args.sample_seed}",
            flush=True,
        )
        raw_outcomes = []
        points = run_paired_curve_experiment(
            model,
            cases,
            noise_rates=tuple(args.noise_rates),
            posterior_thresholds=tuple(args.thresholds),
            max_questions=args.max_questions,
            seed=experiment_seed,
            progress=True,
            disease_to_features=condition_graph,
            include_prevalence=not args.no_prevalence,
            workers=args.workers,
            executor_type=args.executor,
            raw_outcomes=raw_outcomes,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        save_curve_csv(points, run_dir / "ddxplus_curve_results.csv")
        save_case_outcomes_csv(raw_outcomes, run_dir / "ddxplus_case_outcomes.csv")
        plot_accuracy_turns_noise(
            points, run_dir / "ddxplus_accuracy_turns_noise.png"
        )
        all_points.extend(points)
        all_outcomes.extend(raw_outcomes)

    points = average_curve_points(all_points)
    if len(experiment_seeds) > 1:
        csv_path = args.output_dir / "ddxplus_multiseed_curve_results.csv"
        figure_path = args.output_dir / "ddxplus_multiseed_accuracy_turns_noise.png"
        paired_path = args.output_dir / "ddxplus_paired_comparisons.csv"
        comparisons = summarize_paired_outcomes(
            all_outcomes,
            baseline_strategy="medrag_rdc" if condition_graph else "prevalence",
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.sample_seed,
        )
        save_curve_csv(points, csv_path)
        save_paired_comparisons_csv(comparisons, paired_path)
        plot_accuracy_turns_noise(points, figure_path)
        print(f"多种子平均曲线：{csv_path}")
        print(f"多种子曲线图：{figure_path}")
        print(f"病例级配对统计：{paired_path}")
    else:
        csv_path = args.output_dir / "ddxplus_curve_results.csv"
        figure_path = args.output_dir / "ddxplus_accuracy_turns_noise.png"
        print(f"原始结果：{csv_path}")
        print(f"曲线图：{figure_path}")
        print(f"病例级配对结果：{args.output_dir / 'ddxplus_case_outcomes.csv'}")
    print("strategy  noise  threshold  accuracy  avg_questions")
    for point in points:
        print(
            f"{point.strategy:10s} {point.noise_rate:5.2f} "
            f"{point.posterior_threshold:9.2f} {point.accuracy:9.3f} "
            f"{point.average_questions:13.2f}"
        )
    return 0


def _ablate_ddxplus_command(args: argparse.Namespace) -> int:
    from .ablation import (
        ABLATION_VARIANTS,
        plot_ablation_curves,
        run_ablation_curve_experiment,
        save_ablation_comparisons_csv,
        summarize_ablation_comparisons,
    )
    from .benchmark import average_curve_points, save_case_outcomes_csv, save_curve_csv
    from .ddxplus import sample_balanced_ddxplus_cases

    model = DiseaseStateModel.load(args.model)
    askable_features = {key for key, spec in model.specs.items() if spec.askable}
    cases = sample_balanced_ddxplus_cases(
        args.patients,
        args.evidences,
        cases_per_disease=args.cases_per_disease,
        seed=args.sample_seed,
        available_features=askable_features,
    )
    sampled_counts = Counter(case.diagnosis for case in cases)
    underfilled = {
        disease: sampled_counts.get(disease, 0)
        for disease in model.diseases
        if sampled_counts.get(disease, 0) < args.cases_per_disease
    }
    print(
        f"消融测试病例：{len(cases)} 例，{len(sampled_counts)} 个疾病；"
        f"每病目标 {args.cases_per_disease} 例。",
        flush=True,
    )
    if underfilled:
        details = ", ".join(
            f"{disease}={count}" for disease, count in underfilled.items()
        )
        print(
            f"警告：下列疾病不足目标数量，不重复采样：{details}",
            flush=True,
        )

    experiment_seeds = tuple(args.seeds) if args.seeds else (args.seed,)
    if len(set(experiment_seeds)) != len(experiment_seeds):
        raise ValueError("--seeds cannot contain duplicates")
    variants = tuple(args.variants) if args.variants else ABLATION_VARIANTS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_points = []
    all_outcomes = []
    for experiment_seed in experiment_seeds:
        run_dir = (
            args.output_dir
            if len(experiment_seeds) == 1
            else args.output_dir / f"seed-{experiment_seed}"
        )
        print(
            f"开始消融 seed={experiment_seed}；固定病例抽样 seed={args.sample_seed}",
            flush=True,
        )
        raw_outcomes = []
        points = run_ablation_curve_experiment(
            model,
            cases,
            variants=variants,
            noise_rates=tuple(args.noise_rates),
            posterior_thresholds=tuple(args.thresholds),
            max_questions=args.max_questions,
            seed=experiment_seed,
            progress=True,
            workers=args.workers,
            executor_type=args.executor,
            raw_outcomes=raw_outcomes,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        save_curve_csv(points, run_dir / "ddxplus_ablation_curve_results.csv")
        save_case_outcomes_csv(
            raw_outcomes, run_dir / "ddxplus_ablation_case_outcomes.csv"
        )
        plot_ablation_curves(points, run_dir / "ddxplus_ablation_curve.png")
        all_points.extend(points)
        all_outcomes.extend(raw_outcomes)

    averaged_points = average_curve_points(all_points)
    if len(experiment_seeds) > 1:
        curve_path = args.output_dir / "ddxplus_ablation_multiseed_curve_results.csv"
        figure_path = args.output_dir / "ddxplus_ablation_multiseed_curve.png"
        comparison_path = args.output_dir / "ddxplus_ablation_comparisons.csv"
        save_curve_csv(averaged_points, curve_path)
        plot_ablation_curves(averaged_points, figure_path)
        print(f"多种子消融曲线：{curve_path}")
        print(f"消融图：{figure_path}")
        if "full_two_layer" in variants:
            comparisons = summarize_ablation_comparisons(
                all_outcomes,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.sample_seed,
            )
            save_ablation_comparisons_csv(comparisons, comparison_path)
            print(f"病例级消融统计：{comparison_path}")
        else:
            print("未包含 full_two_layer；跳过组内基线比较。")
    else:
        print(
            f"消融结果：{args.output_dir / 'ddxplus_ablation_curve_results.csv'}"
        )

    print("strategy             noise  threshold  accuracy  avg_questions")
    for point in averaged_points:
        print(
            f"{point.strategy:20s} {point.noise_rate:5.2f} "
            f"{point.posterior_threshold:9.2f} {point.accuracy:9.3f} "
            f"{point.average_questions:13.2f}"
        )
    return 0


def _analyze_gate_ddxplus_command(args: argparse.Namespace) -> int:
    from .ddxplus import sample_balanced_ddxplus_cases
    from .gate_analysis import (
        collect_gate_events,
        save_gate_events_csv,
        save_gate_summaries_csv,
        summarize_gate_events,
    )

    model = DiseaseStateModel.load(args.model)
    askable_features = {key for key, spec in model.specs.items() if spec.askable}
    cases = sample_balanced_ddxplus_cases(
        args.patients,
        args.evidences,
        cases_per_disease=args.cases_per_disease,
        seed=args.sample_seed,
        available_features=askable_features,
    )
    print(
        f"门控验证病例：{len(cases)} 例；每病目标 {args.cases_per_disease} 例。",
        flush=True,
    )
    events = collect_gate_events(
        model,
        cases,
        noise_rates=tuple(args.noise_rates),
        max_questions=args.max_questions,
        seed=args.seed,
        progress=True,
        workers=args.workers,
        executor_type=args.executor,
    )
    summaries = summarize_gate_events(events)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    event_path = args.output_dir / "ddxplus_gate_events.csv"
    summary_path = args.output_dir / "ddxplus_gate_diagnostics.csv"
    save_gate_events_csv(events, event_path)
    save_gate_summaries_csv(summaries, summary_path)
    print(f"逐轮事件：{event_path}")
    print(f"诊断汇总：{summary_path}")
    print(
        "noise target              positives/events  "
        "surprise_AUC  gate_AUC  precision  recall"
    )
    for row in summaries:
        surprise_auc = (
            f"{row.surprisal_auroc:.3f}"
            if row.surprisal_auroc is not None
            else "  N/A"
        )
        gate_auc = f"{row.gate_auroc:.3f}" if row.gate_auroc is not None else "  N/A"
        precision = (
            f"{row.activation_precision:.3f}"
            if row.activation_precision is not None
            else "  N/A"
        )
        recall = (
            f"{row.activation_recall:.3f}"
            if row.activation_recall is not None
            else "  N/A"
        )
        print(
            f"{row.noise_group:>5s} {row.target:20s} "
            f"{row.positives:6d}/{row.events:<6d} "
            f"{surprise_auc:>12s} {gate_auc:>9s} "
            f"{precision:>9s} {recall:>7s}"
        )
    return 0


def _clarify_ddxplus_command(args: argparse.Namespace) -> int:
    from .ablation import plot_ablation_curves
    from .benchmark import save_case_outcomes_csv, save_curve_csv
    from .clarification import (
        PAMIS_STYLE_VARIANTS,
        run_clarification_curve_experiment,
        save_clarification_summaries_csv,
        summarize_clarification_outcomes,
    )
    from .ddxplus import sample_balanced_ddxplus_cases

    model = DiseaseStateModel.load(args.model)
    askable_features = {key for key, spec in model.specs.items() if spec.askable}
    cases = sample_balanced_ddxplus_cases(
        args.patients,
        args.evidences,
        cases_per_disease=args.cases_per_disease,
        seed=args.sample_seed,
        available_features=askable_features,
    )
    variants = tuple(args.variants) if args.variants else PAMIS_STYLE_VARIANTS
    print(
        f"PaMis-style 验证病例：{len(cases)} 例；"
        f"每病目标 {args.cases_per_disease} 例。",
        flush=True,
    )
    raw_outcomes = []
    points = run_clarification_curve_experiment(
        model,
        cases,
        variants=variants,
        noise_rates=tuple(args.noise_rates),
        posterior_thresholds=tuple(args.thresholds),
        max_questions=args.max_questions,
        seed=args.seed,
        progress=True,
        workers=args.workers,
        executor_type=args.executor,
        raw_outcomes=raw_outcomes,
    )
    summaries = summarize_clarification_outcomes(raw_outcomes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = args.output_dir / "ddxplus_clarification_curve_results.csv"
    outcome_path = args.output_dir / "ddxplus_clarification_case_outcomes.csv"
    diagnostic_path = args.output_dir / "ddxplus_clarification_diagnostics.csv"
    figure_path = args.output_dir / "ddxplus_clarification_curve.png"
    save_curve_csv(points, curve_path)
    save_case_outcomes_csv(raw_outcomes, outcome_path)
    save_clarification_summaries_csv(summaries, diagnostic_path)
    plot_ablation_curves(points, figure_path)
    print(f"澄清曲线：{curve_path}")
    print(f"逐病例结果：{outcome_path}")
    print(f"检测诊断：{diagnostic_path}")
    print("strategy             noise accuracy questions clarify precision recall")
    summary_index = {
        (row.strategy, row.noise_rate, row.posterior_threshold): row
        for row in summaries
    }
    for point in points:
        if point.posterior_threshold != 0.85:
            continue
        summary = summary_index[
            (point.strategy, point.noise_rate, point.posterior_threshold)
        ]
        precision = (
            f"{summary.detection_precision:.3f}"
            if summary.detection_precision is not None
            else "N/A"
        )
        recall = (
            f"{summary.detection_recall:.3f}"
            if summary.detection_recall is not None
            else "N/A"
        )
        print(
            f"{point.strategy:20s} {point.noise_rate:5.2f} "
            f"{point.accuracy:8.3f} {point.average_questions:9.2f} "
            f"{summary.average_clarifications:7.2f} "
            f"{precision:>9s} {recall:>6s}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="powerful-medrag",
        description="两层隐变量医疗问答研究原型",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="从显式状态 JSONL 训练模型")
    fit_parser.add_argument("--cases", required=True, type=Path)
    fit_parser.add_argument("--schema", required=True, type=Path)
    fit_parser.add_argument("--output", required=True, type=Path)
    fit_parser.add_argument("--alpha", type=float, default=1.0)
    fit_parser.add_argument("--disease-prior-alpha", type=float, default=1.0)
    fit_parser.add_argument("--hierarchical-strength", type=float, default=0.0)
    fit_parser.set_defaults(func=_fit_command)

    inspect_parser = subparsers.add_parser("inspect", help="查看某变量的疾病条件分布")
    inspect_parser.add_argument("--model", required=True, type=Path)
    inspect_parser.add_argument("--feature", required=True)
    inspect_parser.set_defaults(func=_inspect_command)

    ddx_parser = subparsers.add_parser(
        "fit-ddxplus", help="流式读取官方 DDXPlus CSV/ZIP 并训练"
    )
    ddx_parser.add_argument("--patients", required=True, type=Path)
    ddx_parser.add_argument("--evidences", required=True, type=Path)
    ddx_parser.add_argument("--output", required=True, type=Path)
    ddx_parser.add_argument("--limit", type=int)
    ddx_parser.add_argument("--alpha", type=float, default=1.0)
    ddx_parser.add_argument("--disease-prior-alpha", type=float, default=1.0)
    ddx_parser.add_argument("--hierarchical-strength", type=float, default=0.0)
    ddx_parser.set_defaults(func=_fit_ddxplus_command)

    benchmark_parser = subparsers.add_parser(
        "benchmark-ddxplus", help="在独立测试集生成准确率—轮数—噪声曲线"
    )
    benchmark_parser.add_argument("--model", required=True, type=Path)
    benchmark_parser.add_argument("--patients", required=True, type=Path)
    benchmark_parser.add_argument("--evidences", required=True, type=Path)
    benchmark_parser.add_argument("--conditions", type=Path)
    benchmark_parser.add_argument("--output-dir", required=True, type=Path)
    benchmark_parser.add_argument("--cases-per-disease", type=int, default=20)
    benchmark_parser.add_argument("--max-questions", type=int, default=15)
    benchmark_parser.add_argument(
        "--noise-rates", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3]
    )
    benchmark_parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.60, 0.70, 0.80, 0.85, 0.90, 0.95],
    )
    benchmark_parser.add_argument("--seed", type=int, default=2026)
    benchmark_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="多个回答噪声种子；病例集合由 --sample-seed 固定",
    )
    benchmark_parser.add_argument("--sample-seed", type=int, default=2026)
    benchmark_parser.add_argument("--workers", type=int, default=1)
    benchmark_parser.add_argument(
        "--executor", choices=("thread", "process"), default="thread"
    )
    benchmark_parser.add_argument("--bootstrap-samples", type=int, default=2000)
    benchmark_parser.add_argument("--no-prevalence", action="store_true")
    benchmark_parser.set_defaults(func=_benchmark_ddxplus_command)

    ablation_parser = subparsers.add_parser(
        "ablate-ddxplus", help="在固定 DDXPlus 病例上运行报告层消融"
    )
    ablation_parser.add_argument("--model", required=True, type=Path)
    ablation_parser.add_argument("--patients", required=True, type=Path)
    ablation_parser.add_argument("--evidences", required=True, type=Path)
    ablation_parser.add_argument("--output-dir", required=True, type=Path)
    ablation_parser.add_argument("--cases-per-disease", type=int, default=100)
    ablation_parser.add_argument("--max-questions", type=int, default=15)
    ablation_parser.add_argument(
        "--noise-rates", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3]
    )
    ablation_parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.60, 0.70, 0.80, 0.85, 0.90, 0.95],
    )
    ablation_parser.add_argument("--seed", type=int, default=2026)
    ablation_parser.add_argument("--seeds", type=int, nargs="+")
    ablation_parser.add_argument("--sample-seed", type=int, default=2026)
    ablation_parser.add_argument("--workers", type=int, default=1)
    ablation_parser.add_argument(
        "--executor", choices=("thread", "process"), default="thread"
    )
    ablation_parser.add_argument("--bootstrap-samples", type=int, default=2000)
    ablation_parser.add_argument(
        "--variants",
        nargs="+",
        choices=(
            "full_two_layer",
            "direct_reliable",
            "unknown_as_negative",
            "no_misreport",
            "adaptive_heuristic",
            "adaptive_sparse",
            "adaptive_history",
        ),
    )
    ablation_parser.set_defaults(func=_ablate_ddxplus_command)

    gate_parser = subparsers.add_parser(
        "analyze-gate-ddxplus",
        help="在 DDXPlus 验证集上诊断误报门控信号",
    )
    gate_parser.add_argument("--model", required=True, type=Path)
    gate_parser.add_argument("--patients", required=True, type=Path)
    gate_parser.add_argument("--evidences", required=True, type=Path)
    gate_parser.add_argument("--output-dir", required=True, type=Path)
    gate_parser.add_argument("--cases-per-disease", type=int, default=20)
    gate_parser.add_argument("--max-questions", type=int, default=15)
    gate_parser.add_argument(
        "--noise-rates", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3]
    )
    gate_parser.add_argument("--seed", type=int, default=2026)
    gate_parser.add_argument("--sample-seed", type=int, default=2026)
    gate_parser.add_argument("--workers", type=int, default=1)
    gate_parser.add_argument(
        "--executor", choices=("thread", "process"), default="thread"
    )
    gate_parser.set_defaults(func=_analyze_gate_ddxplus_command)

    clarification_parser = subparsers.add_parser(
        "clarify-ddxplus",
        help="运行 PaMis-style 异常检测与受控澄清基线",
    )
    clarification_parser.add_argument("--model", required=True, type=Path)
    clarification_parser.add_argument("--patients", required=True, type=Path)
    clarification_parser.add_argument("--evidences", required=True, type=Path)
    clarification_parser.add_argument("--output-dir", required=True, type=Path)
    clarification_parser.add_argument("--cases-per-disease", type=int, default=20)
    clarification_parser.add_argument("--max-questions", type=int, default=15)
    clarification_parser.add_argument(
        "--noise-rates", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3]
    )
    clarification_parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.60, 0.70, 0.80, 0.85, 0.90, 0.95],
    )
    clarification_parser.add_argument(
        "--variants",
        nargs="+",
        choices=(
            "pamis_style_s25",
            "pamis_style_s30",
            "pamis_style_s35",
            "pamis_style_s40",
            "retro_risk_r50_b1",
            "retro_utility_u010_b1",
            "retro_utility_u025_b1",
            "retro_utility_u050_b1",
            "retro_utility_u025_b2",
            "hybrid_s30_u025_b1",
            "hybrid_s30_u025_b2",
        ),
    )
    clarification_parser.add_argument("--seed", type=int, default=2026)
    clarification_parser.add_argument("--sample-seed", type=int, default=2026)
    clarification_parser.add_argument("--workers", type=int, default=1)
    clarification_parser.add_argument(
        "--executor", choices=("thread", "process"), default="thread"
    )
    clarification_parser.set_defaults(func=_clarify_ddxplus_command)

    demo_parser = subparsers.add_parser("demo", help="运行一个可重复的端到端示例")
    demo_parser.add_argument(
        "--disease",
        choices=("influenza", "common_cold", "allergic_rhinitis"),
        default="influenza",
    )
    demo_parser.add_argument(
        "--initial-fever", choices=("present", "absent", "__unknown__"), default="present"
    )
    demo_parser.add_argument("--max-questions", type=int, default=6)
    demo_parser.add_argument("--posterior-threshold", type=float, default=0.85)
    demo_parser.add_argument("--cohort-cases", type=int, default=30)
    demo_parser.add_argument("--skip-cohort", action="store_true")
    demo_parser.add_argument("--seed", type=int, default=13)
    demo_parser.set_defaults(func=_demo_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
