"""Phase 8D -- static channel alignment audit (5 components, env vs inference).

Reads the environment's actual parameters (``PatientProfile.from_noise_rate``,
``ChannelParameters.rates``, the simulator's deterministic certainty-cue mapping)
and the inference model's actual parameters (``ChannelParameters.cue_priors``,
``JointReportChannel.reask_mode_prior``), and quantifies each of the five channel
components for mismatch.  No dialogue is run; this is a pure parameter audit.

Components (spec section 2):
  1. report mode prior           P(E)
  2. certainty cue mechanism      P(C | E)  ->  P(E | C)
  3. first answer channel        P(Y | Z, E)
  4. re-ask mode transition      P(E' | E)   (the (1-rho) prior term)
  5. joint re-ask channel        P(Y, Y' | Z)  (composed from 1/2/4)

``env_cue_conditioned_prior`` is the one function reused by the offline paired
audit: it derives the environment's true P(E | cue) from P(E) + the deterministic
cue mechanism + the confusion rates.

Red lines: read-only; writes only into this artifact directory; no git commit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from powerful_medrag.channel import AnswerChannel, ChannelParameters, ReportMode
from powerful_medrag.schema import CertaintyCue
from powerful_medrag.simulator import PatientProfile

OUT = Path(__file__).resolve().parent
NOISE_RATES = (0.2, 0.3)
MODES = (ReportMode.CERTAIN, ReportMode.UNCERTAIN, ReportMode.UNKNOWN, ReportMode.MISREPORTED)
CUES = (CertaintyCue.NONE, CertaintyCue.UNCERTAIN, CertaintyCue.CERTAIN)


def env_cue_conditioned_prior(
    mode_prior: dict[ReportMode, float],
    rates: dict[ReportMode, object],
) -> dict[CertaintyCue, dict[ReportMode, float]]:
    """The environment's true P(E | cue) implied by its deterministic cue map.

    The simulator (``simulator.py`` / ``joint_patient_simulator.py``) sets the
    observable certainty cue deterministically from the sampled value and mode::

        value == UNKNOWN            -> NONE
        mode == UNCERTAIN           -> UNCERTAIN
        otherwise (value != UNKNOWN)-> CERTAIN

    so ``P(cue | mode)`` is ``rates[mode].unknown`` for NONE, the residual for
    UNCERTAIN / CERTAIN according to the mode, and ``P(E | cue)`` follows by
    Bayes.  ``rates`` may be the ``ChannelParameters().rates`` mapping.
    """
    cue_given_mode: dict[ReportMode, dict[CertaintyCue, float]] = {}
    for mode in MODES:
        unknown_mass = rates[mode].unknown
        cue_given_mode[mode] = {
            CertaintyCue.NONE: unknown_mass,
            CertaintyCue.UNCERTAIN: (
                (1.0 - unknown_mass) if mode is ReportMode.UNCERTAIN else 0.0
            ),
            CertaintyCue.CERTAIN: (
                (1.0 - unknown_mass) if mode is not ReportMode.UNCERTAIN else 0.0
            ),
        }
    cue_marginal = {cue: 0.0 for cue in CUES}
    for mode in MODES:
        for cue in CUES:
            cue_marginal[cue] += mode_prior[mode] * cue_given_mode[mode][cue]
    result: dict[CertaintyCue, dict[ReportMode, float]] = {}
    for cue in CUES:
        denom = cue_marginal[cue]
        result[cue] = {
            mode: (
                (mode_prior[mode] * cue_given_mode[mode][cue] / denom)
                if denom > 0
                else 0.0
            )
            for mode in MODES
        }
    return result


def _kl(p: dict, q: dict) -> float:
    total = 0.0
    for key in p:
        pk = p[key]
        if pk <= 0:
            continue
        qk = max(q.get(key, 0.0), 1e-12)
        total += pk * math.log(pk / qk)
    return total


def _max_abs(p: dict, q: dict) -> float:
    return max(abs(p[k] - q.get(k, 0.0)) for k in p)


def _vec(modes: dict[ReportMode, float]) -> dict[str, float]:
    return {mode.value: round(modes[mode], 6) for mode in MODES}


def main() -> int:
    rates = ChannelParameters().rates
    inference_cue_priors = ChannelParameters().cue_priors

    audit: dict[str, object] = {}

    # -- component 1: unconditional mode prior P(E) -------------------------- #
    comp1 = {}
    for noise in NOISE_RATES:
        comp1[f"env_noise_{noise}"] = _vec(PatientProfile.from_noise_rate(noise).mode_prior)
    comp1["inference_default"] = _vec(dict(inference_cue_priors[CertaintyCue.NONE]))
    audit["component_1_mode_prior"] = comp1

    # -- component 2: cue mechanism P(C|E) -> P(E|C) ------------------------- #
    comp2 = {}
    for noise in NOISE_RATES:
        env_prior = PatientProfile.from_noise_rate(noise).mode_prior
        env_cond = env_cue_conditioned_prior(env_prior, rates)
        comp2[f"env_noise_{noise}"] = {
            cue.value: _vec(env_cond[cue]) for cue in CUES
        }
    comp2["inference_default"] = {
        cue.value: _vec(dict(inference_cue_priors[cue])) for cue in CUES
    }
    audit["component_2_cue_conditioned_prior"] = comp2

    # -- component 3: first-answer channel P(Y|Z,E) --------------------------- #
    comp3 = {}
    for mode in MODES:
        r = rates[mode]
        comp3[mode.value] = {
            "correct": r.correct, "unknown": r.unknown, "wrong": r.wrong
        }
    comp3["env_vs_inference"] = "identical -- both use ChannelParameters().rates"
    audit["component_3_first_answer_rates"] = comp3

    # -- component 4: re-ask transition prior P(e') -------------------------- #
    comp4 = {}
    for noise in NOISE_RATES:
        comp4[f"env_noise_{noise}"] = _vec(PatientProfile.from_noise_rate(noise).mode_prior)
    comp4["inference_reask_mode_prior"] = _vec(
        dict(inference_cue_priors[CertaintyCue.NONE])
    )
    audit["component_4_reask_prior"] = comp4

    # -- component 5: joint channel (composed) -------------------------------- #
    # Example: binary feature {absent, present}, z=present, y=present, y'=present.
    # The joint P(y,y'|z) under env-matched prior vs inference default, at rho=0.
    comp5 = {}
    states = ("absent", "present")
    z, y, yp = "present", "present", "present"
    for noise in NOISE_RATES:
        env_prior = PatientProfile.from_noise_rate(noise).mode_prior
        env_cond = env_cue_conditioned_prior(env_prior, rates)
        env_channel = AnswerChannel(ChannelParameters(rates=rates, cue_priors=env_cond))
        inf_channel = AnswerChannel()
        env_first = env_channel.marginal_probability(y, z, states, CertaintyCue.CERTAIN)
        inf_first = inf_channel.marginal_probability(y, z, states, CertaintyCue.CERTAIN)
        env_joint = env_first * env_channel.marginal_probability(
            yp, z, states, CertaintyCue.NONE, mode_prior=env_prior
        )
        inf_joint = inf_first * inf_channel.marginal_probability(
            yp, z, states, CertaintyCue.NONE
        )
        comp5[f"env_noise_{noise}"] = {
            "P(y|z)": round(env_first, 6),
            "P(y,y'|z)@rho0": round(env_joint, 6),
        }
        comp5[f"inference_vs_env_noise_{noise}"] = {
            "P(y|z)": round(inf_first, 6),
            "P(y,y'|z)@rho0": round(inf_joint, 6),
        }
    audit["component_5_joint_example"] = comp5

    # -- mismatch summary ----------------------------------------------------- #
    mismatch = {}
    for noise in NOISE_RATES:
        env_prior = PatientProfile.from_noise_rate(noise).mode_prior
        env_cond = env_cue_conditioned_prior(env_prior, rates)
        inf_default = dict(inference_cue_priors[CertaintyCue.NONE])
        mismatch[f"noise_{noise}"] = {
            "P(E)": {
                "kl_env_into_inf": round(_kl(dict(env_prior), inf_default), 6),
                "kl_inf_into_env": round(_kl(inf_default, dict(env_prior)), 6),
                "max_abs_diff": round(_max_abs(dict(env_prior), inf_default), 6),
            },
            "P(E|NONE)": {
                "kl_env_into_inf": round(
                    _kl(env_cond[CertaintyCue.NONE], inf_default), 6
                ),
                "max_abs_diff": round(
                    _max_abs(env_cond[CertaintyCue.NONE], inf_default), 6
                ),
            },
            "P(E|CERTAIN)": {
                "kl_env_into_inf": round(
                    _kl(env_cond[CertaintyCue.CERTAIN],
                        dict(inference_cue_priors[CertaintyCue.CERTAIN])), 6
                ),
                "max_abs_diff": round(
                    _max_abs(env_cond[CertaintyCue.CERTAIN],
                             dict(inference_cue_priors[CertaintyCue.CERTAIN])), 6
                ),
            },
            "P(E|UNCERTAIN)": {
                "kl_env_into_inf": round(
                    _kl(env_cond[CertaintyCue.UNCERTAIN],
                        dict(inference_cue_priors[CertaintyCue.UNCERTAIN])), 6
                ),
                "max_abs_diff": round(
                    _max_abs(env_cond[CertaintyCue.UNCERTAIN],
                             dict(inference_cue_priors[CertaintyCue.UNCERTAIN])), 6
                ),
            },
        }
    audit["mismatch_summary"] = mismatch

    (OUT / "channel_alignment_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _write_markdown(audit)
    print(json.dumps(mismatch, indent=2, ensure_ascii=False))
    return 0


def _write_markdown(audit: dict[str, object]) -> None:
    def row(mode_vals: dict[str, float]) -> str:
        return " | ".join(f"{mode_vals[m.value]:.4f}" for m in MODES)

    header = "| cue | CERTAIN | UNCERTAIN | UNKNOWN | MISREPORTED |"

    lines: list[str] = []
    lines.append("# Channel Alignment Audit (Phase 8D §二)\n")
    lines.append("静态参数审计：环境 vs 推断的 5 个通道组件。\n")

    lines.append("## 组件 1 — 报告模式先验 P(E)\n")
    lines.append("| 来源 | CERTAIN | UNCERTAIN | UNKNOWN | MISREPORTED |")
    lines.append("|---|---|---|---|---|")
    c1 = audit["component_1_mode_prior"]
    for key in ("env_noise_0.2", "env_noise_0.3", "inference_default"):
        lines.append(f"| {key} | {row(c1[key])} |")

    lines.append("\n## 组件 2 — cue 机制 P(E|cue)\n")
    for cue in ("none", "uncertain", "certain"):
        lines.append(f"\n### cue = {cue}\n")
        lines.append(header)
        lines.append("|---|---|---|---|---|")
        c2 = audit["component_2_cue_conditioned_prior"]
        for key in ("env_noise_0.2", "env_noise_0.3", "inference_default"):
            lines.append(f"| {key} | {row(c2[key][cue])} |")

    lines.append("\n## 组件 3 — 首答通道 P(Y|Z,E)（env 与推断相同，未失配）\n")
    lines.append("| mode | correct | unknown | wrong |")
    lines.append("|---|---|---|---|")
    c3 = audit["component_3_first_answer_rates"]
    for mode in ("certain", "uncertain", "unknown", "misreported"):
        lines.append(
            f"| {mode} | {c3[mode]['correct']} | {c3[mode]['unknown']} | {c3[mode]['wrong']} |"
        )

    lines.append("\n## 组件 4 — 复问转移先验 P(e')（(1-rho) 项）\n")
    lines.append("| 来源 | CERTAIN | UNCERTAIN | UNKNOWN | MISREPORTED |")
    lines.append("|---|---|---|---|---|")
    c4 = audit["component_4_reask_prior"]
    for key in ("env_noise_0.2", "env_noise_0.3", "inference_reask_mode_prior"):
        lines.append(f"| {key} | {row(c4[key])} |")

    lines.append("\n## 失配摘要（KL env→infer / max|Δ|）\n")
    lines.append("| noise | P(E) KL | P(E\\|NONE) KL | P(E\\|CERTAIN) KL | P(E\\|UNCERTAIN) KL |")
    lines.append("|---|---|---|---|---|")
    m = audit["mismatch_summary"]
    for noise in ("noise_0.2", "noise_0.3"):
        e = m[noise]
        lines.append(
            f"| {noise} | {e['P(E)']['kl_env_into_inf']:.4f} | "
            f"{e['P(E|NONE)']['kl_env_into_inf']:.4f} | "
            f"{e['P(E|CERTAIN)']['kl_env_into_inf']:.4f} | "
            f"{e['P(E|UNCERTAIN)']['kl_env_into_inf']:.4f} |"
        )

    (OUT / "CHANNEL_ALIGNMENT_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
