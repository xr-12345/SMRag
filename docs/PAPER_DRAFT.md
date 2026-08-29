# Paper-facing draft (pre-results, claim-audited)

## Problem formulation

We study reliability-aware active medical interviewing. A patient has disease (D) and contextual
clinical states (Z), but the interviewer observes reports (Y_t) generated through latent reporting
modes (M_t), including certain, uncertain, unknown, and misreported responses. Given dialogue
history (H_t), the agent maintains diagnostic and report-reliability beliefs and chooses among
asking a new question, verifying one previous report, and stopping:

\[
\mathcal A_t=\mathcal Q_{new}\cup\mathcal Q_{verify}\cup\{stop\}.
\]

The objective trades diagnostic loss, new-question burden, verification burden, and unsafe or
premature stopping. Unlike ordinary active diagnosis, a previously acquired fact remains uncertain
and may later become worth checking. Unlike standalone misreport detection, reliability changes the
disease posterior and subsequent acquisition/stopping decisions.

## Method

We use a disease-conditioned categorical model for contextual clinical states and a probabilistic
report channel that marginalizes four latent response modes. Unknown reports are not clinical
negatives, and context is included in feature identity to prevent conditional differences from being
treated as direct contradictions. Repeated reports update one shared latent clinical state.

For new evidence, the agent uses expected reduction in disease entropy minus question burden. For
old reports, it computes a leave-one-out error probability and the total-variation influence of the
report on the disease posterior. A conservative history gate enables verification only after earlier
unknown or uncertain cues. The one-step policy then compares new-question utility with a transparent
risk-and-influence verification proxy. A verified report is incorporated once if answers agree and
resolved to unknown if they conflict. Stopping additionally requires that no unverified high-impact
suspicious report remains. Optional safety constraints can require source-backed red-flag coverage or
bound unresolved high-risk disease mass; no default clinical mapping is assumed.

We also implement a calibrated logistic report gate over online-observable features and a separately
named oracle ceiling. The learned gate is trained and Platt-calibrated only on caller-designated
training/validation events. A single non-conflicting surprising answer is probability-capped to limit
confirmation bias.

## Experimental design

The intended DDXPlus protocol estimates disease-state probabilities from training data, selects gate
and policy settings on validation data, and evaluates a frozen policy on a new untouched holdout or
external dataset. Patient noise is paired by case, feature, occurrence, rate, and seed so overlapping
questions receive identical simulated answers across policies. Baselines include random questioning,
reliable-answer EIG, the fixed two-layer model, no-misreport inference, history gating, the transparent
MedRAG-RDC formula adaptation, and the PaMis-style detect-then-clarify proxy. Learned and oracle gates
are additional ablations, not replacements for a true PaMis implementation.

Primary outcomes jointly report Top-1/Top-k accuracy, new and verification questions, total atomic
questions, turns, Brier, ECE, report-gate AUROC/AUPRC, unnecessary verification, mitigation,
premature stopping, and paired multi-seed uncertainty. The frozen success criteria require no more
than 0.5 percentage-point accuracy loss, no more than 5% relative Brier degradation, and at least one
total question saved relative to the fixed two-layer model. Safety metrics require a frozen external
risk-label source.

At present, the new DDXPlus experiment cannot be run because the local patient archives and fitted
model are absent. A disjoint-seed synthetic toy pilot validates the implementation but fails the
efficiency criterion: the joint policy improves Top-1/Brier and slightly reduces new questions, while
verification increases total atomic questions. This negative pilot result motivates a stricter
expected-value verification model and must not be presented as a successful main experiment.

## Contributions (conditional on confirmatory evidence)

1. **Task formulation.** We formulate active diagnosis in which acquired patient reports retain
   explicit reliability uncertainty and may be selectively reacquired.
2. **Report modeling.** We integrate contextual clinical states, unknown/uncertain responses,
   contradictions, extraction confidence, and latent misreport probability into Bayesian diagnosis.
3. **Decision policy.** We provide an interpretable one-step policy comparing new evidence,
   one-time verification, and reliability-aware stopping, with explicit anti-confirmation-bias caps.
4. **Evaluation protocol.** We separate new questions from verification cost and evaluate diagnosis,
   calibration, reliability discrimination, premature stopping, and paired Pareto behavior.

These are proposed contributions until a frozen external/untouched evaluation succeeds. We do not
claim the first Bayesian active diagnosis, EIG interviewer, noisy medical POMDP, misreport detector,
clarification generator, KG/RAG medical system, or complete reproduction of MedRAG/PaMis. APP,
MedClarify, ProMed, PaMis, and MedExAgent materially constrain the novelty statement; see
`docs/RELATED_WORK.md`.

