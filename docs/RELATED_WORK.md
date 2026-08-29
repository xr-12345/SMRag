# Related-work verification and claim boundaries

Checked: 2026-08-29.  This note favors publisher pages, paper pages, and official code.  It is a
claim-audit document, not a systematic review.

## Comparison matrix

| Work | Primary mechanism | Patient-answer assumption / noise handling | Relation to this project |
|---|---|---|---|
| [MedRAG, WWW 2025](https://arxiv.org/abs/2502.04413) and [official code](https://github.com/SNOWTEAM2023/MedRAG) | KG-elicited RAG, hierarchical disease reasoning, diagnostic reports and follow-up suggestions | Does not expose this repository's probabilistic latent report channel | Use only the published reciprocal-degree formula as a clearly named structured adaptation. The inspected public repository exposes KG retrieval and diagnosis scripts, but no directly reusable closed-loop degree-based interviewer; do not call our comparator a full reproduction. |
| [APP, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.142/) | Guideline-grounded multi-turn dialogue with Bayesian active learning and expected-entropy question choice | Simulated patient profiles; the main contribution is grounded, transparent and empathetic active diagnosis | Establishes that Bayesian active question selection and entropy minimization are not novel by themselves. |
| [MedClarify, 2026 preprint](https://arxiv.org/abs/2602.17308) | Differential diagnosis, case-specific question generation, DEIG selection, Bayesian updating and confidence/margin stopping | Focuses on acquiring missing diagnostic evidence rather than persistent report-reliability belief | Closest active-questioning comparator. Our distinction must be reliability-conditioned posterior/action/stop behavior, not EIG. |
| [ProMed, 2025 preprint](https://arxiv.org/abs/2508.13514) and [official code](https://github.com/hxxding/ProMed) | Shapley Information Gain, MCTS trajectory construction, supervised initialization and RL reward allocation | Interactive patient model supplies answers used for training/evaluation; no equivalent explicit report-reliability belief was found in the inspected overview | Shows that learned proactive questioning, MCTS and information-value rewards are prior art. A precise simulator-answer audit is still required before claiming that all answers are guaranteed correct. |
| [PaMis, ACL Findings 2025](https://aclanthology.org/2025.findings-acl.135/) | Dialogue entity graph, structural-entropy misreport detection, and clarifying-question generation | Misreport is an explicit target | Primary adjacent work. Our `pamis_style_s30` is only a detect-then-clarify mechanism proxy; it omits the graph, structural entropy and language generation and must stay labeled as such. |
| [FollowupQ, ACL 2025](https://aclanthology.org/2025.acl-long.1226/) | Multi-agent generation from patient messages and EHR; 2,300 clinician-authored follow-up questions | Clarifies incomplete/ambiguous asynchronous messages; optimizes useful question generation and provider burden | Motivates natural-language/EHR question quality and burden, but does not substitute for diagnosis/reliability belief evaluation. |
| [KG-Followup, EACL Findings 2026](https://aclanthology.org/2026.findings-eacl.43/) | KG-augmented active in-context learning, EHR- and differential-guided question generation and consolidation | Targets relevant follow-up question sets | Relevant question-generation comparator; KG/RAG is not our main novelty. |
| [Cost-Bounded Active Classification with POMDPs](https://arxiv.org/abs/1810.00097) and [Constrained Active Classification](https://arxiv.org/abs/2008.04768) | Belief-state classification with acquisition cost, confidence/safety constraints and finite horizon | General noisy observations; medical diagnosis is an application | Establishes prior art for cost-sensitive belief-state acquisition and safe stopping. |
| [MedExAgent, 2026 preprint](https://arxiv.org/abs/2605.07058) | Clinical diagnosis as a POMDP with patient questions, exam tool calls and diagnosis action; trained with SFT and RL | Seven patient noise types and three exam noise types in a noisy/incomplete environment | Strong recent overlap. We cannot claim the first noisy medical POMDP or first joint acquisition/diagnosis action space. The defensible focus is an explicit, interpretable report-reliability posterior; one-time verification of a selected old report; and testing how that belief changes Bayesian diagnosis and stopping. |

## Defensible problem distinction

Active diagnostic systems such as APP and MedClarify primarily optimize which missing fact to ask
for.  PaMis primarily detects and mitigates a suspected misreport.  MedExAgent jointly learns
question/exam/diagnosis actions in a noisy environment.  The remaining project-specific hypothesis
is narrower:

> Maintaining an explicit report-reliability belief for each acquired fact, and using its expected
> diagnostic impact to compare a new question, a one-time verification of an old report, and
> stopping, improves the accuracy--burden trade-off under patient-report noise.

This hypothesis requires evidence.  It is not established by the existing retrospective validation
results because those results do not rank verification against a contemporaneous new-question
action and do not use a new untouched confirmatory split.

## Claims that remain prohibited

- first Bayesian, information-gain, entropy-based, POMDP, or cost-sensitive active diagnosis;
- first patient-misreport detection or clarification generation;
- full reproduction or defeat of MedRAG or PaMis using the current proxies;
- first noisy clinical multi-action agent, in view of MedExAgent;
- clinical safety, triage, or red-flag coverage without source-backed labels and external validation;
- real-world patient robustness from DDXPlus structured synthetic simulations.

## Open source-verification tasks

- Inspect ProMed's patient prompts and reward code at a frozen commit before characterizing its
  answer-correctness assumption in the paper table.
- Check whether MedClarify and MedExAgent release runnable code and frozen evaluation data; record
  version/commit identifiers for any reproduced baseline.
- A true PaMis baseline needs its official code/data or an independently verified reimplementation
  of entity-graph construction and structural entropy.  Until then, keep the current proxy separate.
- Treat works released after method freezing as contemporaneous related work and explicitly discuss
  overlap instead of retroactively expanding novelty claims.

