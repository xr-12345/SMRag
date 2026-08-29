# Method: reliability-aware active medical interviewing

## State and observation model

For disease (D), contextual clinical state (Z_j), latent report mode (M_t), and observed
answer (Y_t), the implemented generative factorization is

\[
D \rightarrow Z_j \rightarrow M_t \rightarrow Y_t.
\]

The disease model estimates (P(D)) and (P(Z_j\mid D)). `FeatureKey` makes context part of
variable identity, so activity-, time-, or location-specific states are not automatically treated as
contradictions. `UNKNOWN` is a report, never a latent negative state.

For a cue-conditioned report-mode prior, disease updating marginalizes the mode:

\[
P(D=d\mid Y_t,H_{t-1}) \propto P(D=d\mid H_{t-1})
\sum_z P(Z_j=z\mid d,H_{t-1})
\sum_m P(Y_t\mid z,m)P(m\mid H_{t-1}).
\]

Repeated reports about the same `FeatureKey` update one shared (Z_j), rather than sampling a new
clinical truth at every repetition.

## Observable trust gate

The conservative heuristic gate uses answer surprisal, direct same-context conflict, certainty, and
unknown status. One non-conflicting surprising answer is capped at 0.15 misreport prior; direct
conflict permits at most 0.40. The history gate avoids current-answer confirmation bias by using only
previous unknown/uncertain cues.

The learned gate is calibrated logistic regression over runtime-observable features:

- answer surprisal;
- number of direct conflicts;
- uncertain and unknown indicators;
- fraction of earlier unknown/uncertain reports;
- current top-disease probability and normalized disease entropy;
- diagnostic impact of accepting the answer under the fixed ordinary channel;
- extraction uncertainty, when an upstream parser supplies confidence.

All features are computed before accepting the answer. Training uses a caller-owned train/calibration
split; `fit-gate` groups the calibration split by case ID. The gate is capped by the same single-answer
and conflict limits. `OracleMisreportGate` reads explicitly privileged simulator metadata and is
available only as a named ceiling baseline.

## Joint action policy

At each turn the policy exposes

\[
\mathcal A_t=\mathcal Q_{new}\cup\mathcal Q_{verify}\cup\{stop\}.
\]

For a new question (q), the implemented one-step score is

\[
U_{new}(q)=EIG_D(q)-\beta c_q.
\]

For an old report (i), the policy computes label-free leave-one-out report error probability
(p_{err,i}), and diagnostic influence

\[
\Delta_i=\tfrac12\sum_d|b(d)-b_{-i}(d)|.
\]

The current transparent verification proxy is

\[
U_{verify}(i)=p_{err,i}\Delta_i
+\alpha h_2(p_{err,i})+\gamma\Delta_i-c_{verify}.
\]

This is deliberately described as a one-step observable proxy, not an exact POMDP value function.
Verification is available only after a configurable number of previous unknown/uncertain cues, each
report can be verified once, and a global verification budget prevents loops. Agreement is counted
once; disagreement or either unknown resolves to `UNKNOWN`, then the belief history is replayed.

## Reliability- and safety-aware stopping

Confident stopping requires all of the following:

1. top posterior or top-1/top-2 margin reaches its configured threshold;
2. no acquisition action exceeds the configured marginal-utility threshold;
3. no unverified report has leave-one-out risk-by-influence above the suspicious-report threshold;
4. an optional dataset-supplied safety constraint is clear.

`RequiredFeatureSafetyConstraint` and `HighRiskDiseaseSafetyConstraint` provide mechanisms for
source-backed red-flag coverage. The repository intentionally supplies no default DDXPlus red-flag
mapping. When no valid safety labels are configured, results must not be described as safety-aware
clinical triage.

## Observation schema boundary

`Observation` now carries optional extraction confidence, source, time, activity, location,
duration, severity, conflict status, and evidence span. Existing structured data remains backward
compatible because every new field is optional/defaulted. The repository still lacks a natural-
language extractor that populates and validates these fields.

