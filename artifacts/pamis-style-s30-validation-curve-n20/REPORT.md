# PaMis-style clarification baseline: validation report

## Scope

- Dataset split: DDXPlus validation only.
- Balanced sample: 980 cases, up to 20 cases for each of 49 diseases.
- Answer-noise seed and case-sampling seed: 2026.
- Total report-turn budget: 15. A clarification consumes one full turn.
- Posterior stopping thresholds: 0.60, 0.70, 0.80, 0.85, 0.90, 0.95.
- The detector threshold was selected on the validation split from surprisal
  cutoffs 2.5, 3.0, 3.5, and 4.0. The frozen variant is `pamis_style_s30`.

This is a structured **PaMis-style proxy**, not a reproduction of PaMis. It
implements the mechanism "detect an anomalous answer, then ask for controlled
clarification" using answer surprisal. It does not reproduce PaMis's dialogue
entity graph, structural-entropy detector, or natural-language generator.

## Protocol

1. Score a certain, known report before updating the disease posterior:
   `surprisal = -log P(report | dialogue history)`.
2. If surprisal is at least 3.0 and a turn remains, repeat the same evidence
   question through the same patient-noise channel.
3. If both reports agree, incorporate the evidence once. If they disagree or
   either report is unknown, abstain by incorporating `UNKNOWN`.
4. Treat the original/clarification pair atomically when applying a posterior
   stopping threshold.

The repeated answer receives no hidden reliability advantage. This conservative
choice prevents the baseline from gaining accuracy through privileged simulator
information.

## Main results at posterior threshold 0.85

| Noise | Accuracy | Total turns | Clarifications/case | Detection precision | Detection recall | Mitigation after detection |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 94.39% | 5.48 | 0.047 | 0.0% | N/A | N/A |
| 10% | 88.37% | 5.98 | 0.063 | 9.7% | 3.7% | 100% |
| 20% | 81.22% | 6.52 | 0.069 | 22.1% | 4.9% | 100% |
| 30% | 70.71% | 6.92 | 0.089 | 44.8% | 7.4% | 100% |

The detector becomes more precise as answer noise rises, but it catches only a
small fraction of harmful misreports before the dialogue stops. Every detected
harmful report was neutralized under this protocol; detection recall, not
post-detection resolution, is the bottleneck.

## Equal-turn comparison

Reference accuracy was linearly interpolated on each reference curve at the
candidate's average turn budget. No extrapolation was used. These interpolated
figures are descriptive rather than confidence intervals or hypothesis tests.

At threshold 0.85:

| Noise | PaMis-style accuracy | Adaptive-history accuracy at same turns | History minus PaMis-style | No-misreport accuracy at same turns | No-misreport minus PaMis-style |
|---:|---:|---:|---:|---:|---:|
| 0% | 94.39% | 94.60% | +0.21 pp | 94.58% | +0.19 pp |
| 10% | 88.37% | 87.98% | -0.38 pp | 88.06% | -0.30 pp |
| 20% | 81.22% | 81.62% | +0.40 pp | 80.98% | -0.24 pp |
| 30% | 70.71% | 71.97% | +1.26 pp | 71.00% | +0.29 pp |

Across all comparable stopping points, the mean adaptive-history advantage was
+0.25, -0.42, +0.10, and +0.66 percentage points at 0%, 10%, 20%, and 30%
noise, respectively. The clarification proxy is therefore a competitive
efficiency baseline, but does not show a stable accuracy advantage. Its gains
over the no-misreport baseline are also small and disappear at 30% noise.

## Decision

Do not spend the independent test set on this proxy yet. The next meaningful
iteration should improve **detector recall under a clarification budget**, while
keeping the current same-channel clarification and turn accounting fixed. A
stronger paper-facing comparison should either implement PaMis's graph-based
structural-entropy detector or port the comparison to a dataset with the
required dialogue entity graph and natural-language utterances.
