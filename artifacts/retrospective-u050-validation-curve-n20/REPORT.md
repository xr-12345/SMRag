# Retrospective clarification validation report

## Frozen method

- Detector: leave-one-out report error probability multiplied by diagnostic
  influence (total variation between the full and leave-one-out disease
  posteriors).
- Utility threshold: 0.05.
- Retrospective clarification budget: one per case.
- Detector channel: one fixed population-level answer channel for every noise
  condition. Injected simulator noise and latent truth are not visible to the
  detector.
- If clarification lowers confidence below the stopping threshold, EIG
  questioning resumes until the threshold or the total 15-turn budget is met.

The candidate was selected on the DDXPlus validation split from risk-only,
utility thresholds 0.025/0.05, budgets 1/2, and an online-plus-retrospective
hybrid. The independent test split was not used.

## Threshold-selection result at posterior threshold 0.85

| Noise | Accuracy | Total turns | Clarifications/case | Precision | Recall |
|---:|---:|---:|---:|---:|---:|
| 0% | 94.90% | 5.80 | 0.35 | 0.0% | N/A |
| 10% | 88.78% | 6.49 | 0.40 | 11.6% | 26.3% |
| 20% | 83.78% | 7.26 | 0.42 | 22.6% | 28.3% |
| 30% | 74.49% | 7.68 | 0.45 | 30.9% | 24.6% |

Relative to the online surprisal `pamis_style_s30` baseline:

| Noise | Accuracy difference | 95% case-bootstrap CI | Extra turns | Exact McNemar p |
|---:|---:|---:|---:|---:|
| 0% | +0.51 pp | [0.00, 1.02] pp | +0.32 | 0.1250 |
| 10% | +0.41 pp | [-0.92, 1.63] pp | +0.51 | 0.6440 |
| 20% | +2.55 pp | [1.02, 4.08] pp | +0.74 | 0.00126 |
| 30% | +3.78 pp | [1.94, 5.71] pp | +0.76 | 0.000110 |

Thus, the recall improvement translates into paired diagnostic-accuracy gains
at 20% and 30% answer noise. The high-precision risk-only detector reached
60.6% precision and 16.6% recall at 30% noise but did not improve accuracy;
the more aggressive two-clarification hybrid reached 53.3% recall but required
8.65 turns and had 29.1% precision.

## Equal-turn comparison over the full stopping curve

Accuracy was linearly interpolated on each reference curve at the candidate's
average turn budget. No extrapolation was used; these comparisons are
descriptive, not hypothesis tests.

Mean reference-minus-candidate accuracy across comparable curve points:

| Noise | Online s30 | Adaptive history | No-misreport |
|---:|---:|---:|---:|
| 0% | +1.19 pp | +1.42 pp | +1.42 pp |
| 10% | +1.70 pp | +1.21 pp | +1.44 pp |
| 20% | -0.43 pp | -0.37 pp | -0.67 pp |
| 30% | -1.51 pp | -0.55 pp | -1.36 pp |

At the candidate's 0.85 operating point, it exceeds adaptive history by 0.75
pp at 20% noise and 1.16 pp at 30% noise at the same interpolated turn budget,
but trails by about 0.40--0.42 pp at 0% and 10% noise.

## Conclusion

Retrospective auditing solves much of the original low-recall failure and is
useful in moderate/high-noise dialogues. Its remaining weakness is excessive
false clarification in clean and low-noise dialogues. The next iteration
should activate retrospective auditing from an online history-reliability
estimate instead of enabling it unconditionally.
