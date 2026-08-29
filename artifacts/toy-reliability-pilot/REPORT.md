# Toy reliability-aware policy pilot

This is a synthetic smoke/pilot experiment, not a DDXPlus or clinical result.
It uses disjoint toy-data generation seeds for model fitting and evaluation.
Frozen policy for this run: posterior=0.85, margin=0.7, min_utility=0.08, verification_cost=0.03, minimum_history_cues=1.

| strategy | noise | top-1 | new | verify | total | Brier | ECE | premature stop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive_history | 0.00 | 0.808 | 3.77 | 0.00 | 3.77 | 0.276 | 0.023 | 0.081 |
| adaptive_history | 0.10 | 0.761 | 4.07 | 0.00 | 4.07 | 0.337 | 0.032 | 0.089 |
| adaptive_history | 0.20 | 0.727 | 4.28 | 0.00 | 4.28 | 0.382 | 0.046 | 0.087 |
| adaptive_history | 0.30 | 0.693 | 4.51 | 0.00 | 4.51 | 0.416 | 0.055 | 0.080 |
| full_two_layer | 0.00 | 0.810 | 4.27 | 0.00 | 4.27 | 0.267 | 0.037 | 0.052 |
| full_two_layer | 0.10 | 0.769 | 4.50 | 0.00 | 4.50 | 0.322 | 0.041 | 0.058 |
| full_two_layer | 0.20 | 0.738 | 4.59 | 0.00 | 4.59 | 0.377 | 0.052 | 0.076 |
| full_two_layer | 0.30 | 0.694 | 4.78 | 0.00 | 4.78 | 0.415 | 0.054 | 0.066 |
| joint_new_verify_stop | 0.00 | 0.817 | 4.13 | 0.21 | 4.34 | 0.259 | 0.031 | 0.052 |
| joint_new_verify_stop | 0.10 | 0.778 | 4.36 | 0.34 | 4.69 | 0.320 | 0.040 | 0.062 |
| joint_new_verify_stop | 0.20 | 0.742 | 4.48 | 0.44 | 4.91 | 0.364 | 0.035 | 0.071 |
| joint_new_verify_stop | 0.30 | 0.702 | 4.68 | 0.52 | 5.20 | 0.405 | 0.047 | 0.068 |
| ordinary_eig_reliable | 0.00 | 0.803 | 3.72 | 0.00 | 3.72 | 0.279 | 0.036 | 0.086 |
| ordinary_eig_reliable | 0.10 | 0.761 | 4.02 | 0.00 | 4.02 | 0.337 | 0.057 | 0.089 |
| ordinary_eig_reliable | 0.20 | 0.733 | 4.19 | 0.00 | 4.19 | 0.383 | 0.061 | 0.100 |
| ordinary_eig_reliable | 0.30 | 0.698 | 4.42 | 0.00 | 4.42 | 0.420 | 0.072 | 0.090 |
| random_reliable | 0.00 | 0.802 | 4.56 | 0.00 | 4.56 | 0.282 | 0.018 | 0.090 |
| random_reliable | 0.10 | 0.764 | 4.80 | 0.00 | 4.80 | 0.332 | 0.050 | 0.080 |
| random_reliable | 0.20 | 0.729 | 4.96 | 0.00 | 4.96 | 0.395 | 0.058 | 0.107 |
| random_reliable | 0.30 | 0.686 | 5.02 | 0.00 | 5.02 | 0.440 | 0.083 | 0.101 |

## Interpretation against the frozen criteria

This pilot validates action plumbing and paired accounting. The table below reports joint-policy minus `full_two_layer`; negative question deltas are better.

| noise | accuracy delta (pp) | total-question delta | relative Brier change |
|---:|---:|---:|---:|
| 0.00 | +0.67 | +0.07 | -3.1% |
| 0.10 | +0.89 | +0.20 | -0.6% |
| 0.20 | +0.44 | +0.32 | -3.5% |
| 0.30 | +0.78 | +0.42 | -2.3% |

The pilot **does not** meet the frozen requirement to save at least one total atomic question.
It is a synthetic toy trend check, not evidence for DDXPlus performance or clinical safety.
