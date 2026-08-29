# Three retrospective clarification methods: validation comparison

## Methods

All methods use the same 980 DDXPlus validation cases, feature-indexed answer
noise, 15-turn total budget, seed 2026, and six posterior stopping thresholds.
The independent test split was not used.

1. `retro_risk_r50_b1`: clarify the report with the highest leave-one-out error
   probability when that probability is at least 0.50; maximum one audit.
2. `retro_utility_u050_b1`: multiply leave-one-out error probability by the
   report's diagnostic influence; threshold 0.05; maximum one audit.
3. `hybrid_s30_u025_b2`: immediately clarify extreme online surprisal, then
   perform retrospective utility auditing at threshold 0.025; maximum two
   retrospective audits. Already clarified reports are excluded.

## Results at posterior threshold 0.85

| Method | Noise | Accuracy | Turns | Clarifications | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Risk only | 0% | 94.39% | 5.45 | 0.02 | 0.0% | N/A |
| Risk only | 10% | 87.55% | 5.98 | 0.08 | 44.9% | 21.7% |
| Risk only | 20% | 80.92% | 6.50 | 0.09 | 49.5% | 14.9% |
| Risk only | 30% | 70.71% | 6.84 | 0.14 | 60.6% | 16.6% |
| Risk x influence | 0% | 94.90% | 5.80 | 0.35 | 0.0% | N/A |
| Risk x influence | 10% | 88.78% | 6.49 | 0.40 | 11.6% | 26.3% |
| Risk x influence | 20% | 83.78% | 7.26 | 0.42 | 22.6% | 28.3% |
| Risk x influence | 30% | 74.49% | 7.68 | 0.45 | 30.9% | 24.6% |
| Hybrid | 0% | 94.49% | 6.18 | 0.76 | 0.0% | N/A |
| Hybrid | 10% | 89.18% | 7.04 | 0.88 | 9.7% | 47.4% |
| Hybrid | 20% | 84.29% | 7.98 | 0.95 | 16.8% | 46.8% |
| Hybrid | 30% | 76.33% | 8.65 | 1.08 | 29.1% | 53.3% |

## What each ablation shows

### Risk only

At 30% noise, risk-only detection has 60.6% precision and 16.6% recall while
using only 0.14 clarification turns per case. However, its 70.71% accuracy is
identical to the online `s30` baseline at threshold 0.85. Across the full curve
it is roughly tied with `s30` and consistently below adaptive history. This is
evidence that detecting likely errors is insufficient unless diagnostic impact
is also considered.

### Risk x influence

Compared with risk only at threshold 0.85, utility auditing improves accuracy
by 1.22, 2.86, and 3.78 percentage points at 10%, 20%, and 30% noise. The latter
two paired differences have exact McNemar p-values `4.06e-5` and `3.76e-5`.
It costs 0.50--0.83 additional turns. At equal turn budgets, risk only is better
in the 0%/10% groups, while utility auditing is better by an average 0.62/1.43
percentage points at 20%/30% noise.

### Hybrid

Hybrid achieves the highest raw accuracy and recall in the 10--30% noise
groups. At 30% noise it exceeds utility auditing by 1.84 percentage points at
threshold 0.85, but consumes 0.97 more turns. Across equal-budget curve points,
utility auditing is better on average by 2.54, 1.85, 1.39, and 0.12 percentage
points at 0%, 10%, 20%, and 30% noise. Thus, immediate and retrospective
detectors are complementary for raw recall, but the aggressive combination is
not the most question-efficient strategy.

Compared with adaptive history at equal turn budgets, hybrid trails by 3.66,
2.40, and 0.63 percentage points at 0%, 10%, and 20% noise, then leads by 0.54
percentage points at 30% noise.

## Recommended operating points

- Minimum extra turns / high detector precision: `retro_risk_r50_b1`.
- Best moderate/high-noise accuracy--turn balance: `retro_utility_u050_b1`.
- Maximum recall when question cost is secondary: `hybrid_s30_u025_b2`.

For the main method, keep `retro_utility_u050_b1`. The next improvement should
gate retrospective auditing with dialogue-history reliability, turning it off
in clean/low-noise dialogues and allowing the stronger branch only when the
history indicates unreliable reporting.
