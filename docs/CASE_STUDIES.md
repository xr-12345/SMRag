# Synthetic case diagnostics

These are reproducible toy examples from seed 2026. They illustrate mechanisms and failures; they
are not clinical cases and do not establish causal effects.

## Apparent success: `common_cold-75`, noise 0.0

The full two-layer baseline ended with `allergic_rhinitis`, whereas the joint policy ended with the
correct `common_cold` prediction. Both observed fever absent, itchy eyes absent, unknown myalgia,
sore throat absent, dry cough absent, and runny nose present. The joint policy performed one
verification after an unknown history cue. The repeated answer agreed, so it did not introduce a new
clinical value. Consequently this is a paired outcome difference involving inference/stopping, not
evidence that verification itself caused the correction.

## Failure despite catching an error: `influenza-46`, noise 0.1

The patient simulator misreported runny nose as absent. The joint policy later verified the item,
received present, resolved the conflict to `UNKNOWN`, and correctly removed the wrong report. Despite
that mitigation, it ended with `common_cold`; the full two-layer baseline ended with the correct
`influenza`. This is an important failure mode: removing a harmful report need not improve Top-1 when
the remaining evidence/model probabilities still favor another disease.

## Confirmation-bias warning: `common_cold-44`, noise 0.2

Before turn four, the no-misreport trajectory's Top-1 diagnosis was wrong with probability 0.937.
The patient then gave a correct, certain `myalgia=present` answer. Because that answer had predictive
probability 0.066 under the current history, the surprisal gate raised its misreport prior above the
0.5% baseline even though the answer was correct. The cap kept the increase small in this toy case,
but the event demonstrates why disagreement with the current diagnosis cannot be treated as direct
evidence that the patient is wrong. The history-only gate avoids this current-answer feedback.

## Aggregate pilot lesson

The frozen joint policy improved toy Top-1 and Brier relative to `full_two_layer`, and reduced new
questions slightly, but verification made total atomic questions higher. It therefore fails the
pre-registered one-question efficiency target. The next validation experiment should test a more
selective expected value of verification, rather than increasing its weight or hiding its cost.

