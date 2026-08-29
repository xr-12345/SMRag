"""Structured patient simulator and end-to-end dialogue runner."""

from __future__ import annotations

import random
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Mapping

from .belief import BeliefTracker, UpdateResult, entropy
from .channel import AnswerChannel, ReportMode
from .estimation import DiseaseStateModel
from .questioning import QuestionScore, QuestionSelector
from .schema import UNKNOWN, CertaintyCue, FeatureKey, Observation, VariableSpec


ObservationTransform = Callable[[Observation, VariableSpec], Observation]


@dataclass(frozen=True)
class PatientProfile:
    mode_prior: Mapping[ReportMode, float] = field(
        default_factory=lambda: {
            ReportMode.CERTAIN: 0.82,
            ReportMode.UNCERTAIN: 0.10,
            ReportMode.UNKNOWN: 0.05,
            ReportMode.MISREPORTED: 0.03,
        }
    )

    def __post_init__(self) -> None:
        if set(self.mode_prior) != set(ReportMode):
            raise ValueError("patient profile must cover every report mode")
        if min(self.mode_prior.values()) < 0:
            raise ValueError("report mode probabilities cannot be negative")
        if abs(sum(self.mode_prior.values()) - 1.0) > 1e-9:
            raise ValueError("report mode probabilities must sum to one")

    @classmethod
    def from_noise_rate(
        cls,
        noise_rate: float,
        *,
        uncertain_share: float = 0.40,
        unknown_share: float = 0.30,
        misreported_share: float = 0.30,
    ) -> "PatientProfile":
        """Split total unreliable-answer mass into three explicit mechanisms."""

        if not 0 <= noise_rate <= 1:
            raise ValueError("noise_rate must be in [0, 1]")
        shares = (uncertain_share, unknown_share, misreported_share)
        if min(shares) < 0 or abs(sum(shares) - 1.0) > 1e-9:
            raise ValueError("noise shares must be nonnegative and sum to one")
        return cls(
            mode_prior={
                ReportMode.CERTAIN: 1.0 - noise_rate,
                ReportMode.UNCERTAIN: noise_rate * uncertain_share,
                ReportMode.UNKNOWN: noise_rate * unknown_share,
                ReportMode.MISREPORTED: noise_rate * misreported_share,
            }
        )


class StructuredPatientSimulator:
    """Samples reports from fixed latent truth without involving an LLM."""

    def __init__(
        self,
        *,
        diagnosis: str,
        latent_states: Mapping[FeatureKey, str],
        model: DiseaseStateModel,
        channel: AnswerChannel | None = None,
        profile: PatientProfile | None = None,
        seed: int = 0,
    ) -> None:
        if diagnosis not in model.diseases:
            raise ValueError(f"unknown simulated diagnosis: {diagnosis}")
        self.diagnosis = diagnosis
        self.latent_states = dict(latent_states)
        self.model = model
        self.channel = channel or AnswerChannel()
        self.profile = profile or PatientProfile()
        self.seed = seed
        self._answer_counts: dict[FeatureKey, int] = {}
        self.rng = random.Random(seed)

    @classmethod
    def sample_case(
        cls,
        model: DiseaseStateModel,
        diagnosis: str,
        *,
        channel: AnswerChannel | None = None,
        profile: PatientProfile | None = None,
        seed: int = 0,
    ) -> "StructuredPatientSimulator":
        rng = random.Random(seed)
        latent_states: dict[FeatureKey, str] = {}
        for key in model.specs:
            distribution = model.state_distribution(diagnosis, key)
            target = rng.random()
            cumulative = 0.0
            selected = next(iter(distribution))
            for value, probability in distribution.items():
                selected = value
                cumulative += probability
                if target <= cumulative:
                    break
            latent_states[key] = selected
        simulator = cls(
            diagnosis=diagnosis,
            latent_states=latent_states,
            model=model,
            channel=channel,
            profile=profile,
            seed=seed,
        )
        simulator.rng = rng
        return simulator

    def answer(self, key: FeatureKey) -> tuple[Observation, ReportMode]:
        spec = self.model.specs[key]
        if key not in self.latent_states:
            return (
                Observation(
                    key=key,
                    value=UNKNOWN,
                    oracle_report_mode=ReportMode.UNKNOWN.value,
                ),
                ReportMode.UNKNOWN,
            )
        occurrence = self._answer_counts.get(key, 0)
        self._answer_counts[key] = occurrence + 1
        digest = hashlib.blake2b(
            f"{self.seed}|{key.token}|{occurrence}".encode("utf-8"),
            digest_size=8,
        ).digest()
        answer_rng = random.Random(int.from_bytes(digest, "big"))
        observed_value, mode = self.channel.sample(
            self.latent_states[key],
            spec.values,
            rng=answer_rng,
            mode_prior=self.profile.mode_prior,
        )
        if observed_value == UNKNOWN:
            certainty = CertaintyCue.NONE
        elif mode == ReportMode.UNCERTAIN:
            certainty = CertaintyCue.UNCERTAIN
        else:
            # A misreport may be said confidently; mode remains unobserved.
            certainty = CertaintyCue.CERTAIN
        return (
            Observation(
                key=key,
                value=observed_value,
                certainty=certainty,
                oracle_report_mode=mode.value,
            ),
            mode,
        )


@dataclass(frozen=True)
class StopRule:
    max_questions: int = 10
    posterior_threshold: float = 0.85
    entropy_threshold: float = 0.30
    minimum_question_utility: float = 1e-4

    def __post_init__(self) -> None:
        if self.max_questions < 0:
            raise ValueError("max_questions cannot be negative")
        if not 0 < self.posterior_threshold <= 1:
            raise ValueError("posterior_threshold must be in (0, 1]")
        if self.entropy_threshold < 0 or self.minimum_question_utility < 0:
            raise ValueError("stopping thresholds cannot be negative")


@dataclass(frozen=True)
class DialogueTurn:
    index: int
    question: QuestionScore
    observation: Observation
    true_report_mode: ReportMode
    update: UpdateResult


@dataclass(frozen=True)
class DialogueResult:
    true_diagnosis: str
    predicted_diagnosis: str
    belief: Mapping[str, float]
    turns: tuple[DialogueTurn, ...]
    stop_reason: str

    @property
    def correct(self) -> bool:
        return self.true_diagnosis == self.predicted_diagnosis


def run_dialogue(
    patient: StructuredPatientSimulator,
    *,
    tracker: BeliefTracker | None = None,
    selector: QuestionSelector | None = None,
    stop_rule: StopRule | None = None,
    initial_observations: tuple[Observation, ...] = (),
    observation_transform: ObservationTransform | None = None,
) -> DialogueResult:
    tracker = tracker or BeliefTracker(patient.model, patient.channel)
    selector = selector or QuestionSelector()
    rule = stop_rule or StopRule()
    asked: set[FeatureKey] = {
        update.observation.key for update in tracker.history
    }
    turns: list[DialogueTurn] = []

    for observation in initial_observations:
        if observation_transform is not None:
            observation = observation_transform(
                observation, patient.model.specs[observation.key]
            )
        tracker.update(observation)
        asked.add(observation.key)

    stop_reason = "question_budget"
    while len(turns) < rule.max_questions:
        top_probability = max(tracker.belief.values())
        if top_probability >= rule.posterior_threshold:
            stop_reason = "posterior_threshold"
            break
        if entropy(tracker.belief) <= rule.entropy_threshold:
            stop_reason = "entropy_threshold"
            break
        ranking = selector.rank(tracker, excluded=asked)
        if not ranking:
            stop_reason = "no_questions_left"
            break
        question = ranking[0]
        if question.utility < rule.minimum_question_utility:
            stop_reason = "low_question_utility"
            break
        observation, true_mode = patient.answer(question.key)
        if observation_transform is not None:
            observation = observation_transform(
                observation, patient.model.specs[observation.key]
            )
        update = tracker.update(observation)
        turns.append(
            DialogueTurn(
                index=len(turns) + 1,
                question=question,
                observation=observation,
                true_report_mode=true_mode,
                update=update,
            )
        )
        asked.add(question.key)

    predicted_diagnosis = max(tracker.belief, key=tracker.belief.__getitem__)
    return DialogueResult(
        true_diagnosis=patient.diagnosis,
        predicted_diagnosis=predicted_diagnosis,
        belief=dict(tracker.belief),
        turns=tuple(turns),
        stop_reason=stop_reason,
    )
