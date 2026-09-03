"""Per-feature evidence memory for the joint report model.

A feature that is asked once, then re-asked under verification, contributes ONE
joint likelihood factor -- not two independent single-answer factors.  This
module keeps the original answer and every verification answer together so the
joint tracker can treat them as a single evidence unit, and so conflicting
answers are never silently compressed to ``UNKNOWN``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping

from .joint_report_channel import VerificationType
from .schema import FeatureKey, Observation


@dataclass(frozen=True)
class ReportBundle:
    """All answers for one clinical feature, in arrival order.

    ``original`` is the first answer; ``verifications`` are the re-asks.  Nothing
    is overwritten and no conflict is resolved to ``UNKNOWN`` here -- the joint
    likelihood resolves conflicts probabilistically.
    """

    key: FeatureKey
    original: Observation
    verifications: tuple[Observation, ...] = ()
    verification_types: tuple[VerificationType, ...] = ()

    def __post_init__(self) -> None:
        if len(self.verifications) != len(self.verification_types):
            raise ValueError(
                "verifications and verification_types must be one-to-one"
            )
        for verification in self.verifications:
            if verification.key != self.key:
                raise ValueError(
                    f"verification key {verification.key.token} does not match "
                    f"bundle key {self.key.token}"
                )

    @property
    def is_verified(self) -> bool:
        return len(self.verifications) > 0

    @property
    def all_answers(self) -> tuple[Observation, ...]:
        return (self.original, *self.verifications)


class ReliabilityMemory:
    """A mapping from feature key to its (possibly verified) report bundle."""

    def __init__(self) -> None:
        self._bundles: dict[FeatureKey, ReportBundle] = {}

    def observe_single(self, observation: Observation) -> ReportBundle:
        if observation.key in self._bundles:
            raise ValueError(
                f"feature {observation.key.token} already has a first answer"
            )
        bundle = ReportBundle(key=observation.key, original=observation)
        self._bundles[observation.key] = bundle
        return bundle

    def observe_verification(
        self,
        key: FeatureKey,
        observation: Observation,
        verification_type: VerificationType = VerificationType.REPEAT,
    ) -> ReportBundle:
        bundle = self._bundles.get(key)
        if bundle is None:
            raise KeyError(
                f"cannot verify unseen feature {key.token} (observe_single first)"
            )
        if observation.key != key:
            raise ValueError(
                f"verification key {observation.key.token} does not match {key.token}"
            )
        updated = ReportBundle(
            key=key,
            original=bundle.original,
            verifications=bundle.verifications + (observation,),
            verification_types=bundle.verification_types + (verification_type,),
        )
        self._bundles[key] = updated
        return updated

    def get(self, key: FeatureKey) -> ReportBundle | None:
        return self._bundles.get(key)

    def __contains__(self, key: FeatureKey) -> bool:
        return key in self._bundles

    def __getitem__(self, key: FeatureKey) -> ReportBundle:
        return self._bundles[key]

    def __iter__(self) -> Iterator[FeatureKey]:
        return iter(self._bundles)

    def items(self) -> Iterable[tuple[FeatureKey, ReportBundle]]:
        return self._bundles.items()

    def __len__(self) -> int:
        return len(self._bundles)

    def keys(self) -> Iterable[FeatureKey]:
        return self._bundles.keys()

    def values(self) -> Iterable[ReportBundle]:
        return self._bundles.values()

    def as_dict(self) -> Mapping[FeatureKey, ReportBundle]:
        return dict(self._bundles)
