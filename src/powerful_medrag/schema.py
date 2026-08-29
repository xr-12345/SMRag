"""Core, dependency-free schemas used by the probabilistic model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping
from urllib.parse import quote, unquote


UNKNOWN = "__unknown__"


class CertaintyCue(str, Enum):
    """Surface cue detected from an answer; it is not the latent report mode."""

    NONE = "none"
    CERTAIN = "certain"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, order=True)
class FeatureKey:
    """A clinical variable plus the context in which it holds.

    Context is part of identity.  Thus ``pain[activity=walking]`` and
    ``pain[activity=resting]`` are different variables, preventing a conditional
    difference from being mislabeled as a contradictory answer.
    """

    name: str
    context: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("feature name cannot be empty")
        normalized = tuple(sorted((str(k), str(v)) for k, v in self.context))
        if len({key for key, _ in normalized}) != len(normalized):
            raise ValueError("context keys must be unique")
        object.__setattr__(self, "context", normalized)

    @classmethod
    def from_parts(
        cls, name: str, context: Mapping[str, str] | None = None
    ) -> "FeatureKey":
        return cls(name=name, context=tuple((context or {}).items()))

    @property
    def token(self) -> str:
        if not self.context:
            return quote(self.name, safe="")
        encoded_context = "&".join(
            f"{quote(key, safe='')}={quote(value, safe='')}"
            for key, value in self.context
        )
        return f"{quote(self.name, safe='')}|{encoded_context}"

    @classmethod
    def from_token(cls, token: str) -> "FeatureKey":
        name_token, separator, context_token = token.partition("|")
        context: list[tuple[str, str]] = []
        if separator:
            for item in context_token.split("&"):
                key, equal, value = item.partition("=")
                if not equal:
                    raise ValueError(f"invalid feature token context: {token!r}")
                context.append((unquote(key), unquote(value)))
        return cls(name=unquote(name_token), context=tuple(context))

    def display_name(self) -> str:
        if not self.context:
            return self.name
        rendered = ", ".join(f"{key}={value}" for key, value in self.context)
        return f"{self.name} [{rendered}]"


@dataclass(frozen=True)
class VariableSpec:
    """Possible latent states and question metadata for one variable."""

    key: FeatureKey
    values: tuple[str, ...]
    question: str = ""
    cost: float = 1.0
    prerequisite: FeatureKey | None = None
    askable: bool = True

    def __post_init__(self) -> None:
        if len(self.values) < 2:
            raise ValueError(f"{self.key.display_name()} needs at least two states")
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"duplicate states for {self.key.display_name()}")
        if UNKNOWN in self.values:
            raise ValueError(f"{UNKNOWN!r} is a report, not a true clinical state")
        if self.cost <= 0:
            raise ValueError("question cost must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key.token,
            "values": list(self.values),
            "question": self.question,
            "cost": self.cost,
            "prerequisite": self.prerequisite.token if self.prerequisite else None,
            "askable": self.askable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "VariableSpec":
        return cls(
            key=FeatureKey.from_token(str(data["key"])),
            values=tuple(str(value) for value in data["values"]),  # type: ignore[arg-type]
            question=str(data.get("question", "")),
            cost=float(data.get("cost", 1.0)),
            prerequisite=(
                FeatureKey.from_token(str(data["prerequisite"]))
                if data.get("prerequisite")
                else None
            ),
            askable=bool(data.get("askable", True)),
        )


@dataclass(frozen=True)
class ClinicalCase:
    """One training or simulated case with explicitly observed true states.

    Missing keys remain missing.  They are never silently converted to
    ``absent`` unless a dataset adapter explicitly guarantees completeness.
    """

    diagnosis: str
    states: Mapping[FeatureKey, str]
    case_id: str = ""
    initial_observations: tuple["Observation", ...] = ()

    def __post_init__(self) -> None:
        if not self.diagnosis.strip():
            raise ValueError("diagnosis cannot be empty")


@dataclass(frozen=True)
class Observation:
    """A patient's reported answer.

    ``certainty`` is observable language evidence such as “肯定” or “好像”.
    Whether the answer is a misreport remains latent and is inferred later.
    """

    key: FeatureKey
    value: str
    certainty: CertaintyCue = CertaintyCue.NONE
    raw_text: str = ""


def make_binary_spec(
    name: str,
    *,
    question: str = "",
    context: Mapping[str, str] | None = None,
    cost: float = 1.0,
) -> VariableSpec:
    return VariableSpec(
        key=FeatureKey.from_parts(name, context),
        values=("absent", "present"),
        question=question,
        cost=cost,
    )


def validate_cases(cases: Iterable[ClinicalCase], specs: Iterable[VariableSpec]) -> None:
    spec_map = {spec.key: spec for spec in specs}
    for case in cases:
        for key, state in case.states.items():
            if key not in spec_map:
                raise ValueError(
                    f"case {case.case_id or '<unknown>'} contains unspecified feature {key.token}"
                )
            if state == UNKNOWN:
                continue
            if state not in spec_map[key].values:
                raise ValueError(
                    f"invalid state {state!r} for feature {key.display_name()}"
                )
