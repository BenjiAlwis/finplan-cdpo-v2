from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Literal

DomainName = Literal["portfolio", "retirement", "loan"]

@dataclass
class TaskInstance:
    task_id: str
    domain: DomainName
    profile: dict[str, Any]
    constraints: dict[str, Any]
    preferences: dict[str, Any]
    context: dict[str, Any]
    expected_plan_schema: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class ParsedPlan:
    domain: DomainName
    raw_text: str
    structured: dict[str, Any]
    parse_success: bool
    parse_error: str | None = None
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class HardConstraintResult:
    checks: dict[str, int]
    @property
    def all_pass(self) -> int:
        return int(all(v == 1 for v in self.checks.values()))
    def to_dict(self) -> dict[str, Any]:
        return {"checks": self.checks, "all_pass": self.all_pass}

@dataclass
class SoftPreferenceResult:
    scores: dict[str, float]
    @property
    def mean_score(self) -> float:
        return sum(self.scores.values()) / len(self.scores) if self.scores else 0.0
    def to_dict(self) -> dict[str, Any]:
        return {"scores": self.scores, "mean_score": self.mean_score}

@dataclass
class RewardBundle:
    task_id: str
    domain: DomainName
    hard: HardConstraintResult
    soft: SoftPreferenceResult
    metadata: dict[str, Any] = field(default_factory=dict)
    @property
    def combined_quality(self) -> float:
        return float(self.hard.all_pass) * self.soft.mean_score
    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "hard": self.hard.to_dict(),
            "soft": self.soft.to_dict(),
            "combined_quality": self.combined_quality,
            "metadata": self.metadata,
        }
