from __future__ import annotations
from abc import ABC, abstractmethod
from finplan.types import TaskInstance, ParsedPlan, SoftPreferenceResult

class BaseScorer(ABC):
    @abstractmethod
    def score(self, task: TaskInstance, plan: ParsedPlan) -> SoftPreferenceResult:
        raise NotImplementedError
