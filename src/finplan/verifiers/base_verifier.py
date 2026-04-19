from __future__ import annotations
from abc import ABC, abstractmethod
from finplan.types import TaskInstance, ParsedPlan, HardConstraintResult

class BaseVerifier(ABC):
    @abstractmethod
    def verify(self, task: TaskInstance, plan: ParsedPlan) -> HardConstraintResult:
        raise NotImplementedError
