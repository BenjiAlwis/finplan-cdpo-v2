from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable
from finplan.types import TaskInstance

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, n: int, difficulty: str) -> Iterable[TaskInstance]:
        raise NotImplementedError
