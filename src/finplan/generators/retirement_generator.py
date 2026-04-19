from __future__ import annotations

import random
import uuid
from typing import Iterable

from finplan.generators.base_generator import BaseGenerator
from finplan.types import TaskInstance


class RetirementGenerator(BaseGenerator):
    """Generates retirement-planning task instances with a more balanced
    feasibility distribution for the Monte Carlo retirement environment.
    """

    def generate(self, n: int, difficulty: str) -> Iterable[TaskInstance]:
        for _ in range(n):
            current_age = random.randint(55, 70)
            target_age = random.randint(max(current_age + 15, 80), 100)

            if difficulty == "easy":
                portfolio_value = random.randint(1300000, 2600000)
                min_income = random.randint(2200, 3200)
                inflation_assumption = random.choice([0.02, 0.025, 0.03])
                return_assumption = random.choice([0.05, 0.055, 0.06])

            elif difficulty == "medium":
                portfolio_value = random.randint(800000, 1900000)
                min_income = random.randint(2500, 3800)
                inflation_assumption = random.choice([0.02, 0.025, 0.03])
                return_assumption = random.choice([0.045, 0.05, 0.055])

            else:  # hard
                portfolio_value = random.randint(500000, 1400000)
                min_income = random.randint(2500, 4200)
                inflation_assumption = random.choice([0.025, 0.03, 0.035])
                return_assumption = random.choice([0.04, 0.045, 0.05])

            bequest_preference = random.choice([0.2, 0.5, 0.8])
            lifestyle_priority = random.choice([0.3, 0.6, 0.9])

            yield TaskInstance(
                task_id=str(uuid.uuid4()),
                domain="retirement",
                profile={
                    "portfolio_value": float(portfolio_value),
                    "current_age": current_age,
                    "target_age": target_age,
                },
                constraints={
                    "min_monthly_income": float(min_income),
                },
                preferences={
                    "bequest_preference": float(bequest_preference),
                    "lifestyle_priority": float(lifestyle_priority),
                },
                context={
                    "inflation_assumption": float(inflation_assumption),
                    "portfolio_return_assumption": float(return_assumption),
                },
                expected_plan_schema={
                    "type": "json",
                    "required_fields": [
                        "monthly_income_target",
                        "inflation_adjusted",
                    ],
                },
                metadata={
                    "difficulty": difficulty,
                    "task_family": "retirement_planning",
                },
            )
