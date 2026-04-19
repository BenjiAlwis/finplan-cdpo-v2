from __future__ import annotations
import random, uuid
from typing import Iterable
from finplan.generators.base_generator import BaseGenerator
from finplan.types import TaskInstance

class LoanGenerator(BaseGenerator):
    def generate(self, n: int, difficulty: str) -> Iterable[TaskInstance]:
        for _ in range(n):
            if difficulty == "easy":
                income, max_dti, max_ltv, rate, band = random.randint(120000,250000), random.choice([0.43,0.45,0.48]), random.choice([0.80,0.85,0.90]), random.choice([0.045,0.05,0.055]), random.choice(["good","excellent"])
            elif difficulty == "medium":
                income, max_dti, max_ltv, rate, band = random.randint(70000,180000), random.choice([0.38,0.40,0.43]), random.choice([0.75,0.80,0.85]), random.choice([0.05,0.055,0.06]), random.choice(["fair","good","excellent"])
            else:
                income, max_dti, max_ltv, rate, band = random.randint(40000,120000), random.choice([0.33,0.36,0.38]), random.choice([0.70,0.75,0.80]), random.choice([0.06,0.065,0.07]), random.choice(["fair","good"])
            yield TaskInstance(
                task_id=str(uuid.uuid4()),
                domain="loan",
                profile={"annual_income": float(income), "credit_band": band},
                constraints={"max_dti": float(max_dti), "max_ltv": float(max_ltv)},
                preferences={"payment_flexibility_preference": float(random.choice([0.3,0.6,0.9])), "prepayment_preference": float(random.choice([0.2,0.5,0.8]))},
                context={"base_rate": float(rate)},
                expected_plan_schema={"type":"json","required_fields":["loan_amount","term_months","rate_type"]},
                metadata={"difficulty": difficulty, "task_family": "loan_structuring"},
            )
