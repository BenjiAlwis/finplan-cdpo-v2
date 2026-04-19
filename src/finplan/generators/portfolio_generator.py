from __future__ import annotations
import random, uuid
from typing import Iterable
from finplan.generators.base_generator import BaseGenerator
from finplan.types import TaskInstance

class PortfolioGenerator(BaseGenerator):
    def generate(self, n: int, difficulty: str) -> Iterable[TaskInstance]:
        for _ in range(n):
            if difficulty == "easy":
                max_dd, min_assets = 0.30, 3
            elif difficulty == "medium":
                max_dd, min_assets = 0.22, 4
            else:
                max_dd, min_assets = 0.16, 5
            yield TaskInstance(
                task_id=str(uuid.uuid4()),
                domain="portfolio",
                profile={"age": random.randint(25,70), "income": random.randint(60000,250000)},
                constraints={"max_drawdown": max_dd, "min_assets": min_assets, "banned_sectors": random.choice([[],["TOBACCO"],["FOSSIL"],["TOBACCO","FOSSIL"]])},
                preferences={"target_risk": random.choice([0.10,0.12,0.14,0.16,0.18]), "target_esg": random.choice([0.4,0.6,0.8]), "turnover_penalty_scale": random.choice([0.6,0.8,1.0])},
                context={"market_regime": random.choice(["bull","neutral","bear"]), "current_allocations": [{"asset":"US_EQ","sector":"TECH","weight":0.50},{"asset":"BONDS","sector":"GOVT","weight":0.30},{"asset":"CASH","sector":"CASH","weight":0.20}]},
                expected_plan_schema={"type":"json","required_fields":["allocations"]},
                metadata={"difficulty": difficulty},
            )
