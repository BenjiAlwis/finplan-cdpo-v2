from __future__ import annotations

from finplan.parsers.plan_parser import PlanParser
from finplan.rewards.reward_bundle import RewardBundleBuilder
from finplan.verifiers.portfolio_verifier import PortfolioVerifier
from finplan.verifiers.retirement_verifier import RetirementVerifier
from finplan.verifiers.loan_verifier import LoanVerifier
from finplan.scorers.portfolio_scorer import PortfolioScorer
from finplan.scorers.retirement_scorer import RetirementScorer
from finplan.scorers.loan_scorer import LoanScorer
from finplan.types import TaskInstance, ParsedPlan, RewardBundle


class FinPlanEnv:
    def __init__(self) -> None:
        self.parser = PlanParser()
        self._builders = {
            "portfolio": RewardBundleBuilder(PortfolioVerifier(), PortfolioScorer()),
            "retirement": RewardBundleBuilder(RetirementVerifier(), RetirementScorer()),
            "loan": RewardBundleBuilder(LoanVerifier(), LoanScorer()),
        }

    def evaluate(self, task: TaskInstance, raw_plan_text: str) -> tuple[ParsedPlan, RewardBundle]:
        parsed = self.parser.parse(raw_plan_text, task.domain, task.expected_plan_schema)
        reward = self._builders[task.domain].build(task, parsed)

        violated = [k for k, v in reward.hard.checks.items() if v == 0]
        reward.metadata.update({
            "task_id": task.task_id,
            "domain": task.domain,
            "difficulty": task.metadata.get("difficulty"),
            "hard_channel": dict(reward.hard.checks),
            "soft_channel": dict(reward.soft.scores),
            "hard_pass_count": sum(reward.hard.checks.values()),
            "hard_total_count": len(reward.hard.checks),
            "soft_mean_score": reward.soft.mean_score,
            "all_constraints_pass": reward.hard.all_pass,
            "violated_constraints": violated,
            "parse_success": parsed.parse_success,
            "parse_error": parsed.parse_error,
        })

        return parsed, reward
