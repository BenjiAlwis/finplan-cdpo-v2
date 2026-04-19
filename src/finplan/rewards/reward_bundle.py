from __future__ import annotations
from finplan.scorers.base_scorer import BaseScorer
from finplan.verifiers.base_verifier import BaseVerifier
from finplan.types import TaskInstance, ParsedPlan, RewardBundle

class RewardBundleBuilder:
    def __init__(self, verifier: BaseVerifier, scorer: BaseScorer) -> None:
        self.verifier = verifier
        self.scorer = scorer
    def build(self, task: TaskInstance, plan: ParsedPlan) -> RewardBundle:
        hard = self.verifier.verify(task, plan)
        soft = self.scorer.score(task, plan)
        violated = [k for k, v in hard.checks.items() if v == 0]
        return RewardBundle(task.task_id, task.domain, hard, soft, {
            "hard_pass_count": sum(hard.checks.values()),
            "hard_total_count": len(hard.checks),
            "soft_score_count": len(soft.scores),
            "violated_constraints": violated,
        })
