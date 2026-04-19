from __future__ import annotations
from finplan.scorers.base_scorer import BaseScorer
from finplan.sim.financial_models import estimate_interest_cost_score, estimate_payment_flexibility_score, estimate_prepayment_optionality_score
from finplan.types import TaskInstance, ParsedPlan, SoftPreferenceResult
from finplan.utils.math_utils import clip01

class LoanScorer(BaseScorer):
    def score(self, task: TaskInstance, plan: ParsedPlan) -> SoftPreferenceResult:
        if not plan.parse_success:
            return SoftPreferenceResult({"interest_cost_score":0.0,"payment_flexibility":0.0,"prepayment_optionality":0.0})
        s = plan.structured
        return SoftPreferenceResult({
            "interest_cost_score": clip01(estimate_interest_cost_score(float(s.get("loan_amount",0.0)), float(s.get("annual_rate", task.context.get("base_rate",0.05))), int(s.get("term_months",360)))),
            "payment_flexibility": clip01(estimate_payment_flexibility_score(int(s.get("term_months",360)), bool(s.get("prepayment_option",False)))),
            "prepayment_optionality": clip01(estimate_prepayment_optionality_score(bool(s.get("prepayment_option",False)))),
        })
