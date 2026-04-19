from __future__ import annotations
from finplan.scorers.base_scorer import BaseScorer
from finplan.sim.financial_models import run_portfolio_backtest, estimate_turnover
from finplan.types import TaskInstance, ParsedPlan, SoftPreferenceResult
from finplan.utils.math_utils import clip01

class PortfolioScorer(BaseScorer):
    def score(self, task: TaskInstance, plan: ParsedPlan) -> SoftPreferenceResult:
        if not plan.parse_success:
            return SoftPreferenceResult({"risk_alignment":0.0,"esg_alignment":0.0,"tax_efficiency":0.0,"risk_adjusted_return":0.0})
        allocations = plan.structured.get("allocations", [])
        bt = run_portfolio_backtest(allocations, market_regime=str(task.context.get("market_regime","neutral")))
        target_risk = float(task.preferences.get("target_risk",0.15))
        target_esg = float(task.preferences.get("target_esg",0.6))
        turnover = estimate_turnover(task.context.get("current_allocations", []), allocations)
        return SoftPreferenceResult({
            "risk_alignment": clip01(1.0 - abs(bt["annualized_volatility"] - target_risk) / max(target_risk,1e-6)),
            "esg_alignment": clip01(1.0 - abs(bt["esg_score"] - target_esg)),
            "tax_efficiency": clip01(1.0 - float(task.preferences.get("turnover_penalty_scale",1.0)) * turnover),
            "risk_adjusted_return": clip01(bt["risk_adjusted_return"]),
        })
