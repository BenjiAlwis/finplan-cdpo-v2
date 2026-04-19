from __future__ import annotations
from finplan.sim.financial_models import run_portfolio_backtest
from finplan.types import TaskInstance, ParsedPlan, HardConstraintResult
from finplan.verifiers.base_verifier import BaseVerifier

class PortfolioVerifier(BaseVerifier):
    def verify(self, task: TaskInstance, plan: ParsedPlan) -> HardConstraintResult:
        if not plan.parse_success:
            return HardConstraintResult({"parse_valid":0,"weights_sum_valid":0,"banned_sector_valid":0,"diversification_valid":0,"drawdown_valid":0})
        allocations = plan.structured.get("allocations", [])
        bt = run_portfolio_backtest(allocations, market_regime=str(task.context.get("market_regime","neutral")))
        checks = {
            "parse_valid": 1,
            "weights_sum_valid": int(abs(sum(float(a.get("weight",0.0)) for a in allocations)-1.0) <= 1e-6),
            "banned_sector_valid": int(len(set(task.constraints.get("banned_sectors",[])).intersection({str(a.get("sector","")) for a in allocations})) == 0),
            "diversification_valid": int(len([a for a in allocations if float(a.get("weight",0.0)) > 0.0]) >= int(task.constraints.get("min_assets",1))),
            "drawdown_valid": int(bt["max_drawdown"] <= float(task.constraints.get("max_drawdown",0.2))),
        }
        return HardConstraintResult(checks)
