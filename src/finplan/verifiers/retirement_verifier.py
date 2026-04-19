from __future__ import annotations

from finplan.sim.financial_models import run_retirement_monte_carlo
from finplan.types import HardConstraintResult, ParsedPlan, TaskInstance
from finplan.verifiers.base_verifier import BaseVerifier


class RetirementVerifier(BaseVerifier):
    """Binary hard checks for retirement planning.

    Updated version:
    - uses Monte Carlo simulation
    - checks survival probability against a threshold
    - still checks income floor and inflation handling deterministically
    """

    def verify(self, task: TaskInstance, plan: ParsedPlan) -> HardConstraintResult:
        if not plan.parse_success:
            return HardConstraintResult(
                checks={
                    "parse_valid": 0,
                    "no_early_depletion": 0,
                    "income_floor_valid": 0,
                    "inflation_handling_valid": 0,
                }
            )

        structured = plan.structured
        simulation = self._run_simulation(task, structured)

        checks = {
            "parse_valid": 1,
            "no_early_depletion": self._check_survival_threshold(task, simulation),
            "income_floor_valid": self._check_income_floor(task, structured),
            "inflation_handling_valid": self._check_inflation_handling(structured),
        }
        return HardConstraintResult(checks=checks)

    def _run_simulation(self, task: TaskInstance, structured: dict) -> dict:
        starting_balance = float(task.profile.get("portfolio_value", 0.0))
        current_age = int(task.profile.get("current_age", 65))
        target_age = int(task.profile.get("target_age", 90))
        inflation_assumption = float(task.context.get("inflation_assumption", 0.03))

        portfolio_return_assumption = float(
            structured.get(
                "portfolio_return_assumption",
                task.context.get("portfolio_return_assumption", 0.05),
            )
        )
        annual_volatility = float(
            task.context.get("retirement_annual_volatility", 0.12)
        )
        monthly_income_target = float(structured.get("monthly_income_target", 0.0))
        n_scenarios = int(task.context.get("retirement_n_scenarios", 1000))
        seed = int(task.context.get("retirement_seed", 42))

        return run_retirement_monte_carlo(
            starting_balance=starting_balance,
            monthly_income_target=monthly_income_target,
            current_age=current_age,
            target_age=target_age,
            inflation_assumption=inflation_assumption,
            portfolio_return_assumption=portfolio_return_assumption,
            annual_volatility=annual_volatility,
            n_scenarios=n_scenarios,
            seed=seed,
        )

    def _check_survival_threshold(self, task: TaskInstance, simulation: dict) -> int:
        threshold = float(
            task.constraints.get(
                "min_survival_probability",
                task.context.get("min_survival_probability", 0.80),
            )
        )
        survival_probability = float(simulation.get("survival_probability", 0.0))
        return int(survival_probability >= threshold)

    def _check_income_floor(self, task: TaskInstance, structured: dict) -> int:
        min_income = float(task.constraints.get("min_monthly_income", 0.0))
        proposed_income = float(structured.get("monthly_income_target", 0.0))
        return int(proposed_income >= min_income)

    def _check_inflation_handling(self, structured: dict) -> int:
        return int(bool(structured.get("inflation_adjusted", False)))
