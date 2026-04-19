from __future__ import annotations

from finplan.scorers.base_scorer import BaseScorer
from finplan.sim.financial_models import (
    run_retirement_monte_carlo,
    estimate_bequest_alignment,
)
from finplan.types import TaskInstance, ParsedPlan, SoftPreferenceResult
from finplan.utils.math_utils import clip01


class RetirementScorer(BaseScorer):
    def score(self, task: TaskInstance, plan: ParsedPlan) -> SoftPreferenceResult:
        if not plan.parse_success:
            return SoftPreferenceResult(
                {
                    "lifestyle_quality": 0.0,
                    "bequest_alignment": 0.0,
                    "withdrawal_smoothness": 0.0,
                }
            )

        s = plan.structured
        starting_balance = float(task.profile.get("portfolio_value", 0.0))

        mc = run_retirement_monte_carlo(
            starting_balance=starting_balance,
            monthly_income_target=float(s.get("monthly_income_target", 0.0)),
            current_age=int(task.profile.get("current_age", 65)),
            target_age=int(task.profile.get("target_age", 90)),
            inflation_assumption=float(task.context.get("inflation_assumption", 0.03)),
            portfolio_return_assumption=float(
                s.get(
                    "portfolio_return_assumption",
                    task.context.get("portfolio_return_assumption", 0.05),
                )
            ),
            annual_volatility=float(
                task.context.get("retirement_annual_volatility", 0.12)
            ),
            n_scenarios=int(task.context.get("retirement_n_scenarios", 1000)),
            seed=int(task.context.get("retirement_seed", 42)),
        )

        return SoftPreferenceResult(
            {
                "lifestyle_quality": clip01(
                    float(mc.get("lifestyle_quality_score", 0.0))
                ),
                "bequest_alignment": clip01(
                    estimate_bequest_alignment(
                        ending_balance=float(mc.get("ending_balance", 0.0)),
                        bequest_preference=float(
                            task.preferences.get("bequest_preference", 0.5)
                        ),
                        starting_balance=starting_balance,
                    )
                ),
                "withdrawal_smoothness": clip01(
                    float(mc.get("withdrawal_smoothness_score", 0.0))
                ),
            }
        )
