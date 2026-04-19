from __future__ import annotations

import json
from pathlib import Path

from finplan.env.finplan_env import FinPlanEnv
from finplan.sim.financial_models import (
    compute_dti,
    compute_ltv,
    compute_monthly_payment,
    estimate_turnover,
    run_portfolio_backtest,
    run_retirement_monte_carlo,
)
from finplan.types import TaskInstance
from finplan.utils.io import read_jsonl, write_jsonl


def make_placeholder_plan(task: TaskInstance) -> str:
    if task.domain == "portfolio":
        banned = set(task.constraints.get("banned_sectors", []))
        first_sector = "HEALTH" if "TECH" in banned else "TECH"
        return json.dumps(
            {
                "allocations": [
                    {"asset": "US_EQ", "sector": first_sector, "weight": 0.30},
                    {"asset": "INTL_EQ", "sector": "INDUSTRIALS", "weight": 0.20},
                    {"asset": "BONDS", "sector": "GOVT", "weight": 0.35},
                    {"asset": "CASH", "sector": "CASH", "weight": 0.15},
                ]
            }
        )

    if task.domain == "retirement":
        return json.dumps(
            {
                "monthly_income_target": max(
                    float(task.constraints.get("min_monthly_income", 2500.0)),
                    3200.0,
                ),
                "inflation_adjusted": True,
                "portfolio_return_assumption": float(
                    task.context.get("portfolio_return_assumption", 0.05)
                ),
            }
        )

    return json.dumps(
        {
            "loan_amount": 250000.0,
            "term_months": 360,
            "rate_type": "fixed",
            "annual_rate": float(task.context.get("base_rate", 0.055)),
            "property_value": 350000.0,
            "existing_monthly_debt": 300.0,
            "prepayment_option": True,
        }
    )


def _build_portfolio_evidence(task: TaskInstance, structured: dict) -> dict:
    allocations = structured.get("allocations", [])
    bt = run_portfolio_backtest(
        allocations,
        market_regime=str(task.context.get("market_regime", "neutral")),
    )

    actual_weight_sum = sum(float(x.get("weight", 0.0)) for x in allocations)
    actual_num_assets = sum(1 for x in allocations if float(x.get("weight", 0.0)) > 0.0)
    actual_sectors = [str(x.get("sector", "")) for x in allocations]

    current_allocations = task.context.get("current_allocations", [])
    actual_turnover = estimate_turnover(current_allocations, allocations)

    return {
        "portfolio_actual_weight_sum": actual_weight_sum,
        "portfolio_actual_num_assets": actual_num_assets,
        "portfolio_actual_sectors": actual_sectors,
        "portfolio_banned_sectors_reference": task.constraints.get("banned_sectors", []),
        "portfolio_required_min_assets": task.constraints.get("min_assets"),
        "portfolio_drawdown_threshold": task.constraints.get("max_drawdown"),
        "portfolio_actual_max_drawdown": bt.get("max_drawdown"),
        "portfolio_actual_annualized_return": bt.get("annualized_return", bt.get("annual_return")),
        "portfolio_actual_annualized_volatility": bt.get(
            "annualized_volatility", bt.get("annual_volatility")
        ),
        "portfolio_actual_esg_score": bt.get("esg_score"),
        "portfolio_actual_risk_adjusted_return": bt.get(
            "risk_adjusted_return", bt.get("sharpe_like")
        ),
        "portfolio_actual_turnover": actual_turnover,
        "portfolio_target_risk_reference": task.preferences.get("target_risk"),
        "portfolio_target_esg_reference": task.preferences.get("target_esg"),
        "portfolio_turnover_penalty_scale_reference": task.preferences.get(
            "turnover_penalty_scale"
        ),
        "portfolio_market_regime_reference": task.context.get("market_regime"),
    }


def _build_retirement_evidence(task: TaskInstance, structured: dict) -> dict:
    starting_balance = float(task.profile.get("portfolio_value", 0.0))
    monthly_income_target = float(structured.get("monthly_income_target", 0.0))
    current_age = int(task.profile.get("current_age", 65))
    target_age = int(task.profile.get("target_age", 90))
    inflation_assumption = float(task.context.get("inflation_assumption", 0.03))
    portfolio_return_assumption = float(
        structured.get(
            "portfolio_return_assumption",
            task.context.get("portfolio_return_assumption", 0.05),
        )
    )
    annual_volatility = float(task.context.get("retirement_annual_volatility", 0.12))
    n_scenarios = int(task.context.get("retirement_n_scenarios", 1000))
    seed = int(task.context.get("retirement_seed", 42))

    mc = run_retirement_monte_carlo(
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

    return {
        "retirement_actual_monthly_income_target": monthly_income_target,
        "retirement_required_min_monthly_income": task.constraints.get("min_monthly_income"),
        "retirement_actual_inflation_adjusted_flag": bool(
            structured.get("inflation_adjusted", False)
        ),
        "retirement_required_survival_threshold": task.constraints.get(
            "min_survival_probability",
            task.context.get("min_survival_probability", 0.80),
        ),
        "retirement_actual_survival_probability": mc.get("survival_probability"),
        "retirement_actual_depletion_probability": mc.get("depletion_probability"),
        "retirement_actual_ending_balance_median": mc.get(
            "median_ending_balance", mc.get("ending_balance")
        ),
        "retirement_actual_ending_balance_mean": mc.get("mean_ending_balance"),
        "retirement_actual_years_survived_median": mc.get("years_survived"),
        "retirement_actual_lifestyle_quality_base": float(monthly_income_target) / 6000.0,
        "retirement_actual_lifestyle_quality_score": mc.get("lifestyle_quality_score"),
        "retirement_actual_withdrawal_smoothness": mc.get("withdrawal_smoothness_score"),
        "retirement_actual_starting_balance": starting_balance,
        "retirement_bequest_preference_reference": task.preferences.get("bequest_preference"),
        "retirement_inflation_assumption_reference": inflation_assumption,
        "retirement_return_assumption_reference": portfolio_return_assumption,
        "retirement_annual_volatility_reference": annual_volatility,
        "retirement_n_scenarios_reference": n_scenarios,
    }


def _build_loan_evidence(task: TaskInstance, structured: dict) -> dict:
    loan_amount = float(structured.get("loan_amount", 0.0))
    property_value = float(structured.get("property_value", 0.0))
    annual_rate = float(structured.get("annual_rate", 0.05))
    term_months = int(structured.get("term_months", 360))
    existing_monthly_debt = float(structured.get("existing_monthly_debt", 0.0))
    annual_income = float(task.profile.get("annual_income", 0.0))
    rate_type = str(structured.get("rate_type", ""))

    monthly_payment = compute_monthly_payment(
        principal=loan_amount,
        annual_rate=annual_rate,
        term_months=term_months,
    )
    actual_dti = compute_dti(
        monthly_payment=monthly_payment,
        annual_income=annual_income,
        existing_monthly_debt=existing_monthly_debt,
    )
    actual_ltv = compute_ltv(
        loan_amount=loan_amount,
        property_value=property_value,
    )

    total_paid = monthly_payment * term_months
    total_interest = max(0.0, total_paid - loan_amount)
    total_interest_ratio = total_interest / loan_amount if loan_amount > 1e-8 else 0.0

    return {
        "loan_actual_monthly_payment": monthly_payment,
        "loan_actual_dti": actual_dti,
        "loan_dti_threshold": task.constraints.get("max_dti"),
        "loan_actual_ltv": actual_ltv,
        "loan_ltv_threshold": task.constraints.get("max_ltv"),
        "loan_actual_rate_type": rate_type,
        "loan_actual_total_interest_ratio": total_interest_ratio,
        "loan_actual_term_months": term_months,
        "loan_actual_prepayment_option": bool(structured.get("prepayment_option", False)),
        "loan_actual_annual_rate": annual_rate,
        "loan_actual_loan_amount": loan_amount,
        "loan_actual_property_value": property_value,
        "loan_actual_existing_monthly_debt": existing_monthly_debt,
    }


def build_audit_evidence(task: TaskInstance, structured: dict, parse_success: bool) -> dict:
    if not parse_success:
        return {}

    if task.domain == "portfolio":
        return _build_portfolio_evidence(task, structured)
    if task.domain == "retirement":
        return _build_retirement_evidence(task, structured)
    return _build_loan_evidence(task, structured)


def verify_file(input_path: str, output_path: str) -> None:
    env = FinPlanEnv()
    rows = read_jsonl(input_path)
    out_rows = []

    for row in rows:
        task = TaskInstance(**row)
        raw_plan = make_placeholder_plan(task)
        parsed, reward = env.evaluate(task, raw_plan)
        audit_evidence = build_audit_evidence(task, parsed.structured, parsed.parse_success)

        out_rows.append(
            {
                "task": task.to_dict(),
                "raw_plan": raw_plan,
                "parsed": parsed.to_dict(),
                "reward": reward.to_dict(),
                "task_id": task.task_id,
                "domain": task.domain,
                "parse_success": parsed.parse_success,
                "parse_error": parsed.parse_error,
                "hard_checks": reward.hard.checks,
                "soft_scores": reward.soft.scores,
                "all_constraints_pass": reward.hard.all_pass,
                "soft_mean_score": reward.soft.mean_score,
                "combined_quality": reward.combined_quality,
                "violated_constraints": reward.metadata.get("violated_constraints", []),
                "hard_channel": reward.metadata.get("hard_channel", {}),
                "soft_channel": reward.metadata.get("soft_channel", {}),
                "audit_evidence": audit_evidence,
            }
        )

    write_jsonl(output_path, out_rows)
    print(f"Wrote {len(out_rows)} rows to {output_path}")


def main() -> None:
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    verify_file(
        "data/raw/portfolio_instances.jsonl",
        "data/processed/portfolio_verified.jsonl",
    )
    verify_file(
        "data/raw/retirement_instances.jsonl",
        "data/processed/retirement_verified.jsonl",
    )
    verify_file(
        "data/raw/loan_instances.jsonl",
        "data/processed/loan_verified.jsonl",
    )


if __name__ == "__main__":
    main()
