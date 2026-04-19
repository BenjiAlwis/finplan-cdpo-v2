from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from finplan.utils.io import read_jsonl
from finplan.utils.seeding import seed_everything


PROCESSED_FILES = [
    ("portfolio", "data/processed/portfolio_verified.jsonl"),
    ("retirement", "data/processed/retirement_verified.jsonl"),
    ("loan", "data/processed/loan_verified.jsonl"),
]


def _safe_json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _difficulty_rank(value: str | None) -> int:
    order = {"easy": 0, "medium": 1, "hard": 2}
    return order.get(str(value).lower(), 99)


def _row_signature(row: dict) -> tuple:
    task = row.get("task", {})
    reward = row.get("reward", {})
    metadata = reward.get("metadata", {})

    domain = row.get("domain", task.get("domain"))
    difficulty = task.get("metadata", {}).get("difficulty", metadata.get("difficulty"))
    all_constraints_pass = row.get(
        "all_constraints_pass",
        reward.get("hard", {}).get("all_pass"),
    )
    violated_constraints = tuple(
        row.get("violated_constraints", metadata.get("violated_constraints", []))
    )
    raw_plan = row.get("raw_plan", "")

    return (
        domain,
        difficulty,
        int(bool(all_constraints_pass)),
        violated_constraints,
        raw_plan,
    )


def _diverse_sample(rows: list[dict], n: int) -> list[dict]:
    shuffled = rows[:]
    random.shuffle(shuffled)

    sorted_rows = sorted(
        shuffled,
        key=lambda r: (
            _difficulty_rank(
                r.get("task", {}).get("metadata", {}).get("difficulty")
                or r.get("reward", {}).get("metadata", {}).get("difficulty")
            ),
            int(
                bool(
                    r.get(
                        "all_constraints_pass",
                        r.get("reward", {}).get("hard", {}).get("all_pass", 0),
                    )
                )
            ),
            float(r.get("combined_quality", r.get("reward", {}).get("combined_quality", 0.0))),
        ),
    )

    selected: list[dict] = []
    seen_signatures: set[tuple] = set()

    for row in sorted_rows:
        sig = _row_signature(row)
        if sig in seen_signatures:
            continue
        selected.append(row)
        seen_signatures.add(sig)
        if len(selected) >= n:
            return selected

    for row in sorted_rows:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) >= n:
            break

    return selected


def _extract_audit_row(domain: str, row: dict) -> dict:
    task = row.get("task", {})
    reward = row.get("reward", {})
    metadata = reward.get("metadata", {})

    profile = task.get("profile", {})
    constraints = task.get("constraints", {})
    preferences = task.get("preferences", {})
    context = task.get("context", {})
    evidence = row.get("audit_evidence", {})

    return {
        # Core identifiers
        "task_id": row.get("task_id", task.get("task_id")),
        "domain": row.get("domain", task.get("domain", domain)),
        "difficulty": task.get("metadata", {}).get("difficulty", metadata.get("difficulty")),

        # Raw plan + evaluation outputs
        "raw_plan": row.get("raw_plan", ""),
        "parse_success": row.get("parse_success", row.get("parsed", {}).get("parse_success")),
        "parse_error": row.get("parse_error", row.get("parsed", {}).get("parse_error")),
        "all_constraints_pass": row.get(
            "all_constraints_pass",
            reward.get("hard", {}).get("all_pass"),
        ),
        "violated_constraints": _safe_json_dumps(
            row.get("violated_constraints", metadata.get("violated_constraints", []))
        ),
        "hard_checks": _safe_json_dumps(
            row.get("hard_checks", reward.get("hard", {}).get("checks", {}))
        ),
        "soft_scores": _safe_json_dumps(
            row.get("soft_scores", reward.get("soft", {}).get("scores", {}))
        ),
        "soft_mean_score": row.get(
            "soft_mean_score",
            reward.get("soft", {}).get("mean_score"),
        ),
        "combined_quality": row.get(
            "combined_quality",
            reward.get("combined_quality"),
        ),

        # Portfolio task fields
        "portfolio_max_drawdown": constraints.get("max_drawdown", ""),
        "portfolio_min_assets": constraints.get("min_assets", ""),
        "portfolio_banned_sectors": _safe_json_dumps(constraints.get("banned_sectors", [])),
        "portfolio_target_risk": preferences.get("target_risk", ""),
        "portfolio_target_esg": preferences.get("target_esg", ""),
        "portfolio_market_regime": context.get("market_regime", ""),

        # Portfolio evidence
        "portfolio_actual_weight_sum": evidence.get("portfolio_actual_weight_sum", ""),
        "portfolio_actual_num_assets": evidence.get("portfolio_actual_num_assets", ""),
        "portfolio_actual_sectors": _safe_json_dumps(
            evidence.get("portfolio_actual_sectors", [])
        ),
        "portfolio_actual_max_drawdown": evidence.get("portfolio_actual_max_drawdown", ""),
        "portfolio_actual_annualized_return": evidence.get(
            "portfolio_actual_annualized_return", ""
        ),
        "portfolio_actual_annualized_volatility": evidence.get(
            "portfolio_actual_annualized_volatility", ""
        ),
        "portfolio_actual_esg_score": evidence.get("portfolio_actual_esg_score", ""),
        "portfolio_actual_risk_adjusted_return": evidence.get(
            "portfolio_actual_risk_adjusted_return", ""
        ),
        "portfolio_actual_turnover": evidence.get("portfolio_actual_turnover", ""),
        "portfolio_turnover_penalty_scale": preferences.get("turnover_penalty_scale", ""),

        # Retirement task fields
        "retirement_portfolio_value": profile.get("portfolio_value", ""),
        "retirement_current_age": profile.get("current_age", ""),
        "retirement_target_age": profile.get("target_age", ""),
        "retirement_min_monthly_income": constraints.get("min_monthly_income", ""),
        "retirement_inflation_assumption": context.get("inflation_assumption", ""),
        "retirement_return_assumption": context.get("portfolio_return_assumption", ""),
        "retirement_survival_threshold": constraints.get(
            "min_survival_probability",
            context.get("min_survival_probability", ""),
        ),

        # Retirement evidence
        "retirement_actual_monthly_income_target": evidence.get(
            "retirement_actual_monthly_income_target", ""
        ),
        "retirement_actual_inflation_adjusted_flag": evidence.get(
            "retirement_actual_inflation_adjusted_flag", ""
        ),
        "retirement_actual_survival_probability": evidence.get(
            "retirement_actual_survival_probability", ""
        ),
        "retirement_actual_depletion_probability": evidence.get(
            "retirement_actual_depletion_probability", ""
        ),
        "retirement_actual_ending_balance_median": evidence.get(
            "retirement_actual_ending_balance_median", ""
        ),
        "retirement_actual_ending_balance_mean": evidence.get(
            "retirement_actual_ending_balance_mean", ""
        ),
        "retirement_actual_years_survived_median": evidence.get(
            "retirement_actual_years_survived_median", ""
        ),
        "retirement_actual_lifestyle_quality_base": evidence.get(
            "retirement_actual_lifestyle_quality_base", ""
        ),
        "retirement_actual_lifestyle_quality_score": evidence.get(
            "retirement_actual_lifestyle_quality_score", ""
        ),
        "retirement_actual_withdrawal_smoothness": evidence.get(
            "retirement_actual_withdrawal_smoothness", ""
        ),
        "retirement_actual_starting_balance": evidence.get(
            "retirement_actual_starting_balance", ""
        ),
        "retirement_bequest_preference": preferences.get("bequest_preference", ""),

        # Loan task fields
        "loan_annual_income": profile.get("annual_income", ""),
        "loan_credit_band": profile.get("credit_band", ""),
        "loan_max_dti": constraints.get("max_dti", ""),
        "loan_max_ltv": constraints.get("max_ltv", ""),
        "loan_base_rate": context.get("base_rate", ""),

        # Loan evidence
        "loan_actual_monthly_payment": evidence.get("loan_actual_monthly_payment", ""),
        "loan_actual_dti": evidence.get("loan_actual_dti", ""),
        "loan_actual_ltv": evidence.get("loan_actual_ltv", ""),
        "loan_actual_rate_type": evidence.get("loan_actual_rate_type", ""),
        "loan_actual_total_interest_ratio": evidence.get(
            "loan_actual_total_interest_ratio", ""
        ),
        "loan_actual_term_months": evidence.get("loan_actual_term_months", ""),
        "loan_actual_prepayment_option": evidence.get("loan_actual_prepayment_option", ""),
        "loan_actual_annual_rate": evidence.get("loan_actual_annual_rate", ""),
        "loan_actual_loan_amount": evidence.get("loan_actual_loan_amount", ""),
        "loan_actual_property_value": evidence.get("loan_actual_property_value", ""),
        "loan_actual_existing_monthly_debt": evidence.get(
            "loan_actual_existing_monthly_debt", ""
        ),

        # Human audit fields
        "auditor_label": "",
        "auditor_notes": "",
        "is_reward_correct": "",
        "is_parse_result_correct": "",
    }


def main() -> None:
    seed_everything(42)

    out_path = Path("data/audits/week1_manual_audit.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    per_domain = {
        "portfolio": 17,
        "retirement": 17,
        "loan": 16,
    }

    selected_rows: list[dict] = []

    for domain, path in PROCESSED_FILES:
        rows = read_jsonl(path)
        sampled = _diverse_sample(rows, per_domain[domain])
        selected_rows.extend(_extract_audit_row(domain, row) for row in sampled)

    fieldnames = [
        "task_id",
        "domain",
        "difficulty",
        "raw_plan",
        "parse_success",
        "parse_error",
        "all_constraints_pass",
        "violated_constraints",
        "hard_checks",
        "soft_scores",
        "soft_mean_score",
        "combined_quality",

        "portfolio_max_drawdown",
        "portfolio_min_assets",
        "portfolio_banned_sectors",
        "portfolio_target_risk",
        "portfolio_target_esg",
        "portfolio_market_regime",
        "portfolio_actual_weight_sum",
        "portfolio_actual_num_assets",
        "portfolio_actual_sectors",
        "portfolio_actual_max_drawdown",
        "portfolio_actual_annualized_return",
        "portfolio_actual_annualized_volatility",
        "portfolio_actual_esg_score",
        "portfolio_actual_risk_adjusted_return",
        "portfolio_actual_turnover",
        "portfolio_turnover_penalty_scale",

        "retirement_portfolio_value",
        "retirement_current_age",
        "retirement_target_age",
        "retirement_min_monthly_income",
        "retirement_inflation_assumption",
        "retirement_return_assumption",
        "retirement_survival_threshold",
        "retirement_actual_monthly_income_target",
        "retirement_actual_inflation_adjusted_flag",
        "retirement_actual_survival_probability",
        "retirement_actual_depletion_probability",
        "retirement_actual_ending_balance_median",
        "retirement_actual_ending_balance_mean",
        "retirement_actual_years_survived_median",
        "retirement_actual_lifestyle_quality_base",
        "retirement_actual_lifestyle_quality_score",
        "retirement_actual_withdrawal_smoothness",
        "retirement_actual_starting_balance",
        "retirement_bequest_preference",

        "loan_annual_income",
        "loan_credit_band",
        "loan_max_dti",
        "loan_max_ltv",
        "loan_base_rate",
        "loan_actual_monthly_payment",
        "loan_actual_dti",
        "loan_actual_ltv",
        "loan_actual_rate_type",
        "loan_actual_total_interest_ratio",
        "loan_actual_term_months",
        "loan_actual_prepayment_option",
        "loan_actual_annual_rate",
        "loan_actual_loan_amount",
        "loan_actual_property_value",
        "loan_actual_existing_monthly_debt",

        "auditor_label",
        "auditor_notes",
        "is_reward_correct",
        "is_parse_result_correct",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    print(f"Wrote {len(selected_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
