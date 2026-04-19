from __future__ import annotations
import json
from pathlib import Path
from finplan.types import TaskInstance
from finplan.utils.io import read_jsonl, write_jsonl

RAW_INPUTS = [("portfolio", "data/raw/portfolio_instances.jsonl"), ("retirement", "data/raw/retirement_instances.jsonl"), ("loan", "data/raw/loan_instances.jsonl")]

def make_invalid_portfolio(task: TaskInstance):
    banned = task.constraints.get("banned_sectors", [])
    bad_sector = banned[0] if banned else "TOBACCO"
    return [
        {"failure_mode": "weights_sum_invalid", "raw_plan": json.dumps({"allocations": [{"asset": "US_EQ", "sector": "TECH", "weight": 0.7}, {"asset": "BONDS", "sector": "GOVT", "weight": 0.4}]})},
        {"failure_mode": "banned_sector_invalid", "raw_plan": json.dumps({"allocations": [{"asset": "US_EQ", "sector": bad_sector, "weight": 0.5}, {"asset": "BONDS", "sector": "GOVT", "weight": 0.5}]})},
        {"failure_mode": "diversification_invalid", "raw_plan": json.dumps({"allocations": [{"asset": "US_EQ", "sector": "TECH", "weight": 1.0}]})},
        {"failure_mode": "schema_invalid_missing_allocations", "raw_plan": json.dumps({"not_allocations": []})},
    ]

def make_invalid_retirement(task: TaskInstance):
    floor = float(task.constraints.get("min_monthly_income", 3000.0))
    return [
        {"failure_mode": "income_floor_invalid", "raw_plan": json.dumps({"monthly_income_target": max(0.0, floor - 1000.0), "inflation_adjusted": True})},
        {"failure_mode": "inflation_handling_invalid", "raw_plan": json.dumps({"monthly_income_target": floor + 500.0, "inflation_adjusted": False})},
        {"failure_mode": "early_depletion_invalid", "raw_plan": json.dumps({"monthly_income_target": floor + 5000.0, "inflation_adjusted": True})},
        {"failure_mode": "schema_invalid_missing_fields", "raw_plan": json.dumps({"monthly_income_target": floor + 500.0})},
    ]

def make_invalid_loan(task: TaskInstance):
    return [
        {"failure_mode": "dti_invalid", "raw_plan": json.dumps({"loan_amount": 600000.0, "property_value": 900000.0, "annual_rate": 0.09, "term_months": 120, "rate_type": "fixed", "existing_monthly_debt": 2500.0, "prepayment_option": True})},
        {"failure_mode": "ltv_invalid", "raw_plan": json.dumps({"loan_amount": 350000.0, "property_value": 380000.0, "annual_rate": 0.05, "term_months": 360, "rate_type": "fixed", "existing_monthly_debt": 200.0, "prepayment_option": True})},
        {"failure_mode": "regulatory_invalid", "raw_plan": json.dumps({"loan_amount": 250000.0, "property_value": 400000.0, "annual_rate": 0.05, "term_months": 360, "rate_type": "balloon", "existing_monthly_debt": 200.0, "prepayment_option": True})},
        {"failure_mode": "schema_invalid_missing_fields", "raw_plan": json.dumps({"loan_amount": 250000.0, "term_months": 360})},
    ]

def main() -> None:
    rows_out = []
    for domain, path in RAW_INPUTS:
        rows = read_jsonl(path)[:10]
        for row in rows:
            task = TaskInstance(**row)
            invalids = make_invalid_portfolio(task) if domain == "portfolio" else make_invalid_retirement(task) if domain == "retirement" else make_invalid_loan(task)
            for item in invalids:
                rows_out.append({"task": task.to_dict(), "task_id": task.task_id, "domain": task.domain, "difficulty": task.metadata.get("difficulty"), "failure_mode": item["failure_mode"], "raw_plan": item["raw_plan"]})
    out_path = Path("data/audits/invalid_plans.jsonl")
    write_jsonl(out_path, rows_out)
    print(f"Wrote {len(rows_out)} invalid-plan rows to {out_path}")

if __name__ == "__main__":
    main()
