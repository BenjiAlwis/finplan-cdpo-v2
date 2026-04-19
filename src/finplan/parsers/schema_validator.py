from __future__ import annotations
from typing import Any

def validate_plan_schema(domain: str, structured: dict[str, Any], expected_schema: dict[str, Any]) -> tuple[bool, list[str]]:
    if domain == "portfolio":
        return validate_portfolio_schema(structured, expected_schema)
    if domain == "retirement":
        return validate_retirement_schema(structured, expected_schema)
    if domain == "loan":
        return validate_loan_schema(structured, expected_schema)
    return False, [f"Unknown domain: {domain}"]

def _req(structured: dict[str, Any], fields: list[str]) -> list[str]:
    return [f"Missing required field: {f}" for f in fields if f not in structured]

def validate_portfolio_schema(structured: dict[str, Any], expected_schema: dict[str, Any]) -> tuple[bool, list[str]]:
    errors = _req(structured, expected_schema.get("required_fields", ["allocations"]))
    allocs = structured.get("allocations")
    if not isinstance(allocs, list) or not allocs:
        errors.append("Field 'allocations' must be a non-empty list.")
        return False, errors
    for i, item in enumerate(allocs):
        if not isinstance(item, dict):
            errors.append(f"Allocation {i} must be an object.")
            continue
        for key in ["asset", "sector", "weight"]:
            if key not in item:
                errors.append(f"Allocation {i} missing '{key}'.")
    return len(errors) == 0, errors

def validate_retirement_schema(structured: dict[str, Any], expected_schema: dict[str, Any]) -> tuple[bool, list[str]]:
    errors = _req(structured, expected_schema.get("required_fields", ["monthly_income_target", "inflation_adjusted"]))
    if "monthly_income_target" in structured and not isinstance(structured["monthly_income_target"], (int, float)):
        errors.append("Field 'monthly_income_target' must be numeric.")
    if "inflation_adjusted" in structured and not isinstance(structured["inflation_adjusted"], bool):
        errors.append("Field 'inflation_adjusted' must be boolean.")
    return len(errors) == 0, errors

def validate_loan_schema(structured: dict[str, Any], expected_schema: dict[str, Any]) -> tuple[bool, list[str]]:
    errors = _req(structured, expected_schema.get("required_fields", ["loan_amount", "term_months", "rate_type"]))
    if "rate_type" in structured and structured["rate_type"] not in {"fixed", "variable"}:
        errors.append("Field 'rate_type' must be 'fixed' or 'variable'.")
    return len(errors) == 0, errors
