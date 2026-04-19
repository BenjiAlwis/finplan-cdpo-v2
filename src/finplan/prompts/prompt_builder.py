from __future__ import annotations

import json
from typing import Any

from finplan.types import TaskInstance


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _output_schema_hint(task: TaskInstance) -> str:
    schema = task.expected_plan_schema or {}
    required = schema.get("required_fields", [])
    if required:
        req = ", ".join(required)
        return (
            "Return only a valid JSON object with these required top-level fields: "
            f"{req}. Do not include markdown fences or prose outside JSON."
        )
    return "Return only a valid JSON object. Do not include markdown fences or prose outside JSON."


def build_portfolio_prompt(task: TaskInstance) -> str:
    return (
        "You are a financial planning assistant. Produce a portfolio allocation plan as JSON.\n\n"
        "Client/task context:\n"
        f"Profile:\n{_json_block(task.profile)}\n\n"
        f"Hard constraints:\n{_json_block(task.constraints)}\n\n"
        f"Soft preferences:\n{_json_block(task.preferences)}\n\n"
        f"Market/environment context:\n{_json_block(task.context)}\n\n"
        "Goal:\n"
        "Propose a diversified portfolio allocation that satisfies the hard constraints first, then aligns with the soft preferences.\n\n"
        f"Output requirements:\n{_output_schema_hint(task)}"
    )


def build_retirement_prompt(task: TaskInstance) -> str:
    return (
        "You are a financial planning assistant. Produce a retirement income plan as JSON.\n\n"
        "Client/task context:\n"
        f"Profile:\n{_json_block(task.profile)}\n\n"
        f"Hard constraints:\n{_json_block(task.constraints)}\n\n"
        f"Soft preferences:\n{_json_block(task.preferences)}\n\n"
        f"Economic/environment context:\n{_json_block(task.context)}\n\n"
        "Goal:\n"
        "Propose a retirement drawdown plan that preserves feasibility under the hard constraints before optimizing the softer lifestyle and bequest preferences.\n\n"
        f"Output requirements:\n{_output_schema_hint(task)}"
    )


def build_loan_prompt(task: TaskInstance) -> str:
    return (
        "You are a financial planning assistant. Produce a loan structuring recommendation as JSON.\n\n"
        "Client/task context:\n"
        f"Profile:\n{_json_block(task.profile)}\n\n"
        f"Hard constraints:\n{_json_block(task.constraints)}\n\n"
        f"Soft preferences:\n{_json_block(task.preferences)}\n\n"
        f"Rate/environment context:\n{_json_block(task.context)}\n\n"
        "Goal:\n"
        "Propose a compliant loan structure that satisfies underwriting and regulatory constraints before optimizing cost and flexibility.\n\n"
        f"Output requirements:\n{_output_schema_hint(task)}"
    )


def build_prompt(task: TaskInstance) -> str:
    if task.domain == "portfolio":
        return build_portfolio_prompt(task)
    if task.domain == "retirement":
        return build_retirement_prompt(task)
    if task.domain == "loan":
        return build_loan_prompt(task)
    raise ValueError(f"Unsupported domain: {task.domain}")
