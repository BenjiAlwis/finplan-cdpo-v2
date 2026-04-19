from __future__ import annotations
import json
from typing import Any
from finplan.parsers.schema_validator import validate_plan_schema
from finplan.types import DomainName, ParsedPlan

class PlanParser:
    def parse(self, raw_text: str, domain: DomainName, expected_schema: dict[str, Any] | None = None) -> ParsedPlan:
        expected_schema = expected_schema or {}
        try:
            structured = json.loads(raw_text)
        except Exception as exc:
            return ParsedPlan(domain=domain, raw_text=raw_text, structured={}, parse_success=False, parse_error=f"JSON parse error: {exc}")
        if not isinstance(structured, dict):
            return ParsedPlan(domain=domain, raw_text=raw_text, structured={}, parse_success=False, parse_error="Top-level parsed object must be a JSON object.")
        ok, errors = validate_plan_schema(domain, structured, expected_schema)
        if not ok:
            return ParsedPlan(domain=domain, raw_text=raw_text, structured=structured, parse_success=False, parse_error="Schema validation failed: " + "; ".join(errors))
        return ParsedPlan(domain=domain, raw_text=raw_text, structured=structured, parse_success=True, parse_error=None)
