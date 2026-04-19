from __future__ import annotations
from pathlib import Path
from finplan.env.finplan_env import FinPlanEnv
from finplan.types import TaskInstance
from finplan.utils.io import read_jsonl, write_jsonl

EXPECTED = {
    "portfolio": {"weights_sum_invalid": "weights_sum_valid", "banned_sector_invalid": "banned_sector_valid", "diversification_invalid": "diversification_valid", "schema_invalid_missing_allocations": "parse_valid"},
    "retirement": {"income_floor_invalid": "income_floor_valid", "inflation_handling_invalid": "inflation_handling_valid", "early_depletion_invalid": "no_early_depletion", "schema_invalid_missing_fields": "parse_valid"},
    "loan": {"dti_invalid": "dti_valid", "ltv_invalid": "ltv_valid", "regulatory_invalid": "regulatory_valid", "schema_invalid_missing_fields": "parse_valid"},
}

def main() -> None:
    env = FinPlanEnv(); rows = read_jsonl("data/audits/invalid_plans.jsonl"); out = []
    for row in rows:
        task = TaskInstance(**row["task"])
        parsed, reward = env.evaluate(task, row["raw_plan"])
        expected_check = EXPECTED[row["domain"]][row["failure_mode"]]
        observed = int((not parsed.parse_success and reward.hard.checks["parse_valid"] == 0) if expected_check == "parse_valid" else reward.hard.checks.get(expected_check, 1) == 0)
        out.append({"task_id": row["task_id"], "domain": row["domain"], "difficulty": row.get("difficulty"), "failure_mode": row["failure_mode"], "expected_failed_check": expected_check, "expected_failure_observed": observed, "parse_success": parsed.parse_success, "parse_error": parsed.parse_error, "hard_checks": reward.hard.checks, "soft_scores": reward.soft.scores, "violated_constraints": reward.metadata.get("violated_constraints", []), "all_constraints_pass": reward.hard.all_pass, "combined_quality": reward.combined_quality})
    out_path = Path("data/audits/invalid_plan_stress_results.jsonl")
    write_jsonl(out_path, out)
    total = len(out); correct = sum(int(r["expected_failure_observed"]) for r in out)
    print(f"Wrote {total} stress-test rows to {out_path}")
    print(f"Expected failure observed in {correct}/{total} cases ({correct / max(total, 1):.2%})")

if __name__ == "__main__":
    main()
