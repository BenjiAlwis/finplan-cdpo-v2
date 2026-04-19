from __future__ import annotations
import csv, json
from pathlib import Path
from finplan.utils.io import read_jsonl, write_jsonl

PROCESSED = [("portfolio", "data/processed/portfolio_verified.jsonl"), ("retirement", "data/processed/retirement_verified.jsonl"), ("loan", "data/processed/loan_verified.jsonl")]

def _flatten(row: dict) -> dict:
    task = row.get("task", {}); reward = row.get("reward", {}); md = reward.get("metadata", {})
    hard = row.get("hard_checks", reward.get("hard", {}).get("checks", {}))
    soft = row.get("soft_scores", reward.get("soft", {}).get("scores", {}))
    violated = row.get("violated_constraints", md.get("violated_constraints", []))
    flat = {
        "task_id": row.get("task_id", task.get("task_id")),
        "domain": row.get("domain", task.get("domain")),
        "difficulty": task.get("metadata", {}).get("difficulty", md.get("difficulty")),
        "parse_success": row.get("parse_success", row.get("parsed", {}).get("parse_success")),
        "parse_error": row.get("parse_error", row.get("parsed", {}).get("parse_error")),
        "all_constraints_pass": row.get("all_constraints_pass", reward.get("hard", {}).get("all_pass")),
        "soft_mean_score": row.get("soft_mean_score", reward.get("soft", {}).get("mean_score")),
        "combined_quality": row.get("combined_quality", reward.get("combined_quality")),
        "hard_channel_json": json.dumps(hard, ensure_ascii=False),
        "soft_channel_json": json.dumps(soft, ensure_ascii=False),
        "violated_constraints_json": json.dumps(violated, ensure_ascii=False),
        "hard_pass_count": md.get("hard_pass_count", sum(int(v) for v in hard.values()) if hard else 0),
        "hard_total_count": md.get("hard_total_count", len(hard)),
        "soft_score_count": len(soft),
        "num_violated_constraints": len(violated),
    }
    for k, v in hard.items(): flat[f"hard__{k}"] = v
    for k, v in soft.items(): flat[f"soft__{k}"] = v
    return flat

def main() -> None:
    rows = []
    for _, path in PROCESSED:
        rows.extend(_flatten(r) for r in read_jsonl(path))
    out_dir = Path("data/diagnostics"); out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "week2_diagnostics.jsonl"; csv_path = out_dir / "week2_diagnostics.csv"
    write_jsonl(jsonl_path, rows)
    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {jsonl_path}")
    print(f"Wrote {len(rows)} rows to {csv_path}")

if __name__ == "__main__":
    main()
