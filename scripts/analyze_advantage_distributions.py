from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON on {path}:{line_no}: {exc}') from exc
            if isinstance(payload, dict):
                yield payload


def flatten_summary(row: dict[str, Any]) -> dict[str, Any]:
    """Keep scalar fields and compact nested collapse summaries."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, (str, int, float, bool)) or nested_value is None:
                    out[f'{key}.{nested_key}'] = nested_value
                elif nested_key != 'collapsed_bucket_examples':
                    out[f'{key}.{nested_key}'] = json.dumps(nested_value, sort_keys=True)
    return out


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_plan_rows(plan_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pattern = Counter(str(row.get('violation_pattern', '')) for row in plan_rows)
    by_domain = Counter(str(row.get('domain', 'unknown')) for row in plan_rows)
    n = max(len(plan_rows), 1)
    return {
        'plan_rows': len(plan_rows),
        'domains': dict(by_domain),
        'distinct_violation_patterns': len(by_pattern),
        'top_violation_patterns': by_pattern.most_common(20),
        'hard_pass_rate': sum(float(row.get('all_constraints_pass') or 0.0) for row in plan_rows) / n,
        'parse_success_rate': sum(float(bool(row.get('parse_success'))) for row in plan_rows) / n,
        'mean_soft_score': sum(float(row.get('soft_mean_score') or 0.0) for row in plan_rows) / n,
        'mean_combined_quality': sum(float(row.get('combined_quality') or 0.0) for row in plan_rows) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Summarize GRPO/GDPO/CDPO diagnostics JSONL files for signal-collapse analysis.'
    )
    parser.add_argument(
        'diagnostics',
        nargs='+',
        help='One or more diagnostics.jsonl files, e.g. data/rl_runs/grpo_baseline/diagnostics.jsonl',
    )
    parser.add_argument('--out-dir', default='data/analysis/week23', help='Directory for CSV/JSON summaries.')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_rows: list[dict[str, Any]] = []
    plan_rows_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for raw_path in args.diagnostics:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        for row in iter_jsonl(path):
            row.setdefault('source_file', str(path))
            record_type = row.get('record_type')
            method = str(row.get('method', path.parent.name))
            if record_type == 'batch_summary':
                batch_rows.append(flatten_summary(row))
            elif record_type == 'plan_eval':
                plan_rows_by_method[method].append(row)

    write_csv(batch_rows, out_dir / 'batch_summaries.csv')

    method_summaries = {
        method: summarize_plan_rows(rows)
        for method, rows in sorted(plan_rows_by_method.items())
    }
    with (out_dir / 'plan_eval_summary.json').open('w', encoding='utf-8') as f:
        json.dump(method_summaries, f, ensure_ascii=False, indent=2, sort_keys=True)

    collapse_rows: list[dict[str, Any]] = []
    for row in batch_rows:
        collapse_rows.append(
            {
                'source_file': row.get('source_file'),
                'method': row.get('method'),
                'call_index': row.get('call_index'),
                'reward_name': row.get('reward_name'),
                'hard_pass_rate': row.get('hard_pass_rate'),
                'mean_soft_score': row.get('mean_soft_score'),
                'mean_combined_quality': row.get('mean_combined_quality'),
                'reward_distinct_groups': row.get('reward_distinct_groups'),
                'grpo_advantage_groups': row.get('grpo_advantage_groups'),
                'gdpo_diagnostic_advantage_groups': row.get('gdpo_diagnostic_advantage_groups'),
                'cdpo_advantage_groups': row.get('cdpo_advantage_groups'),
                'grpo_collapsed_buckets': row.get('grpo_signal_collapse.collapsed_advantage_buckets'),
                'gdpo_collapsed_buckets': row.get('gdpo_signal_collapse.collapsed_advantage_buckets'),
                'cdpo_collapsed_buckets': row.get('cdpo_signal_collapse.collapsed_advantage_buckets'),
                'grpo_distinct_patterns': row.get('grpo_signal_collapse.distinct_patterns'),
                'gdpo_distinct_patterns': row.get('gdpo_signal_collapse.distinct_patterns'),
                'cdpo_distinct_patterns': row.get('cdpo_signal_collapse.distinct_patterns'),
            }
        )
    write_csv(collapse_rows, out_dir / 'signal_collapse_summary.csv')

    print(f'Wrote {out_dir / "batch_summaries.csv"}')
    print(f'Wrote {out_dir / "signal_collapse_summary.csv"}')
    print(f'Wrote {out_dir / "plan_eval_summary.json"}')


if __name__ == '__main__':
    main()
