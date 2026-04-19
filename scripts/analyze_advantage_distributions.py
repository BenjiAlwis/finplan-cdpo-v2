from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--grpo-log', type=str, required=True)
    parser.add_argument('--gdpo-log', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='data/analysis')
    return parser.parse_args()


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]


def _extract_scalar(record: dict[str, Any], keys: list[str], default: float = 0.0) -> float:
    for key in keys:
        if key in record:
            try:
                return float(record[key])
            except (TypeError, ValueError):
                continue
    return default


def _summarize(records: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        step = int(record.get('step', record.get('global_step', 0)))
        grouped[step].append(record)

    rows: list[dict[str, Any]] = []
    for step in sorted(grouped):
        chunk = grouped[step]
        rows.append(
            {
                'method': method,
                'step': step,
                'n_records': len(chunk),
                'hard_pass_rate': sum(_extract_scalar(r, ['hard_pass_rate', 'batch/hard_pass_rate']) for r in chunk) / len(chunk),
                'mean_soft_score': sum(_extract_scalar(r, ['mean_soft_score', 'batch/mean_soft_score']) for r in chunk) / len(chunk),
                'mean_combined_quality': sum(_extract_scalar(r, ['mean_combined_quality', 'batch/mean_combined_quality']) for r in chunk) / len(chunk),
                'advantage_groups': sum(_extract_scalar(r, ['advantage_groups/combined_quality', 'advantage_groups']) for r in chunk) / len(chunk),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _summarize(_read_jsonl(args.grpo_log), 'grpo')
    rows += _summarize(_read_jsonl(args.gdpo_log), 'gdpo')

    out_csv = out_dir / 'advantage_distribution_summary.csv'
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['method', 'step', 'n_records', 'hard_pass_rate', 'mean_soft_score', 'mean_combined_quality', 'advantage_groups'],
        )
        writer.writeheader()
        writer.writerows(rows)

    verdict = {
        'grpo_rows': sum(1 for r in rows if r['method'] == 'grpo'),
        'gdpo_rows': sum(1 for r in rows if r['method'] == 'gdpo'),
        'note': 'Lower distinct advantage-group counts in GRPO than GDPO are the primary signal-collapse indicator.',
    }
    with (out_dir / 'signal_collapse_summary.json').open('w', encoding='utf-8') as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    print(f'Wrote {out_csv}')
    print(f'Wrote {out_dir / "signal_collapse_summary.json"}')


if __name__ == '__main__':
    main()
