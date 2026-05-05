from __future__ import annotations

import argparse
import csv
from pathlib import Path
from summarize_rl_run import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare multiple RL diagnostics runs.')
    parser.add_argument('--run', action='append', required=True, help='label:path/to/run_or_diagnostics.jsonl')
    parser.add_argument('--csv-out', type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for item in args.run:
        if ':' not in item:
            raise ValueError(f'Expected label:path, got {item}')
        label, raw_path = item.split(':', 1)
        path = Path(raw_path)
        if path.is_dir():
            path = path / 'diagnostics.jsonl'
        m = summarize(path)
        rows.append({
            'method': label,
            'batch_summaries': m.get('batch_summaries', 0),
            'plan_evals': m.get('plan_evals', 0),
            'parse_success_rate': round(m.get('parse_success_rate', 0.0), 4),
            'hard_pass_rate': round(m.get('hard_pass_rate', 0.0), 4),
            'mean_combined_quality': round(m.get('mean_combined_quality', 0.0), 4),
            'mean_soft_score': round(m.get('mean_soft_score', 0.0), 4),
            'mean_reward': round(m.get('mean_reward', 0.0), 4),
            'grpo_collapsed_buckets': m.get('grpo_collapsed_buckets', 0),
            'gdpo_collapsed_buckets': m.get('gdpo_collapsed_buckets', 0),
            'cdpo_collapsed_buckets': m.get('cdpo_collapsed_buckets', 0),
            'mean_cdpo_alpha': round(m.get('mean_cdpo_alpha', 0.0), 4),
            'parse_success_true_plans': m.get('parse_success_true_plans', ''),
            'all_constraints_pass_plans': m.get('all_constraints_pass_plans', ''),
        })

    headers = list(rows[0].keys())
    print(','.join(headers))
    for row in rows:
        print(','.join(str(row[h]) for h in headers))

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print(f'Wrote {args.csv_out}')


if __name__ == '__main__':
    main()
