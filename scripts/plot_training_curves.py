from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = [
    ('hard_pass_rate', 'Constraint Compliance Rate'),
    ('mean_soft_score', 'Soft Preference Score'),
    ('mean_combined_quality', 'Combined Quality'),
    ('advantage_groups', 'Distinct Advantage Groups'),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary-csv', type=str, default='data/analysis/advantage_distribution_summary.csv')
    parser.add_argument('--output-dir', type=str, default='data/analysis')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    with open(args.summary_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped[row['method']].append(row)

    for metric, title in METRICS:
        plt.figure(figsize=(7, 4))
        for method, rows in grouped.items():
            rows = sorted(rows, key=lambda r: int(r['step']))
            xs = [int(r['step']) for r in rows]
            ys = [float(r[metric]) for r in rows]
            plt.plot(xs, ys, label=method.upper())
        plt.xlabel('Training step')
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        out_path = out_dir / f'{metric}.png'
        plt.savefig(out_path)
        plt.close()
        print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
