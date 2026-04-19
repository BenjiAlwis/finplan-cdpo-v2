from __future__ import annotations

import argparse
import csv
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary-csv', type=str, default='data/analysis/advantage_distribution_summary.csv')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped = defaultdict(list)
    with open(args.summary_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped[row['method']].append(row)

    for method in sorted(grouped):
        rows = sorted(grouped[method], key=lambda r: int(r['step']))
        last = rows[-1]
        print(f'\nMethod: {method.upper()}')
        print(f"  Final step: {last['step']}")
        print(f"  Final hard pass rate: {float(last['hard_pass_rate']):.4f}")
        print(f"  Final mean soft score: {float(last['mean_soft_score']):.4f}")
        print(f"  Final mean combined quality: {float(last['mean_combined_quality']):.4f}")
        print(f"  Final distinct advantage groups: {float(last['advantage_groups']):.2f}")

    if 'grpo' in grouped and 'gdpo' in grouped:
        grpo_last = sorted(grouped['grpo'], key=lambda r: int(r['step']))[-1]
        gdpo_last = sorted(grouped['gdpo'], key=lambda r: int(r['step']))[-1]
        grpo_groups = float(grpo_last['advantage_groups'])
        gdpo_groups = float(gdpo_last['advantage_groups'])
        print('\nWeek 2 verdict:')
        if gdpo_groups > grpo_groups:
            print('  Evidence is consistent with monolithic GRPO exhibiting more advantage collapse than GDPO.')
        elif gdpo_groups < grpo_groups:
            print('  GDPO did not preserve more distinct advantage groups in the final step; inspect full curves before concluding.')
        else:
            print('  Final-step distinct advantage groups are tied; inspect earlier steps and raw logs.')


if __name__ == '__main__':
    main()
