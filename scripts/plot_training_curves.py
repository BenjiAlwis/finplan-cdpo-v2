from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> float | None:
    if value in (None, ''):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def plot_metric(rows: list[dict[str, Any]], metric: str, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        y = to_float(row.get(metric))
        x = to_float(row.get('call_index'))
        if y is None or x is None:
            continue
        method = str(row.get('method') or row.get('source_file') or 'unknown')
        reward_name = str(row.get('reward_name') or '')
        # For GDPO there may be many per-signal reward summaries. Use all rows for
        # diagnostics, but keep labels compact.
        label = method if not reward_name else f'{method}:{reward_name}'
        series[label].append((x, y))

    if not series:
        print(f'Skipping {metric}: no data')
        return

    fig = plt.figure()
    for label, points in sorted(series.items()):
        points = sorted(points)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, label=label)
    plt.xlabel('Reward-function call index')
    plt.ylabel(metric)
    plt.title(metric.replace('_', ' ').title())
    plt.legend(fontsize='small')
    fig.tight_layout()
    output_path = out_dir / f'{metric}.png'
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f'Wrote {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot Week 2/3 training and diagnostic curves.')
    parser.add_argument(
        '--summary-csv',
        default='data/analysis/week23/batch_summaries.csv',
        help='CSV produced by scripts/analyze_advantage_distributions.py',
    )
    parser.add_argument('--out-dir', default='data/analysis/week23/figures')
    parser.add_argument(
        '--metrics',
        nargs='*',
        default=[
            'hard_pass_rate',
            'mean_soft_score',
            'mean_combined_quality',
            'reward_distinct_groups',
            'grpo_advantage_groups',
            'gdpo_diagnostic_advantage_groups',
            'cdpo_advantage_groups',
            'grpo_signal_collapse.collapsed_advantage_buckets',
            'gdpo_signal_collapse.collapsed_advantage_buckets',
            'cdpo_signal_collapse.collapsed_advantage_buckets',
        ],
    )
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    if not summary_csv.exists():
        raise FileNotFoundError(
            f'{summary_csv} not found. Run scripts/analyze_advantage_distributions.py first.'
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(summary_csv)
    for metric in args.metrics:
        plot_metric(rows, metric, out_dir)


if __name__ == '__main__':
    main()
