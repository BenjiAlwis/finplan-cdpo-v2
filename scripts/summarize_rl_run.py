from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def mean(values):
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def load_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row.get('record_type') == 'batch_summary':
                summaries.append(row)
            elif row.get('record_type') == 'plan_eval':
                plans.append(row)
    return summaries, plans


def collapse_sum(summaries: list[dict[str, Any]], key: str) -> int:
    return sum(r.get(key, {}).get('collapsed_advantage_buckets', 0) for r in summaries)


def summarize(path: Path) -> dict[str, Any]:
    summaries, plans = load_rows(path)
    if not summaries:
        raise ValueError(f'No batch_summary rows found in {path}')

    metrics: dict[str, Any] = {
        'path': str(path),
        'batch_summaries': len(summaries),
        'plan_evals': len(plans),
        'parse_success_rate': mean(r.get('parse_success_rate', 0.0) for r in summaries),
        'hard_pass_rate': mean(r.get('hard_pass_rate', 0.0) for r in summaries),
        'mean_combined_quality': mean(r.get('mean_combined_quality', 0.0) for r in summaries),
        'mean_soft_score': mean(r.get('mean_soft_score', 0.0) for r in summaries),
        'mean_reward': mean(r.get('reward_mean', 0.0) for r in summaries),
        'grpo_collapsed_buckets': collapse_sum(summaries, 'grpo_signal_collapse'),
        'gdpo_collapsed_buckets': collapse_sum(summaries, 'gdpo_signal_collapse'),
        'cdpo_collapsed_buckets': collapse_sum(summaries, 'cdpo_signal_collapse'),
        'mean_grpo_advantage_groups': mean(r.get('grpo_advantage_groups', 0.0) for r in summaries),
        'mean_gdpo_advantage_groups': mean(r.get('gdpo_diagnostic_advantage_groups', 0.0) for r in summaries),
        'mean_cdpo_advantage_groups': mean(r.get('cdpo_advantage_groups', 0.0) for r in summaries),
    }
    if any('cdpo_alpha' in r for r in summaries):
        metrics['mean_cdpo_alpha'] = mean(r.get('cdpo_alpha') for r in summaries)
        metrics['min_cdpo_alpha'] = min((float(r.get('cdpo_alpha')) for r in summaries if r.get('cdpo_alpha') is not None), default=0.0)
        metrics['max_cdpo_alpha'] = max((float(r.get('cdpo_alpha')) for r in summaries if r.get('cdpo_alpha') is not None), default=0.0)

    if plans:
        metrics['parse_success_true_plans'] = sum(1 for r in plans if r.get('parse_success'))
        metrics['all_constraints_pass_plans'] = sum(1 for r in plans if r.get('all_constraints_pass'))
        by_domain = defaultdict(list)
        for r in plans:
            by_domain[r.get('domain', 'unknown')].append(r)
        metrics['domain_metrics'] = {}
        for domain, rows in sorted(by_domain.items()):
            metrics['domain_metrics'][domain] = {
                'n': len(rows),
                'parse_success': mean(1.0 if r.get('parse_success') else 0.0 for r in rows),
                'hard_pass': mean(float(r.get('all_constraints_pass', 0.0)) for r in rows),
                'combined_quality': mean(r.get('combined_quality', 0.0) for r in rows),
                'soft_mean_score': mean(r.get('soft_mean_score', 0.0) for r in rows),
            }
        violations = Counter()
        for r in plans:
            for v in r.get('violated_constraints', []):
                violations[v] += 1
        metrics['top_violated_constraints'] = violations.most_common(10)
        metrics['top_violation_patterns'] = Counter(r.get('violation_pattern') for r in plans).most_common(10)
    return metrics


def print_summary(metrics: dict[str, Any]) -> None:
    print('=== RL run summary ===')
    for key in [
        'path', 'batch_summaries', 'plan_evals', 'parse_success_rate', 'hard_pass_rate',
        'mean_combined_quality', 'mean_soft_score', 'mean_reward',
        'grpo_collapsed_buckets', 'gdpo_collapsed_buckets', 'cdpo_collapsed_buckets',
        'mean_grpo_advantage_groups', 'mean_gdpo_advantage_groups', 'mean_cdpo_advantage_groups',
        'mean_cdpo_alpha', 'min_cdpo_alpha', 'max_cdpo_alpha',
        'parse_success_true_plans', 'all_constraints_pass_plans'
    ]:
        if key in metrics:
            print(f'{key}: {metrics[key]}')
    if 'domain_metrics' in metrics:
        print('\n=== Domain metrics ===')
        for domain, vals in metrics['domain_metrics'].items():
            print(domain, vals)
    if 'top_violated_constraints' in metrics:
        print('\n=== Top violated constraints ===')
        for name, count in metrics['top_violated_constraints']:
            print(count, name)
    if 'top_violation_patterns' in metrics:
        print('\n=== Top violation patterns ===')
        for pattern, count in metrics['top_violation_patterns']:
            print(count, pattern)


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize a diagnostics.jsonl RL run.')
    parser.add_argument('--run', type=Path, required=True, help='Path to diagnostics.jsonl or run directory')
    parser.add_argument('--json-out', type=Path, default=None)
    args = parser.parse_args()

    path = args.run
    if path.is_dir():
        path = path / 'diagnostics.jsonl'
    metrics = summarize(path)
    print_summary(metrics)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        print(f'Wrote {args.json_out}')


if __name__ == '__main__':
    main()
