from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def count_distinct_advantage_groups(values: Iterable[float], tol: float = 1e-6) -> int:
    groups: list[float] = []
    for value in sorted(float(v) for v in values):
        if not groups or abs(value - groups[-1]) > tol:
            groups.append(value)
    return len(groups)


def summarize_batch(eval_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not eval_rows:
        return {
            'batch_size': 0.0,
            'parse_success_rate': 0.0,
            'hard_pass_rate': 0.0,
            'mean_soft_score': 0.0,
            'mean_combined_quality': 0.0,
        }

    n = len(eval_rows)
    parse_successes = sum(1 for row in eval_rows if row.get('parse_success'))
    hard_passes = sum(1 for row in eval_rows if row.get('all_constraints_pass'))
    mean_soft = sum(float(row.get('soft_mean_score', 0.0)) for row in eval_rows) / n
    mean_combined = sum(float(row.get('combined_quality', 0.0)) for row in eval_rows) / n

    return {
        'batch_size': float(n),
        'parse_success_rate': parse_successes / n,
        'hard_pass_rate': hard_passes / n,
        'mean_soft_score': mean_soft,
        'mean_combined_quality': mean_combined,
    }


def summarize_constraint_failures(eval_rows: list[dict[str, Any]]) -> dict[str, float]:
    counter: Counter[str] = Counter()
    n = len(eval_rows)
    if n == 0:
        return {}

    for row in eval_rows:
        for name, value in row.get('hard_checks', {}).items():
            if int(value) == 0:
                counter[name] += 1

    return {f'failure_rate/{name}': count / n for name, count in counter.items()}


def summarize_soft_scores(eval_rows: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in eval_rows:
        for name, value in row.get('soft_scores', {}).items():
            totals[name] = totals.get(name, 0.0) + float(value)
            counts[name] = counts.get(name, 0) + 1
    return {
        f'soft_score/{name}': totals[name] / counts[name]
        for name in totals
        if counts[name] > 0
    }


def maybe_log_metric(kwargs: dict[str, Any], key: str, value: float) -> None:
    fn = kwargs.get('log_metric')
    if callable(fn):
        fn(key, float(value))


def maybe_log_extra(kwargs: dict[str, Any], key: str, value: Any) -> None:
    fn = kwargs.get('log_extra')
    if callable(fn):
        fn(key, value)
