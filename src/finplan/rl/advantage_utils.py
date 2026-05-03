from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


EPS = 1e-8


@dataclass(frozen=True)
class AdvantageSummary:
    distinct_groups: int
    mean: float
    std: float
    min_value: float
    max_value: float


def mean(values: Sequence[float]) -> float:
    return sum(float(v) for v in values) / max(len(values), 1)


def std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mu = mean(values)
    return (sum((float(v) - mu) ** 2 for v in values) / len(values)) ** 0.5


def normalize_zscore(values: Sequence[float], *, divide_by_std: bool = True) -> list[float]:
    if not values:
        return []
    mu = mean(values)
    if not divide_by_std:
        return [float(v) - mu for v in values]
    sigma = std(values)
    if sigma < EPS:
        return [0.0 for _ in values]
    return [(float(v) - mu) / (sigma + EPS) for v in values]


def batch_normalize(values: Sequence[float]) -> list[float]:
    return normalize_zscore(values, divide_by_std=True)


def count_distinct_groups(values: Iterable[float], tol: float = 1e-6) -> int:
    groups: list[float] = []
    for value in sorted(float(v) for v in values):
        if not groups or abs(value - groups[-1]) > tol:
            groups.append(value)
    return len(groups)


def summarize_values(values: Sequence[float]) -> AdvantageSummary:
    if not values:
        return AdvantageSummary(0, 0.0, 0.0, 0.0, 0.0)
    return AdvantageSummary(
        distinct_groups=count_distinct_groups(values),
        mean=mean(values),
        std=std(values),
        min_value=min(float(v) for v in values),
        max_value=max(float(v) for v in values),
    )


def group_indices_by_task(task_json: Sequence[str]) -> list[list[int]]:
    """Group completions belonging to the same prompt/task.

    TRL GRPO batches are normally ordered as prompt_1 completions, prompt_2
    completions, etc. This function does not assume a fixed group size; it
    starts a new group whenever the task_json changes. This makes diagnostics
    robust to different num_generations values.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    previous: str | None = None

    for idx, task in enumerate(task_json):
        if previous is None or task == previous:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
        previous = task

    if current:
        groups.append(current)
    return groups


def violation_pattern(hard_checks: dict[str, Any]) -> str:
    return '|'.join(f'{name}={int(value)}' for name, value in sorted(hard_checks.items()))


def extract_signal_vector(
    row: dict[str, Any],
    *,
    hard_signal_names: Sequence[str],
    soft_signal_names: Sequence[str],
) -> dict[str, float]:
    hard = row.get('hard_checks', {}) or {}
    soft = row.get('soft_scores', {}) or {}
    vector: dict[str, float] = {}
    for name in hard_signal_names:
        vector[f'hard/{name}'] = float(hard.get(name, 0.0))
    for name in soft_signal_names:
        vector[f'soft/{name}'] = float(soft.get(name, 0.0))
    return vector


def compute_grpo_advantages(eval_rows: Sequence[dict[str, Any]], task_json: Sequence[str]) -> list[float]:
    rewards = [float(row.get('combined_quality', 0.0)) for row in eval_rows]
    advantages = [0.0 for _ in rewards]
    for group in group_indices_by_task(task_json):
        normalized = normalize_zscore([rewards[i] for i in group], divide_by_std=True)
        for idx, value in zip(group, normalized):
            advantages[idx] = value
    return advantages


def compute_gdpo_advantages(
    eval_rows: Sequence[dict[str, Any]],
    task_json: Sequence[str],
    *,
    hard_signal_names: Sequence[str],
    soft_signal_names: Sequence[str],
    final_batch_normalize: bool = True,
) -> list[float]:
    """Approximate GDPO normalize-then-sum advantages for diagnostics.

    This treats each hard check and each soft score as a separate reward signal
    and normalizes each signal within a prompt group before summing. It does not
    preserve a hard/soft distinction, which is the intended GDPO baseline.
    """
    n = len(eval_rows)
    advantages = [0.0 for _ in range(n)]
    signal_names = [f'hard/{name}' for name in hard_signal_names] + [f'soft/{name}' for name in soft_signal_names]
    vectors = [extract_signal_vector(row, hard_signal_names=hard_signal_names, soft_signal_names=soft_signal_names) for row in eval_rows]

    for group in group_indices_by_task(task_json):
        group_scores = [0.0 for _ in group]
        for signal in signal_names:
            values = [vectors[i].get(signal, 0.0) for i in group]
            normalized = normalize_zscore(values, divide_by_std=True)
            group_scores = [score + val for score, val in zip(group_scores, normalized)]
        for idx, value in zip(group, group_scores):
            advantages[idx] = value

    if final_batch_normalize:
        advantages = batch_normalize(advantages)
    return advantages


def compute_cdpo_advantages(
    eval_rows: Sequence[dict[str, Any]],
    task_json: Sequence[str],
    *,
    hard_signal_names: Sequence[str],
    soft_signal_names: Sequence[str],
    alpha: float,
    final_batch_normalize: bool = True,
) -> tuple[list[float], list[float], list[float]]:
    """Compute CDPO-style diagnostic advantages.

    Hard constraints are independently mean-centered within the group without
    std division, then summed into a hard channel. Soft preferences are z-scored
    per signal and summed into a soft channel. The channels are combined with
    alpha * hard + (1-alpha) * soft.
    """
    n = len(eval_rows)
    hard_adv = [0.0 for _ in range(n)]
    soft_adv = [0.0 for _ in range(n)]
    combined = [0.0 for _ in range(n)]
    alpha = max(0.0, min(1.0, float(alpha)))

    hard_vectors = [row.get('hard_checks', {}) or {} for row in eval_rows]
    soft_vectors = [row.get('soft_scores', {}) or {} for row in eval_rows]

    for group in group_indices_by_task(task_json):
        group_hard = [0.0 for _ in group]
        group_soft = [0.0 for _ in group]

        for name in hard_signal_names:
            values = [float(hard_vectors[i].get(name, 0.0)) for i in group]
            normalized = normalize_zscore(values, divide_by_std=False)
            group_hard = [score + val for score, val in zip(group_hard, normalized)]

        for name in soft_signal_names:
            values = [float(soft_vectors[i].get(name, 0.0)) for i in group]
            normalized = normalize_zscore(values, divide_by_std=True)
            group_soft = [score + val for score, val in zip(group_soft, normalized)]

        for idx, h_val, s_val in zip(group, group_hard, group_soft):
            hard_adv[idx] = h_val
            soft_adv[idx] = s_val
            combined[idx] = alpha * h_val + (1.0 - alpha) * s_val

    if final_batch_normalize:
        combined = batch_normalize(combined)
    return hard_adv, soft_adv, combined


def summarize_collapse_by_pattern(
    eval_rows: Sequence[dict[str, Any]],
    advantages: Sequence[float],
    *,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Summarize whether distinct constraint patterns map to identical advantages."""
    bucket_to_patterns: dict[float, set[str]] = defaultdict(set)
    pattern_to_advantages: dict[str, set[float]] = defaultdict(set)

    for row, adv in zip(eval_rows, advantages):
        bucket = round(float(adv) / tol) * tol
        pattern = violation_pattern(row.get('hard_checks', {}) or {})
        bucket_to_patterns[bucket].add(pattern)
        pattern_to_advantages[pattern].add(bucket)

    collapsed_buckets = {
        str(bucket): sorted(patterns)
        for bucket, patterns in bucket_to_patterns.items()
        if len(patterns) > 1
    }
    return {
        'distinct_patterns': len(pattern_to_advantages),
        'distinct_advantage_values': len(bucket_to_patterns),
        'collapsed_advantage_buckets': len(collapsed_buckets),
        'collapsed_bucket_examples': dict(list(collapsed_buckets.items())[:10]),
    }
