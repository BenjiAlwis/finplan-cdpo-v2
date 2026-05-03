from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Callable

from finplan.env.finplan_env import FinPlanEnv
from finplan.types import TaskInstance

from .logging_hooks import (
    count_distinct_advantage_groups,
    maybe_log_extra,
    maybe_log_metric,
    summarize_batch,
    summarize_constraint_failures,
    summarize_soft_scores,
)


def _to_task_instance(task_payload: dict[str, Any] | str) -> TaskInstance:
    if isinstance(task_payload, str):
        task_payload = json.loads(task_payload)
    return TaskInstance(**task_payload)


def _serialize_task(task: TaskInstance) -> str:
    if hasattr(task, 'to_dict'):
        payload = task.to_dict()
    elif is_dataclass(task):
        payload = asdict(task)
    else:
        payload = dict(task)
    return json.dumps(payload, ensure_ascii=False)


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if 'content' in completion and isinstance(completion['content'], str):
            return completion['content']
        if 'text' in completion and isinstance(completion['text'], str):
            return completion['text']
        if 'messages' in completion:
            return _completion_to_text(completion['messages'])
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict) and isinstance(item.get('content'), str):
                parts.append(item['content'])
            else:
                parts.append(str(item))
        return '\n'.join(parts)
    return str(completion)


class FinPlanRewardWrapper:
    """Converts TRL completions into FinPlanEnv evaluations and scalar rewards.

    The wrapper caches evaluations because multiple reward functions may be called on
    the same batch of completions during GDPO-style training.
    """

    def __init__(self) -> None:
        self.env = FinPlanEnv()
        self._cache: dict[str, dict[str, Any]] = {}

    def _cache_key(self, task_json: str, completion_text: str) -> str:
        digest = hashlib.sha256()
        digest.update(task_json.encode('utf-8'))
        digest.update(b'\0')
        digest.update(completion_text.encode('utf-8'))
        return digest.hexdigest()

    def evaluate_completion(self, task_json: str, completion_text: str) -> dict[str, Any]:
        key = self._cache_key(task_json, completion_text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        task = _to_task_instance(task_json)
        parsed, reward = self.env.evaluate(task, completion_text)

        result = {
            'task_id': task.task_id,
            'domain': task.domain,
            'parse_success': bool(parsed.parse_success),
            'parse_error': parsed.parse_error,
            'hard_checks': dict(reward.hard.checks),
            'all_constraints_pass': int(reward.hard.all_pass),
            'soft_scores': dict(reward.soft.scores),
            'soft_mean_score': float(reward.soft.mean_score),
            'combined_quality': float(reward.combined_quality),
            'violated_constraints': list(reward.metadata.get('violated_constraints', [])),
            'raw_plan': completion_text,
        }
        self._cache[key] = result
        return result

    def evaluate_batch(
        self,
        completions: list[Any],
        task_json: list[str] | tuple[str, ...],
    ) -> list[dict[str, Any]]:
        if len(completions) != len(task_json):
            raise ValueError('completions and task_json must have the same length')
        return [
            self.evaluate_completion(task_str, _completion_to_text(completion))
            for completion, task_str in zip(completions, task_json)
        ]

    def log_batch_diagnostics(self, eval_rows: list[dict[str, Any]], kwargs: dict[str, Any]) -> None:
        summary = summarize_batch(eval_rows)
        for key, value in summary.items():
            maybe_log_metric(kwargs, key, value)

        for key, value in summarize_constraint_failures(eval_rows).items():
            maybe_log_metric(kwargs, key, value)

        for key, value in summarize_soft_scores(eval_rows).items():
            maybe_log_metric(kwargs, key, value)

        combined = [float(row.get('combined_quality', 0.0)) for row in eval_rows]
        maybe_log_metric(kwargs, 'advantage_groups/combined_quality', count_distinct_advantage_groups(combined))
        maybe_log_extra(kwargs, 'eval_rows_preview', eval_rows[:3])


def build_monolithic_reward_func(wrapper: FinPlanRewardWrapper) -> Callable[..., list[float]]:
    def reward_func(completions: list[Any], task_json: list[str], **kwargs: Any) -> list[float]:
        eval_rows = wrapper.evaluate_batch(completions=completions, task_json=task_json)
        wrapper.log_batch_diagnostics(eval_rows, kwargs)
        return [float(row['combined_quality']) for row in eval_rows]

    return reward_func


def _make_scalar_reward_func(
    wrapper: FinPlanRewardWrapper,
    *,
    name: str,
    extractor: Callable[[dict[str, Any]], float],
) -> Callable[..., list[float]]:
    def reward_func(completions: list[Any], task_json: list[str], **kwargs: Any) -> list[float]:
        eval_rows = wrapper.evaluate_batch(completions=completions, task_json=task_json)
        values = [float(extractor(row)) for row in eval_rows]
        maybe_log_metric(kwargs, f'reward_channel_mean/{name}', sum(values) / max(len(values), 1))
        maybe_log_metric(kwargs, f'reward_channel_groups/{name}', count_distinct_advantage_groups(values))
        return values

    reward_func.__name__ = f'{name}_reward_func'
    return reward_func


def build_gdpo_reward_funcs(wrapper: FinPlanRewardWrapper) -> list[Callable[..., list[float]]]:
    """Returns a compact reward-function set for the GDPO baseline.

    Week 2 only needs multiple reward signals and normalize-then-sum aggregation,
    so this starter uses two channels:
    - hard feasibility (all constraints pass)
    - soft preference quality (mean soft score)
    """
    return [
        _make_scalar_reward_func(
            wrapper,
            name='hard_feasibility',
            extractor=lambda row: float(row['all_constraints_pass']),
        ),
        _make_scalar_reward_func(
            wrapper,
            name='soft_mean_score',
            extractor=lambda row: float(row['soft_mean_score']),
        ),
    ]


def serialize_task_for_dataset(task: TaskInstance) -> str:
    return _serialize_task(task)
