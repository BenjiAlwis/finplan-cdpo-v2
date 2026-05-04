from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Sequence

from finplan.env.finplan_env import FinPlanEnv
from finplan.types import TaskInstance

from .advantage_utils import compute_cdpo_advantages
from .diagnostics import (
    DEFAULT_HARD_SIGNAL_NAMES,
    DEFAULT_SOFT_SIGNAL_NAMES,
    JsonlDiagnosticsLogger,
)
from .logging_hooks import (
    count_distinct_advantage_groups,
    maybe_log_extra,
    maybe_log_metric,
    summarize_batch,
    summarize_constraint_failures,
    summarize_soft_scores,
)


HARD_SIGNAL_NAMES = DEFAULT_HARD_SIGNAL_NAMES
SOFT_SIGNAL_NAMES = DEFAULT_SOFT_SIGNAL_NAMES


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


def _extract_first_json_object(text: str) -> str:
    """Return the first balanced JSON object found in text.

    Early RL completions often include prose or markdown fences around JSON.
    For reward computation, evaluate the first syntactically complete object.
    If no balanced object exists, return the original text so parser errors are
    still informative.
    """
    start = text.find('{')
    if start < 0:
        return text

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]

        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return text


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return _extract_first_json_object(completion)
    if isinstance(completion, dict):
        if 'content' in completion and isinstance(completion['content'], str):
            return _extract_first_json_object(completion['content'])
        if 'text' in completion and isinstance(completion['text'], str):
            return _extract_first_json_object(completion['text'])
        if 'messages' in completion:
            return _completion_to_text(completion['messages'])
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict) and isinstance(item.get('content'), str):
                parts.append(item['content'])
            else:
                parts.append(str(item))
        return _extract_first_json_object('\n'.join(parts))
    return _extract_first_json_object(str(completion))


def _safe_mean(values: Sequence[float]) -> float:
    return sum(float(v) for v in values) / max(len(values), 1)


class FinPlanRewardWrapper:
    """Converts TRL completions into FinPlanEnv evaluations and scalar rewards.

    The wrapper caches evaluations because multiple reward functions may be called
    on the same batch of completions during GDPO-style training.
    """

    def __init__(
        self,
        *,
        diagnostics_path: str | None = None,
        method: str = 'unknown',
        hard_signal_names: Sequence[str] = HARD_SIGNAL_NAMES,
        soft_signal_names: Sequence[str] = SOFT_SIGNAL_NAMES,
    ) -> None:
        self.env = FinPlanEnv()
        self._cache: dict[str, dict[str, Any]] = {}
        self.hard_signal_names = list(hard_signal_names)
        self.soft_signal_names = list(soft_signal_names)
        self.diagnostics = JsonlDiagnosticsLogger(diagnostics_path, method=method)

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

    def log_batch_diagnostics(
        self,
        eval_rows: list[dict[str, Any]],
        kwargs: dict[str, Any],
        *,
        task_json: Sequence[str] | None = None,
        reward_values: Sequence[float] | None = None,
        reward_name: str = 'reward',
        cdpo_alpha: float | None = None,
        write_plan_rows: bool = True,
    ) -> None:
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

        if task_json is not None and reward_values is not None:
            self.diagnostics.log_batch(
                eval_rows=eval_rows,
                task_json=task_json,
                reward_values=reward_values,
                reward_name=reward_name,
                hard_signal_names=self.hard_signal_names,
                soft_signal_names=self.soft_signal_names,
                cdpo_alpha=cdpo_alpha,
                write_plan_rows=write_plan_rows,
            )


def build_monolithic_reward_func(wrapper: FinPlanRewardWrapper) -> Callable[..., list[float]]:
    """GRPO baseline: monolithic scalar reward with parse cold-start shaping.

    The reported combined_quality metric remains the strict hard-gated quality.
    The training reward adds a small parse/schema component so early training is
    not completely zero when the model emits prose or markdown instead of JSON.
    """

    def reward_func(completions: list[Any], task_json: list[str], **kwargs: Any) -> list[float]:
        eval_rows = wrapper.evaluate_batch(completions=completions, task_json=task_json)
        rewards = [
            0.2 * float(bool(row.get('parse_success', False)))
            + 0.8 * float(row.get('combined_quality', 0.0))
            for row in eval_rows
        ]
        wrapper.log_batch_diagnostics(
            eval_rows,
            kwargs,
            task_json=task_json,
            reward_values=rewards,
            reward_name='monolithic_parse_shaped_combined_quality',
        )
        return rewards

    reward_func.__name__ = 'monolithic_parse_shaped_combined_quality_reward_func'
    return reward_func


def _make_scalar_reward_func(
    wrapper: FinPlanRewardWrapper,
    *,
    name: str,
    extractor: Callable[[dict[str, Any]], float],
    log_rows: bool = False,
) -> Callable[..., list[float]]:
    def reward_func(completions: list[Any], task_json: list[str], **kwargs: Any) -> list[float]:
        eval_rows = wrapper.evaluate_batch(completions=completions, task_json=task_json)
        values = [float(extractor(row)) for row in eval_rows]
        maybe_log_metric(kwargs, f'reward_channel_mean/{name}', _safe_mean(values))
        maybe_log_metric(kwargs, f'reward_channel_groups/{name}', count_distinct_advantage_groups(values))
        wrapper.log_batch_diagnostics(
            eval_rows,
            kwargs,
            task_json=task_json,
            reward_values=values,
            reward_name=name,
            write_plan_rows=log_rows,
        )
        return values

    reward_func.__name__ = f'{name}_reward_func'
    return reward_func


def build_gdpo_reward_funcs(wrapper: FinPlanRewardWrapper) -> list[Callable[..., list[float]]]:
    """GDPO baseline: per-reward-signal functions, no hard/soft channel distinction.

    Each individual hard check and each individual soft score is exposed as its
    own reward function. TRL's normalize_then_sum aggregation then normalizes
    each signal independently before summing. Missing signals for another domain
    return 0.0, which is neutral within prompt groups where every completion has
    the same domain.
    """
    funcs: list[Callable[..., list[float]]] = []
    for signal_name in wrapper.hard_signal_names:
        funcs.append(
            _make_scalar_reward_func(
                wrapper,
                name=f'hard_{signal_name}',
                extractor=lambda row, key=signal_name: float((row.get('hard_checks', {}) or {}).get(key, 0.0)),
                log_rows=False,
            )
        )
    for signal_name in wrapper.soft_signal_names:
        funcs.append(
            _make_scalar_reward_func(
                wrapper,
                name=f'soft_{signal_name}',
                extractor=lambda row, key=signal_name: float((row.get('soft_scores', {}) or {}).get(key, 0.0)),
                log_rows=False,
            )
        )
    return funcs


def _alpha_for_step(
    *,
    mode: str,
    call_index: int,
    max_steps: int,
    fixed_alpha: float,
    start_alpha: float,
    end_alpha: float,
    learned_min_alpha: float,
    learned_max_alpha: float,
    latest_hard_pass_rate: float,
) -> float:
    if mode == 'fixed':
        return fixed_alpha
    if mode == 'anneal':
        denom = max(max_steps - 1, 1)
        frac = min(max(call_index / denom, 0.0), 1.0)
        return start_alpha + frac * (end_alpha - start_alpha)
    if mode == 'learned':
        # Lightweight adaptive proxy for a learned/meta-gradient alpha. When hard
        # compliance is low, put more mass on the hard channel; as compliance
        # improves, shift toward soft preference optimization.
        gap = 1.0 - min(max(latest_hard_pass_rate, 0.0), 1.0)
        return learned_min_alpha + gap * (learned_max_alpha - learned_min_alpha)
    raise ValueError(f'Unknown CDPO alpha mode: {mode}')


def build_cdpo_reward_func(
    wrapper: FinPlanRewardWrapper,
    *,
    alpha_mode: str = 'fixed',
    fixed_alpha: float = 0.7,
    start_alpha: float = 0.9,
    end_alpha: float = 0.5,
    learned_min_alpha: float = 0.35,
    learned_max_alpha: float = 0.9,
    max_steps: int = 200,
    final_batch_normalize: bool = True,
) -> Callable[..., list[float]]:
    """CDPO reward function with hard/soft channel-normalized rewards.

    This is a practical TRL-compatible implementation: it computes CDPO-style
    normalized channel rewards inside the reward function and returns the final
    scalar for GRPOTrainer. Diagnostics log the hard channel, soft channel, alpha,
    and final CDPO diagnostic advantage values.
    """
    state = {'call_index': 0, 'latest_hard_pass_rate': 0.0}

    def reward_func(completions: list[Any], task_json: list[str], **kwargs: Any) -> list[float]:
        eval_rows = wrapper.evaluate_batch(completions=completions, task_json=task_json)
        state['call_index'] += 1
        if eval_rows:
            state['latest_hard_pass_rate'] = sum(float(row.get('all_constraints_pass', 0.0)) for row in eval_rows) / len(eval_rows)

        alpha = _alpha_for_step(
            mode=alpha_mode,
            call_index=state['call_index'],
            max_steps=max_steps,
            fixed_alpha=fixed_alpha,
            start_alpha=start_alpha,
            end_alpha=end_alpha,
            learned_min_alpha=learned_min_alpha,
            learned_max_alpha=learned_max_alpha,
            latest_hard_pass_rate=state['latest_hard_pass_rate'],
        )
        hard_adv, soft_adv, cdpo_adv = compute_cdpo_advantages(
            eval_rows,
            task_json,
            hard_signal_names=wrapper.hard_signal_names,
            soft_signal_names=wrapper.soft_signal_names,
            alpha=alpha,
            final_batch_normalize=final_batch_normalize,
        )
        rewards = [float(v) for v in cdpo_adv]

        maybe_log_metric(kwargs, 'cdpo/alpha', alpha)
        maybe_log_metric(kwargs, 'cdpo/hard_channel_mean', _safe_mean(hard_adv))
        maybe_log_metric(kwargs, 'cdpo/soft_channel_mean', _safe_mean(soft_adv))
        maybe_log_metric(kwargs, 'cdpo/reward_mean', _safe_mean(rewards))
        maybe_log_metric(kwargs, 'cdpo/hard_pass_rate', state['latest_hard_pass_rate'])

        wrapper.log_batch_diagnostics(
            eval_rows,
            kwargs,
            task_json=task_json,
            reward_values=rewards,
            reward_name=f'cdpo_{alpha_mode}',
            cdpo_alpha=alpha,
        )
        return rewards

    reward_func.__name__ = f'cdpo_{alpha_mode}_reward_func'
    return reward_func


def serialize_task_for_dataset(task: TaskInstance) -> str:
    return _serialize_task(task)
