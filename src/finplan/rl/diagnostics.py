from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

from finplan.rl.advantage_utils import (
    compute_cdpo_advantages,
    compute_gdpo_advantages,
    compute_grpo_advantages,
    summarize_collapse_by_pattern,
    summarize_values,
    violation_pattern,
)


DEFAULT_HARD_SIGNAL_NAMES = [
    'parse_valid',
    'weights_sum_valid',
    'banned_sector_valid',
    'diversification_valid',
    'drawdown_valid',
    'no_early_depletion',
    'income_floor_valid',
    'inflation_handling_valid',
    'dti_valid',
    'ltv_valid',
    'regulatory_valid',
]

DEFAULT_SOFT_SIGNAL_NAMES = [
    'risk_alignment',
    'esg_alignment',
    'tax_efficiency',
    'risk_adjusted_return',
    'lifestyle_quality',
    'bequest_alignment',
    'withdrawal_smoothness',
    'interest_cost_score',
    'payment_flexibility',
    'prepayment_optionality',
]


class JsonlDiagnosticsLogger:
    """Append-only JSONL logger for reward/advantage diagnostics.

    TensorBoard is useful for curves, but the research claim needs raw rows that
    can be audited later. This logger writes both batch summaries and per-plan
    records, including constraint patterns and diagnostic advantages.
    """

    def __init__(self, path: str | Path | None, *, method: str) -> None:
        self.path = Path(path) if path else None
        self.method = method
        self.call_index = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, payload: dict[str, Any]) -> None:
        if self.path is None:
            return
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + '\n')

    def log_batch(
        self,
        *,
        eval_rows: Sequence[dict[str, Any]],
        task_json: Sequence[str],
        reward_values: Sequence[float],
        reward_name: str,
        hard_signal_names: Sequence[str] = DEFAULT_HARD_SIGNAL_NAMES,
        soft_signal_names: Sequence[str] = DEFAULT_SOFT_SIGNAL_NAMES,
        cdpo_alpha: float | None = None,
        write_plan_rows: bool = True,
    ) -> None:
        self.call_index += 1
        timestamp = time.time()

        grpo_adv = compute_grpo_advantages(eval_rows, task_json)
        gdpo_adv = compute_gdpo_advantages(
            eval_rows,
            task_json,
            hard_signal_names=hard_signal_names,
            soft_signal_names=soft_signal_names,
        )
        hard_adv: list[float] | None = None
        soft_adv: list[float] | None = None
        cdpo_adv: list[float] | None = None
        if cdpo_alpha is not None:
            hard_adv, soft_adv, cdpo_adv = compute_cdpo_advantages(
                eval_rows,
                task_json,
                hard_signal_names=hard_signal_names,
                soft_signal_names=soft_signal_names,
                alpha=cdpo_alpha,
            )

        n = max(len(eval_rows), 1)
        summary = {
            'record_type': 'batch_summary',
            'method': self.method,
            'reward_name': reward_name,
            'call_index': self.call_index,
            'timestamp': timestamp,
            'batch_size': len(eval_rows),
            'hard_pass_rate': sum(float(row.get('all_constraints_pass', 0.0)) for row in eval_rows) / n,
            'parse_success_rate': sum(float(bool(row.get('parse_success', False))) for row in eval_rows) / n,
            'mean_soft_score': sum(float(row.get('soft_mean_score', 0.0)) for row in eval_rows) / n,
            'mean_combined_quality': sum(float(row.get('combined_quality', 0.0)) for row in eval_rows) / n,
            'reward_mean': sum(float(v) for v in reward_values) / max(len(reward_values), 1),
            'reward_distinct_groups': summarize_values(list(reward_values)).distinct_groups,
            'grpo_advantage_groups': summarize_values(grpo_adv).distinct_groups,
            'gdpo_diagnostic_advantage_groups': summarize_values(gdpo_adv).distinct_groups,
            'grpo_signal_collapse': summarize_collapse_by_pattern(eval_rows, grpo_adv),
            'gdpo_signal_collapse': summarize_collapse_by_pattern(eval_rows, gdpo_adv),
        }
        if cdpo_alpha is not None and cdpo_adv is not None and hard_adv is not None and soft_adv is not None:
            summary.update(
                {
                    'cdpo_alpha': float(cdpo_alpha),
                    'cdpo_hard_advantage_groups': summarize_values(hard_adv).distinct_groups,
                    'cdpo_soft_advantage_groups': summarize_values(soft_adv).distinct_groups,
                    'cdpo_advantage_groups': summarize_values(cdpo_adv).distinct_groups,
                    'cdpo_signal_collapse': summarize_collapse_by_pattern(eval_rows, cdpo_adv),
                }
            )
        self._write(summary)

        if not write_plan_rows:
            return

        for i, row in enumerate(eval_rows):
            payload: dict[str, Any] = {
                'record_type': 'plan_eval',
                'method': self.method,
                'reward_name': reward_name,
                'call_index': self.call_index,
                'timestamp': timestamp,
                'row_index': i,
                'task_id': row.get('task_id'),
                'domain': row.get('domain'),
                'parse_success': row.get('parse_success'),
                'hard_checks': row.get('hard_checks', {}),
                'soft_scores': row.get('soft_scores', {}),
                'all_constraints_pass': row.get('all_constraints_pass'),
                'soft_mean_score': row.get('soft_mean_score'),
                'combined_quality': row.get('combined_quality'),
                'violated_constraints': row.get('violated_constraints', []),
                'violation_pattern': violation_pattern(row.get('hard_checks', {}) or {}),
                'reward_value': float(reward_values[i]) if i < len(reward_values) else None,
                'grpo_diagnostic_advantage': grpo_adv[i] if i < len(grpo_adv) else None,
                'gdpo_diagnostic_advantage': gdpo_adv[i] if i < len(gdpo_adv) else None,
            }
            if cdpo_adv is not None and hard_adv is not None and soft_adv is not None:
                payload.update(
                    {
                        'cdpo_alpha': cdpo_alpha,
                        'cdpo_hard_advantage': hard_adv[i],
                        'cdpo_soft_advantage': soft_adv[i],
                        'cdpo_diagnostic_advantage': cdpo_adv[i],
                    }
                )
            self._write(payload)
