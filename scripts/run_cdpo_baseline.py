from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from finplan.rl.reward_wrapper import FinPlanRewardWrapper, build_cdpo_reward_func


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cdpo_cloud_gpu_fixed.yaml')
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _load_json_dataset(path: str):
    return load_dataset('json', data_files=path, split='train')


def _filter_grpo_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    valid = set(inspect.signature(GRPOConfig).parameters)
    return {key: value for key, value in kwargs.items() if key in valid}


def _validate_batch_shape(cfg: dict[str, Any]) -> None:
    effective = int(cfg.get('per_device_train_batch_size', 1)) * int(cfg.get('gradient_accumulation_steps', 1))
    num_generations = int(cfg.get('num_generations', 1))
    if effective % num_generations != 0:
        raise ValueError(
            'For single-process CDPO, per_device_train_batch_size * gradient_accumulation_steps '
            f'must be divisible by num_generations. Got {effective=} and {num_generations=}.'
        )


def build_training_args(cfg: dict[str, Any], output_dir: Path) -> GRPOConfig:
    _validate_batch_shape(cfg)
    raw_kwargs: dict[str, Any] = {
        'output_dir': str(output_dir),
        'run_name': cfg.get('run_name', 'cdpo_baseline'),
        'learning_rate': float(cfg.get('learning_rate', 1e-6)),
        'optim': cfg.get('optim', 'adamw_torch'),
        'per_device_train_batch_size': int(cfg.get('per_device_train_batch_size', 1)),
        'per_device_eval_batch_size': int(cfg.get('per_device_eval_batch_size', 1)),
        'gradient_accumulation_steps': int(cfg.get('gradient_accumulation_steps', 1)),
        'num_generations': int(cfg.get('num_generations', 8)),
        'generation_batch_size': int(cfg.get('generation_batch_size', cfg.get('num_generations', 8))),
        'max_prompt_length': int(cfg.get('max_prompt_length', 1024)),
        'max_completion_length': int(cfg.get('max_completion_length', 256)),
        'temperature': float(cfg.get('temperature', 1.0)),
        'top_p': float(cfg.get('top_p', 1.0)),
        'top_k': int(cfg.get('top_k', 0)),
        'num_train_epochs': float(cfg.get('num_train_epochs', 1.0)),
        'max_steps': int(cfg.get('max_steps', -1)),
        'logging_steps': int(cfg.get('logging_steps', 1)),
        'eval_strategy': cfg.get('eval_strategy', 'steps'),
        'eval_steps': int(cfg.get('eval_steps', 50)),
        'save_steps': int(cfg.get('save_steps', 50)),
        'save_total_limit': int(cfg.get('save_total_limit', 2)),
        'save_strategy': cfg.get('save_strategy', 'steps'),
        'save_only_model': bool(cfg.get('save_only_model', False)),
        'beta': float(cfg.get('beta', 0.04)),
        # CDPO returns a pre-normalized scalar; keep TRL's group normalization on
        # to stabilize gradient magnitudes, matching the final normalization step.
        'scale_rewards': cfg.get('scale_rewards', 'group'),
        'report_to': cfg.get('report_to', 'tensorboard'),
        'remove_unused_columns': bool(cfg.get('remove_unused_columns', False)),
        'log_completions': bool(cfg.get('log_completions', False)),
        'log_unique_prompts': bool(cfg.get('log_unique_prompts', False)),
        'seed': int(cfg.get('seed', 42)),
        'use_cpu': bool(cfg.get('use_cpu', False)),
        'bf16': bool(cfg.get('bf16', False)),
        'fp16': bool(cfg.get('fp16', False)),
        'gradient_checkpointing': bool(cfg.get('gradient_checkpointing', False)),
    }
    return GRPOConfig(**_filter_grpo_kwargs(raw_kwargs))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = _load_json_dataset(cfg['train_path'])
    eval_dataset = _load_json_dataset(cfg['eval_path']) if cfg.get('eval_path') else None

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    diagnostics_path = cfg.get('diagnostics_path', str(output_dir / 'diagnostics.jsonl'))
    wrapper = FinPlanRewardWrapper(diagnostics_path=diagnostics_path, method='cdpo')
    reward_func = build_cdpo_reward_func(
        wrapper,
        alpha_mode=cfg.get('alpha_mode', 'fixed'),
        fixed_alpha=float(cfg.get('fixed_alpha', 0.7)),
        start_alpha=float(cfg.get('start_alpha', 0.9)),
        end_alpha=float(cfg.get('end_alpha', 0.5)),
        learned_min_alpha=float(cfg.get('learned_min_alpha', 0.35)),
        learned_max_alpha=float(cfg.get('learned_max_alpha', 0.9)),
        max_steps=int(cfg.get('max_steps', 200)),
        final_batch_normalize=bool(cfg.get('cdpo_final_batch_normalize', True)),
    )

    training_args = build_training_args(cfg, output_dir)

    trainer = GRPOTrainer(
        model=cfg['model_name'],
        processing_class=tokenizer,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    if bool(cfg.get('save_final_model', True)):
        trainer.save_model(str(output_dir / 'final_model'))
    else:
        print('[runner] Skipping final model save because save_final_model=false', flush=True)

    metadata = {
        'method': 'cdpo',
        'config': cfg,
        'train_rows': len(train_dataset),
        'eval_rows': len(eval_dataset) if eval_dataset is not None else 0,
        'diagnostics_path': diagnostics_path,
        'alpha_mode': cfg.get('alpha_mode', 'fixed'),
    }
    with open(output_dir / 'run_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
