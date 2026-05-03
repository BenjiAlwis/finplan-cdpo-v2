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

from finplan.rl.reward_wrapper import FinPlanRewardWrapper, build_gdpo_reward_funcs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/gdpo_baseline.yaml')
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f'Config file must contain a YAML mapping: {path}')
    return cfg


def _load_json_dataset(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(
            f'Dataset file not found: {path}. '
            'Run `python scripts/build_training_prompts.py` first if needed.'
        )
    return load_dataset('json', data_files=path, split='train')


def _validate_generation_batch(cfg: dict[str, Any]) -> None:
    num_processes = int(cfg.get('num_processes', 1))
    per_device_train_batch_size = int(cfg.get('per_device_train_batch_size', 1))
    gradient_accumulation_steps = int(cfg.get('gradient_accumulation_steps', 1))
    num_generations = int(cfg.get('num_generations', 8))

    effective_batch = num_processes * per_device_train_batch_size * gradient_accumulation_steps
    if effective_batch % num_generations != 0:
        raise ValueError(
            'Invalid GRPO/GDPO batch configuration: '
            f'num_processes({num_processes}) * '
            f'per_device_train_batch_size({per_device_train_batch_size}) * '
            f'gradient_accumulation_steps({gradient_accumulation_steps}) = {effective_batch}, '
            f'which is not divisible by num_generations({num_generations}). '
            'For a single-GPU smoke test, try per_device_train_batch_size=2 and num_generations=2. '
            'For a larger run, try per_device_train_batch_size=4 and num_generations=4, '
            'or per_device_train_batch_size=8 and num_generations=8 if memory allows.'
        )


def _build_grpo_config_kwargs(cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    requested: dict[str, Any] = {
        'output_dir': str(output_dir),
        'run_name': cfg.get('run_name', 'gdpo_baseline'),
        'learning_rate': float(cfg.get('learning_rate', 1e-6)),
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
        'beta': float(cfg.get('beta', 0.04)),
        'scale_rewards': cfg.get('scale_rewards', 'group'),
        'multi_objective_aggregation': cfg.get(
            'multi_objective_aggregation',
            'normalize_then_sum',
        ),
        'reward_weights': cfg.get('reward_weights'),
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

    # Do not pass None for reward_weights unless this TRL version explicitly accepts it.
    if requested['reward_weights'] is None:
        requested.pop('reward_weights')

    signature = inspect.signature(GRPOConfig.__init__)
    supported = set(signature.parameters)

    if 'eval_strategy' not in supported and 'evaluation_strategy' in supported:
        requested['evaluation_strategy'] = requested.pop('eval_strategy')

    filtered = {key: value for key, value in requested.items() if key in supported}
    dropped = sorted(set(requested) - set(filtered))
    if dropped:
        print(
            '[run_gdpo_baseline] Warning: ignoring GRPOConfig keys not supported '
            f'by this installed TRL version: {dropped}'
        )
    return filtered


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    _validate_generation_batch(cfg)

    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = _load_json_dataset(cfg['train_path'])
    eval_dataset = _load_json_dataset(cfg['eval_path']) if cfg.get('eval_path') else None

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    wrapper = FinPlanRewardWrapper()
    reward_funcs = build_gdpo_reward_funcs(wrapper)

    training_args = GRPOConfig(**_build_grpo_config_kwargs(cfg, output_dir))

    trainer = GRPOTrainer(
        model=cfg['model_name'],
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_model(str(output_dir / 'final_model'))

    metadata = {
        'config': cfg,
        'train_rows': len(train_dataset),
        'eval_rows': len(eval_dataset) if eval_dataset is not None else 0,
        'reward_functions': [getattr(fn, '__name__', str(fn)) for fn in reward_funcs],
    }
    with open(output_dir / 'run_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
