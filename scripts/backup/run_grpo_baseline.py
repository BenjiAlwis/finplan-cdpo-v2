from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from finplan.rl.reward_wrapper import FinPlanRewardWrapper, build_monolithic_reward_func


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/grpo_baseline.yaml')
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _load_json_dataset(path: str):
    return load_dataset('json', data_files=path, split='train')


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = _load_json_dataset(cfg['train_path'])
    eval_dataset = _load_json_dataset(cfg['eval_path']) if cfg.get('eval_path') else None

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    wrapper = FinPlanRewardWrapper()
    reward_func = build_monolithic_reward_func(wrapper)

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        run_name=cfg.get('run_name', 'grpo_baseline'),
        learning_rate=float(cfg.get('learning_rate', 1e-6)),
        per_device_train_batch_size=int(cfg.get('per_device_train_batch_size', 1)),
        per_device_eval_batch_size=int(cfg.get('per_device_eval_batch_size', 1)),
        gradient_accumulation_steps=int(cfg.get('gradient_accumulation_steps', 1)),
        num_generations=int(cfg.get('num_generations', 8)),
        max_completion_length=int(cfg.get('max_completion_length', 256)),
        num_train_epochs=float(cfg.get('num_train_epochs', 1.0)),
        max_steps=int(cfg.get('max_steps', -1)),
        logging_steps=int(cfg.get('logging_steps', 1)),
        eval_strategy=cfg.get('eval_strategy', 'steps'),
        eval_steps=int(cfg.get('eval_steps', 50)),
        save_steps=int(cfg.get('save_steps', 50)),
        save_total_limit=int(cfg.get('save_total_limit', 2)),
        beta=float(cfg.get('beta', 0.04)),
        scale_rewards=cfg.get('scale_rewards', 'group'),
        report_to=cfg.get('report_to', 'tensorboard'),
        remove_unused_columns=bool(cfg.get('remove_unused_columns', False)),
        log_completions=bool(cfg.get('log_completions', False)),
        log_unique_prompts=bool(cfg.get('log_unique_prompts', False)),
        seed=int(cfg.get('seed', 42)),
    )

    trainer = GRPOTrainer(
        model=cfg['model_name'],
        processing_class=tokenizer,
        reward_funcs=reward_func,
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
    }
    with open(output_dir / 'run_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
