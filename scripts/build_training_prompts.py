from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from finplan.prompts.prompt_builder import build_prompt
from finplan.rl.reward_wrapper import serialize_task_for_dataset
from finplan.types import TaskInstance
from finplan.utils.io import read_jsonl, write_jsonl


RAW_FILES = [
    ("portfolio", "data/raw/portfolio_instances.jsonl"),
    ("retirement", "data/raw/retirement_instances.jsonl"),
    ("loan", "data/raw/loan_instances.jsonl"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--output-dir', type=str, default='data/training')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for domain, path in RAW_FILES:
        for row in read_jsonl(path):
            task = TaskInstance(**row)
            rows.append(
                {
                    'task_id': task.task_id,
                    'domain': task.domain,
                    'difficulty': (task.metadata or {}).get('difficulty', ''),
                    'prompt': build_prompt(task),
                    'task_json': serialize_task_for_dataset(task),
                }
            )

    random.shuffle(rows)
    split_idx = int(len(rows) * args.train_ratio)
    train_rows = rows[:split_idx]
    eval_rows = rows[split_idx:]

    write_jsonl(str(output_dir / 'train_prompts.jsonl'), train_rows)
    write_jsonl(str(output_dir / 'eval_prompts.jsonl'), eval_rows)

    print(f'Wrote {len(train_rows)} train prompts to {output_dir / "train_prompts.jsonl"}')
    print(f'Wrote {len(eval_rows)} eval prompts to {output_dir / "eval_prompts.jsonl"}')


if __name__ == '__main__':
    main()
