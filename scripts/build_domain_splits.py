from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def get_domain(row: dict[str, Any]) -> str | None:
    if isinstance(row.get('domain'), str):
        return row['domain']
    task_json = row.get('task_json')
    if isinstance(task_json, str):
        try:
            payload = json.loads(task_json)
            if isinstance(payload, dict):
                return payload.get('domain')
        except json.JSONDecodeError:
            return None
    if isinstance(task_json, dict):
        return task_json.get('domain')
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description='Create domain-specific train/eval prompt JSONL files.')
    parser.add_argument('--train-path', type=Path, default=Path('data/training/train_prompts.jsonl'))
    parser.add_argument('--eval-path', type=Path, default=Path('data/training/eval_prompts.jsonl'))
    parser.add_argument('--out-dir', type=Path, default=Path('data/training/domain_splits'))
    args = parser.parse_args()

    for split, path in [('train', args.train_path), ('eval', args.eval_path)]:
        if not path.exists():
            raise FileNotFoundError(path)
        rows = read_jsonl(path)
        by_domain: dict[str, list[dict[str, Any]]] = {'portfolio': [], 'retirement': [], 'loan': []}
        unknown = 0
        for row in rows:
            domain = get_domain(row)
            if domain in by_domain:
                by_domain[domain].append(row)
            else:
                unknown += 1
        for domain, domain_rows in by_domain.items():
            out = args.out_dir / f'{split}_{domain}.jsonl'
            write_jsonl(out, domain_rows)
            print(f'Wrote {len(domain_rows)} {split} rows for {domain} to {out}')
        if unknown:
            print(f'Warning: {unknown} {split} rows had unknown domain and were skipped')


if __name__ == '__main__':
    main()
