# Runpod commands for Week 2 and Week 3

This file assumes the repo has already been pushed to GitHub and you are starting from a fresh Runpod pod.

## 1. Clone and set up the environment

```bash
git clone https://github.com/BenjiAlwis/finplan-cdpo-v2.git
cd finplan-cdpo-v2

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install -r requirements.txt

python -m pytest -q
python scripts/build_training_prompts.py
```

## 2. Smoke tests first

Run these before spending money on full 7B runs.

```bash
accelerate launch scripts/run_grpo_baseline.py \
  --config configs/grpo_runpod_smoke.yaml

accelerate launch scripts/run_cdpo_baseline.py \
  --config configs/cdpo_runpod_smoke.yaml
```

Check diagnostics:

```bash
find data/rl_runs -name 'diagnostics.jsonl' -print
```

## 3. Three-pod layout

### Pod 1: GRPO baseline

```bash
accelerate launch scripts/run_grpo_baseline.py \
  --config configs/grpo_cloud_gpu_baseline.yaml
```

### Pod 2: GDPO baseline

```bash
accelerate launch scripts/run_gdpo_baseline.py \
  --config configs/gdpo_cloud_gpu_baseline.yaml
```

### Pod 3: CDPO fixed-alpha baseline

```bash
accelerate launch scripts/run_cdpo_baseline.py \
  --config configs/cdpo_cloud_gpu_fixed.yaml
```

## 4. CDPO ablations

After fixed-alpha CDPO works, run these as additional ablations.

```bash
accelerate launch scripts/run_cdpo_baseline.py \
  --config configs/cdpo_cloud_gpu_anneal.yaml

accelerate launch scripts/run_cdpo_baseline.py \
  --config configs/cdpo_cloud_gpu_learned.yaml
```

Note: `cdpo_cloud_gpu_learned.yaml` currently uses an adaptive alpha proxy, not a true meta-gradient learned alpha.

## 5. Analyze diagnostics after runs

Copy the `data/rl_runs/*/diagnostics.jsonl` files into one machine or volume, then run:

```bash
python scripts/analyze_advantage_distributions.py \
  data/rl_runs/grpo_baseline/diagnostics.jsonl \
  data/rl_runs/gdpo_baseline/diagnostics.jsonl \
  data/rl_runs/cdpo_fixed/diagnostics.jsonl \
  --out-dir data/analysis/week23
```

Plot curves:

```bash
python scripts/plot_training_curves.py \
  --summary-csv data/analysis/week23/batch_summaries.csv \
  --out-dir data/analysis/week23/figures
```

## 6. What to inspect

Key files:

```bash
tail -n 5 data/rl_runs/grpo_baseline/diagnostics.jsonl
tail -n 5 data/rl_runs/gdpo_baseline/diagnostics.jsonl
tail -n 5 data/rl_runs/cdpo_fixed/diagnostics.jsonl
```

Key outputs:

```text
data/analysis/week23/batch_summaries.csv
data/analysis/week23/signal_collapse_summary.csv
data/analysis/week23/plan_eval_summary.json
data/analysis/week23/figures/*.png
```

The main Week 2 question is whether monolithic GRPO maps distinct violation patterns into fewer advantage groups than GDPO/CDPO.
