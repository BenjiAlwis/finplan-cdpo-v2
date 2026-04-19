# FinPlan CDPO v2

This repo upgrades the Week 1 scaffold to match Section 2.2 more closely:
- portfolio allocation uses deterministic constraint checks + synthetic backtesting simulation
- retirement planning uses Monte Carlo simulation over 1000 scenarios
- loan structuring uses amortization + rule engine

Profile generation remains reproducible and synthetic rather than using an external LLM.

## Run

```bash
python -m pip install -e .
python -m pytest -q
python scripts/generate_instances.py
python scripts/verify_instances.py
python scripts/create_manual_audit_csv.py
python scripts/generate_invalid_plans.py
python scripts/run_invalid_plan_stress_test.py
python scripts/export_week2_diagnostics.py
python scripts/summarize_week1.py
```
