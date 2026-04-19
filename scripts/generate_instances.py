from __future__ import annotations
from finplan.generators.portfolio_generator import PortfolioGenerator
from finplan.generators.retirement_generator import RetirementGenerator
from finplan.generators.loan_generator import LoanGenerator
from finplan.utils.io import write_jsonl
from finplan.utils.seeding import seed_everything

def main() -> None:
    seed_everything(42)
    portfolio_rows, retirement_rows, loan_rows = [], [], []
    for difficulty in ["easy", "medium", "hard"]:
        portfolio_rows.extend([t.to_dict() for t in PortfolioGenerator().generate(200, difficulty)])
        retirement_rows.extend([t.to_dict() for t in RetirementGenerator().generate(200, difficulty)])
        loan_rows.extend([t.to_dict() for t in LoanGenerator().generate(200, difficulty)])
    write_jsonl("data/raw/portfolio_instances.jsonl", portfolio_rows)
    write_jsonl("data/raw/retirement_instances.jsonl", retirement_rows)
    write_jsonl("data/raw/loan_instances.jsonl", loan_rows)
    print(f"Generated portfolio instances: {len(portfolio_rows)}")
    print(f"Generated retirement instances: {len(retirement_rows)}")
    print(f"Generated loan instances: {len(loan_rows)}")
    print("Done.")

if __name__ == "__main__":
    main()
