from __future__ import annotations
from collections import Counter
from statistics import mean
from finplan.utils.io import read_jsonl

PROCESSED = [("portfolio", "data/processed/portfolio_verified.jsonl"), ("retirement", "data/processed/retirement_verified.jsonl"), ("loan", "data/processed/loan_verified.jsonl")]

def summarize_domain(domain: str, path: str) -> dict:
    rows = read_jsonl(path); n = len(rows)
    if n == 0:
        print(f"\nSummary for {domain}\nRows: 0")
        return {"domain": domain, "rows": 0}
    parse_flags=[]; hard_flags=[]; soft_means=[]; combined=[]; fails=Counter(); totals=Counter(); violations=Counter()
    for row in rows:
        reward = row.get("reward", {}); md = reward.get("metadata", {})
        parse_flags.append(int(bool(row.get("parse_success", row.get("parsed", {}).get("parse_success")))))
        hard_flags.append(int(row.get("all_constraints_pass", reward.get("hard", {}).get("all_pass", 0))))
        soft_means.append(float(row.get("soft_mean_score", reward.get("soft", {}).get("mean_score", 0.0))))
        combined.append(float(row.get("combined_quality", reward.get("combined_quality", 0.0))))
        hard = row.get("hard_checks", reward.get("hard", {}).get("checks", {})); violated = row.get("violated_constraints", md.get("violated_constraints", []))
        for k,v in hard.items():
            totals[k]+=1
            if int(v)==0: fails[k]+=1
        for v in violated: violations[v]+=1
    print(f"\nSummary for {domain}")
    print(f"Rows: {n}")
    print(f"Parse success rate: {mean(parse_flags):.4f}")
    print(f"Hard pass rate: {mean(hard_flags):.4f}")
    print(f"Mean soft score: {mean(soft_means):.4f}")
    print(f"Mean combined quality: {mean(combined):.4f}")
    print("\nPer-check failure rates:")
    for k in sorted(totals):
        print(f"  {k}: {fails[k]}/{totals[k]} ({(fails[k]/totals[k] if totals[k] else 0):.2%})")
    print("\nMost common violated constraints:")
    if violations:
        for k,c in violations.most_common(10): print(f"  {k}: {c}")
    else:
        print("  None")
    return {"domain": domain, "rows": n, "parse_success_rate": mean(parse_flags), "hard_pass_rate": mean(hard_flags), "mean_soft_score": mean(soft_means), "mean_combined_quality": mean(combined)}

def main() -> None:
    results=[summarize_domain(d,p) for d,p in PROCESSED]
    valid=[r for r in results if r.get("rows",0)>0]
    print("\n"+"="*60); print("GLOBAL SUMMARY"); print("="*60)
    print(f"Domains summarized: {len(valid)}")
    print(f"Total rows: {sum(r['rows'] for r in valid)}")
    print(f"Average parse success rate: {mean(r['parse_success_rate'] for r in valid):.4f}")
    print(f"Average hard pass rate: {mean(r['hard_pass_rate'] for r in valid):.4f}")
    print(f"Average mean soft score: {mean(r['mean_soft_score'] for r in valid):.4f}")
    print(f"Average combined quality: {mean(r['mean_combined_quality'] for r in valid):.4f}")

if __name__ == "__main__":
    main()
