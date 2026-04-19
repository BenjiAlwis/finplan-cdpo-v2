from __future__ import annotations
from finplan.sim.financial_models import compute_monthly_payment, compute_dti, compute_ltv, check_basic_regulatory_rules
from finplan.types import TaskInstance, ParsedPlan, HardConstraintResult
from finplan.verifiers.base_verifier import BaseVerifier

class LoanVerifier(BaseVerifier):
    def verify(self, task: TaskInstance, plan: ParsedPlan) -> HardConstraintResult:
        if not plan.parse_success:
            return HardConstraintResult({"parse_valid":0,"dti_valid":0,"ltv_valid":0,"regulatory_valid":0})
        s = plan.structured
        pmt = compute_monthly_payment(float(s.get("loan_amount",0.0)), float(s.get("annual_rate",task.context.get("base_rate",0.05))), int(s.get("term_months",360)))
        dti = compute_dti(pmt, float(task.profile.get("annual_income",0.0)), float(s.get("existing_monthly_debt",0.0)))
        ltv = compute_ltv(float(s.get("loan_amount",0.0)), float(s.get("property_value",0.0)))
        reg = check_basic_regulatory_rules(dti, ltv, str(s.get("rate_type","fixed")))
        return HardConstraintResult({
            "parse_valid":1,
            "dti_valid": int(dti <= float(task.constraints.get("max_dti",0.43))),
            "ltv_valid": int(ltv <= float(task.constraints.get("max_ltv",0.8))),
            "regulatory_valid": int(reg),
        })
