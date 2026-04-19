from __future__ import annotations

import math
import random
from statistics import mean, median
from typing import Any


# -----------------------------
# Shared helpers
# -----------------------------

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _withdrawal_smoothness(withdrawals: list[float]) -> float:
    if len(withdrawals) <= 1:
        return 1.0

    growth_rates = []
    for i in range(1, len(withdrawals)):
        prev = withdrawals[i - 1]
        cur = withdrawals[i]
        if prev <= 1e-8:
            continue
        growth_rates.append((cur - prev) / prev)

    if not growth_rates:
        return 1.0

    mean_growth = sum(growth_rates) / len(growth_rates)
    var = sum((g - mean_growth) ** 2 for g in growth_rates) / len(growth_rates)
    std = math.sqrt(var)

    return _clip01(1.0 - 10.0 * std)


# -----------------------------
# Portfolio model
# -----------------------------

ASSET_CLASS_MEAN_RETURN = {
    "US_EQ": 0.08,
    "INTL_EQ": 0.075,
    "BONDS": 0.04,
    "CASH": 0.02,
    "REITS": 0.065,
    "COMMODITIES": 0.05,
    "ALT": 0.055,
}

ASSET_CLASS_VOL = {
    "US_EQ": 0.18,
    "INTL_EQ": 0.20,
    "BONDS": 0.07,
    "CASH": 0.01,
    "REITS": 0.16,
    "COMMODITIES": 0.15,
    "ALT": 0.14,
}

ASSET_CLASS_ESG = {
    "US_EQ": 0.60,
    "INTL_EQ": 0.65,
    "BONDS": 0.75,
    "CASH": 0.90,
    "REITS": 0.55,
    "COMMODITIES": 0.30,
    "ALT": 0.50,
}

SECTOR_RISK_MULTIPLIER = {
    "TECH": 1.10,
    "HEALTH": 0.95,
    "INDUSTRIALS": 1.00,
    "ENERGY": 1.15,
    "FOSSIL": 1.20,
    "TOBACCO": 1.10,
    "GOVT": 0.85,
    "CASH": 0.70,
    "UTILITIES": 0.90,
    "FINANCIALS": 1.00,
    "CONSUMER": 0.95,
}


def normalize_allocations(allocations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = sum(_safe_float(x.get("weight"), 0.0) for x in allocations)
    if total <= 0:
        return allocations

    normalized: list[dict[str, Any]] = []
    for item in allocations:
        copied = dict(item)
        copied["weight"] = _safe_float(item.get("weight"), 0.0) / total
        normalized.append(copied)
    return normalized


def estimate_turnover(
    current_allocations: list[dict[str, Any]],
    proposed_allocations: list[dict[str, Any]],
) -> float:
    cur_map: dict[str, float] = {}
    prop_map: dict[str, float] = {}

    for item in current_allocations:
        cur_map[str(item.get("asset"))] = _safe_float(item.get("weight"), 0.0)

    for item in proposed_allocations:
        prop_map[str(item.get("asset"))] = _safe_float(item.get("weight"), 0.0)

    keys = set(cur_map.keys()).union(prop_map.keys())
    l1 = sum(abs(cur_map.get(k, 0.0) - prop_map.get(k, 0.0)) for k in keys)
    return 0.5 * l1


def run_portfolio_backtest(
    allocations: list[dict[str, Any]],
    market_regime: str = "neutral",
    periods: int = 120,
    seed: int = 42,
) -> dict[str, float]:
    """
    Synthetic portfolio backtest / simulation.

    Returns a dict with keys used by portfolio verifier and scorer.
    Includes both preferred names and backward-compatible aliases.
    """
    allocs = normalize_allocations(allocations)
    if not allocs:
        return {
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "max_drawdown": 1.0,
            "esg_score": 0.0,
            "sharpe_like": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "risk_adjusted_return": 0.0,
        }

    regime_shift = {
        "bull": 0.010,
        "neutral": 0.000,
        "bear": -0.015,
    }.get(market_regime, 0.0)

    regime_vol_multiplier = {
        "bull": 0.90,
        "neutral": 1.00,
        "bear": 1.20,
    }.get(market_regime, 1.00)

    weighted_mean = 0.0
    weighted_vol = 0.0
    esg_score = 0.0

    for item in allocs:
        asset = str(item.get("asset", "US_EQ"))
        sector = str(item.get("sector", "TECH"))
        weight = _safe_float(item.get("weight"), 0.0)

        base_mean = ASSET_CLASS_MEAN_RETURN.get(asset, 0.06)
        base_vol = ASSET_CLASS_VOL.get(asset, 0.15)
        sector_mult = SECTOR_RISK_MULTIPLIER.get(sector, 1.0)

        weighted_mean += weight * base_mean
        weighted_vol += weight * base_vol * sector_mult
        esg_score += weight * ASSET_CLASS_ESG.get(asset, 0.5)

    weighted_mean += regime_shift
    weighted_vol *= regime_vol_multiplier

    rng = random.Random(seed)
    monthly_mean = weighted_mean / 12.0
    monthly_vol = weighted_vol / math.sqrt(12.0)

    portfolio_values = [1.0]
    monthly_returns = []

    for _ in range(periods):
        r = rng.gauss(monthly_mean, monthly_vol)
        r = max(r, -0.95)
        monthly_returns.append(r)
        portfolio_values.append(portfolio_values[-1] * (1.0 + r))

    avg_monthly = mean(monthly_returns) if monthly_returns else 0.0
    annual_return = avg_monthly * 12.0

    if monthly_returns:
        m = avg_monthly
        var = sum((x - m) ** 2 for x in monthly_returns) / len(monthly_returns)
        monthly_sd = math.sqrt(var)
    else:
        monthly_sd = 0.0
    annual_volatility = monthly_sd * math.sqrt(12.0)

    peak = portfolio_values[0]
    max_drawdown = 0.0
    for v in portfolio_values:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0.0
        max_drawdown = max(max_drawdown, dd)

    risk_free_rate = 0.02
    if annual_volatility <= 1e-8:
        sharpe_like = 0.0
    else:
        sharpe = (annual_return - risk_free_rate) / annual_volatility
        sharpe_like = _clip01((sharpe + 1.0) / 4.0)

    return {
        # Preferred names
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "max_drawdown": max_drawdown,
        "esg_score": _clip01(esg_score),
        "sharpe_like": sharpe_like,

        # Backward-compatible aliases expected elsewhere in v2
        "annualized_return": annual_return,
        "annualized_volatility": annual_volatility,
        "risk_adjusted_return": sharpe_like,
    }


# -----------------------------
# Retirement model
# -----------------------------

def run_retirement_monte_carlo(
    starting_balance: float,
    monthly_income_target: float,
    current_age: int,
    target_age: int,
    inflation_assumption: float = 0.03,
    portfolio_return_assumption: float = 0.05,
    annual_volatility: float = 0.12,
    n_scenarios: int = 1000,
    seed: int = 42,
) -> dict:
    years = max(0, target_age - current_age)
    annual_withdrawal_base = float(monthly_income_target) * 12.0

    rng = random.Random(seed)

    survived_flags: list[int] = []
    ending_balances: list[float] = []
    years_survived_list: list[int] = []

    withdrawal_schedule = [
        annual_withdrawal_base * ((1.0 + inflation_assumption) ** y)
        for y in range(years)
    ]
    withdrawal_smoothness_score = _withdrawal_smoothness(withdrawal_schedule)

    for _ in range(n_scenarios):
        balance = float(starting_balance)
        survived = 1
        years_survived = 0

        for y in range(years):
            annual_return = rng.gauss(
                portfolio_return_assumption,
                annual_volatility,
            )
            annual_return = max(annual_return, -0.95)

            balance *= (1.0 + annual_return)
            balance -= withdrawal_schedule[y]

            if balance < 0:
                survived = 0
                years_survived = y
                break

            years_survived = y + 1

        survived_flags.append(survived)
        ending_balances.append(balance)
        years_survived_list.append(years_survived)

    survival_probability = sum(survived_flags) / max(n_scenarios, 1)
    depletion_probability = 1.0 - survival_probability

    mean_ending_balance = mean(ending_balances) if ending_balances else 0.0
    median_ending_balance = median(ending_balances) if ending_balances else 0.0
    median_years_survived = int(median(years_survived_list)) if years_survived_list else 0

    lifestyle_base = _clip01(monthly_income_target / 6000.0)
    lifestyle_quality_score = _clip01(lifestyle_base * (0.5 + 0.5 * survival_probability))

    return {
        "survival_probability": survival_probability,
        "depletion_probability": depletion_probability,
        "depletes_before_target_age": depletion_probability > 0.5,
        "ending_balance": median_ending_balance,
        "median_ending_balance": median_ending_balance,
        "mean_ending_balance": mean_ending_balance,
        "years_survived": median_years_survived,
        "lifestyle_quality_score": lifestyle_quality_score,
        "withdrawal_smoothness_score": withdrawal_smoothness_score,
    }


def estimate_bequest_alignment(
    ending_balance: float,
    bequest_preference: float,
    starting_balance: float,
) -> float:
    if starting_balance <= 1e-8:
        return 0.0
    preserved_ratio = _clip01(ending_balance / starting_balance)
    target = _clip01(bequest_preference)
    return _clip01(1.0 - abs(preserved_ratio - target))


# -----------------------------
# Loan model
# -----------------------------

def compute_monthly_payment(
    principal: float,
    annual_rate: float,
    term_months: int,
) -> float:
    if term_months <= 0:
        return 0.0

    r = annual_rate / 12.0
    if abs(r) < 1e-12:
        return principal / term_months

    numerator = principal * r * ((1.0 + r) ** term_months)
    denominator = ((1.0 + r) ** term_months) - 1.0
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


def compute_dti(
    monthly_payment: float,
    annual_income: float,
    existing_monthly_debt: float = 0.0,
) -> float:
    gross_monthly_income = annual_income / 12.0
    if gross_monthly_income <= 1e-8:
        return 999.0
    return (monthly_payment + existing_monthly_debt) / gross_monthly_income


def compute_ltv(
    loan_amount: float,
    property_value: float,
) -> float:
    if property_value <= 1e-8:
        return 999.0
    return loan_amount / property_value


def check_basic_regulatory_rules(
    dti: float,
    ltv: float,
    rate_type: str,
) -> bool:
    if dti > 0.50:
        return False
    if ltv > 0.95:
        return False
    if rate_type not in {"fixed", "variable"}:
        return False
    return True


def estimate_interest_cost_score(
    loan_amount: float,
    annual_rate: float,
    term_months: int,
) -> float:
    payment = compute_monthly_payment(loan_amount, annual_rate, term_months)
    total_paid = payment * term_months
    total_interest = max(0.0, total_paid - loan_amount)

    if loan_amount <= 1e-8:
        return 0.0

    interest_ratio = total_interest / loan_amount
    return _clip01(1.0 - min(interest_ratio, 1.0))


def estimate_payment_flexibility_score(
    term_months: int,
    prepayment_option: bool,
) -> float:
    term_component = _clip01((term_months - 60) / (360 - 60)) if term_months >= 60 else 0.0
    prepay_bonus = 0.2 if prepayment_option else 0.0
    return _clip01(term_component + prepay_bonus)


def estimate_prepayment_optionality_score(prepayment_option: bool) -> float:
    return 1.0 if prepayment_option else 0.3
