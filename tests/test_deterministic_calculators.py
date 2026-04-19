from finplan.sim.financial_models import run_portfolio_backtest, run_retirement_monte_carlo, compute_dti, compute_ltv

def test_portfolio_backtest_outputs():
    bt = run_portfolio_backtest([{'asset':'US_EQ','sector':'TECH','weight':0.6},{'asset':'BONDS','sector':'GOVT','weight':0.4}])
    assert bt['max_drawdown'] >= 0
    assert 0 <= bt['esg_score'] <= 1

def test_retirement_mc_bounds():
    mc = run_retirement_monte_carlo(1_000_000, 3000, 65, 90, n_scenarios=1000)
    assert 0 <= mc['survival_probability'] <= 1

def test_loan_utils():
    assert compute_ltv(200000, 400000) == 0.5
    assert compute_dti(2000, 120000, 500) > 0
