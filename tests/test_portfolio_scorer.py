from finplan.scorers.portfolio_scorer import PortfolioScorer
from finplan.sim.financial_models import run_portfolio_backtest
from finplan.types import TaskInstance, ParsedPlan

def make_task(target_risk=0.15, target_esg=0.70, turnover_penalty_scale=1.0, market_regime='neutral', current_allocations=None):
    return TaskInstance('portfolio-scorer-test','portfolio',{}, {}, {'target_risk':target_risk,'target_esg':target_esg,'turnover_penalty_scale':turnover_penalty_scale}, {'market_regime':market_regime,'current_allocations':current_allocations or []}, {'type':'json','required_fields':['allocations']})

def make_plan(allocations):
    return ParsedPlan('portfolio','',{'allocations':allocations},True)

def test_scores_bounded():
    scorer = PortfolioScorer(); task = make_task(current_allocations=[{'asset':'US_EQ','sector':'TECH','weight':0.5},{'asset':'BONDS','sector':'GOVT','weight':0.3},{'asset':'CASH','sector':'CASH','weight':0.2}])
    plan = make_plan([{'asset':'US_EQ','sector':'TECH','weight':0.35},{'asset':'INTL_EQ','sector':'INDUSTRIALS','weight':0.20},{'asset':'BONDS','sector':'GOVT','weight':0.35},{'asset':'CASH','sector':'CASH','weight':0.10}])
    for v in scorer.score(task, plan).scores.values(): assert 0 <= v <= 1

def test_risk_alignment_prefers_closer_plan():
    scorer = PortfolioScorer(); task = make_task(target_risk=0.08)
    closer = make_plan([{'asset':'US_EQ','sector':'TECH','weight':0.45},{'asset':'BONDS','sector':'GOVT','weight':0.35},{'asset':'CASH','sector':'CASH','weight':0.20}])
    farther = make_plan([{'asset':'US_EQ','sector':'TECH','weight':0.70},{'asset':'INTL_EQ','sector':'ENERGY','weight':0.20},{'asset':'REITS','sector':'FINANCIALS','weight':0.10}])
    assert abs(run_portfolio_backtest(closer.structured['allocations'])['annualized_volatility'] - 0.08) < abs(run_portfolio_backtest(farther.structured['allocations'])['annualized_volatility'] - 0.08)
    assert scorer.score(task, closer).scores['risk_alignment'] > scorer.score(task, farther).scores['risk_alignment']
