from finplan.env.finplan_env import FinPlanEnv
from finplan.types import TaskInstance

def test_invalid_portfolio_weights_sum():
    env = FinPlanEnv()
    task = TaskInstance('x','portfolio',{}, {'max_drawdown':0.30,'min_assets':2,'banned_sectors':[]}, {'target_risk':0.15,'target_esg':0.6,'turnover_penalty_scale':1.0}, {'market_regime':'neutral','current_allocations':[]}, {'type':'json','required_fields':['allocations']})
    parsed, reward = env.evaluate(task, '{"allocations":[{"asset":"US_EQ","sector":"TECH","weight":0.7},{"asset":"BONDS","sector":"GOVT","weight":0.4}]}')
    assert parsed.parse_success is True
    assert reward.hard.checks['weights_sum_valid'] == 0
