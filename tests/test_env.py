import json
from finplan.env.finplan_env import FinPlanEnv
from finplan.types import TaskInstance

def test_env_basic():
    env = FinPlanEnv()
    task = TaskInstance('x','portfolio',{}, {'max_drawdown':0.30,'min_assets':3,'banned_sectors':[]}, {'target_risk':0.15,'target_esg':0.70,'turnover_penalty_scale':1.0}, {'market_regime':'neutral','current_allocations':[]}, {'type':'json','required_fields':['allocations']})
    raw = json.dumps({'allocations':[{'asset':'US_EQ','sector':'TECH','weight':0.3},{'asset':'INTL_EQ','sector':'INDUSTRIALS','weight':0.2},{'asset':'BONDS','sector':'GOVT','weight':0.35},{'asset':'CASH','sector':'CASH','weight':0.15}]})
    parsed, reward = env.evaluate(task, raw)
    assert parsed.parse_success is True
    assert 'hard_channel' in reward.metadata and 'soft_channel' in reward.metadata
