from finplan.parsers.plan_parser import PlanParser

def test_valid_json_parses():
    parser = PlanParser()
    parsed = parser.parse('{"allocations":[{"asset":"A","sector":"TECH","weight":1.0}]}', 'portfolio', {'required_fields':['allocations']})
    assert parsed.parse_success is True

def test_invalid_json_fails():
    parser = PlanParser()
    parsed = parser.parse('{"a": 1', 'portfolio')
    assert parsed.parse_success is False
