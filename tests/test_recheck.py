import json
from pathlib import Path

R = json.loads((Path(__file__).resolve().parents[1] / "docs" / "recheck.json").read_text())


def test_churn_model_adds_nothing_without_rule_inputs():
    c = R["churn"]
    assert abs(c["accuracy_without_rule_inputs"] - c["majority_accuracy"]) < 0.01


def test_clv_leakage_documented():
    assert R["clv_r2"]["with_formula_inputs"] > 0.99 > 0.1 > R["clv_r2"]["without_formula_inputs"]
