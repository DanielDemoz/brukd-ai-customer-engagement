"""Re-check the churn and CLV models in scripts/, and write docs/recheck.json.

Run from the repo root:  python analysis/recheck.py
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "data" / "shopping_behavior_updated.csv")
AMT, PREV, RATE, SUB = "Purchase Amount (USD)", "Previous Purchases", "Review Rating", "Subscription Status"

# The churn label, exactly as scripts/01_data_preparation.py builds it.
score = ((d[PREV] < d[PREV].quantile(0.25)).astype(int) + (d[RATE] < 3.0).astype(int)
         + (d[SUB] == "No").astype(int) + (d[AMT] < d[AMT].quantile(0.25)).astype(int))
y = (score >= 3).astype(int)
X = pd.get_dummies(d.drop(columns=["Customer ID", "Item Purchased"]), drop_first=True, dtype=float)
rule_cols = [PREV, RATE, AMT, "Subscription Status_Yes"]
cv = StratifiedKFold(5, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=300, random_state=0)
churn = {
    "label_rate": round(float(y.mean()), 4),
    "majority_accuracy": round(float(1 - y.mean()), 4),
    "accuracy_with_rule_inputs": round(float(cross_val_score(rf, X, y, cv=cv).mean()), 4),
    "accuracy_without_rule_inputs": round(float(cross_val_score(rf, X.drop(columns=rule_cols), y, cv=cv).mean()), 4),
}

clv = d[AMT] * d[PREV]
Xc = X.drop(columns=[c for c in X.columns if c in (AMT, PREV)])
kf = KFold(5, shuffle=True, random_state=42)
clv_r2 = {
    "with_formula_inputs": round(float(cross_val_score(HistGradientBoostingRegressor(random_state=0), X, clv, cv=kf, scoring="r2").mean()), 4),
    "without_formula_inputs": round(float(cross_val_score(HistGradientBoostingRegressor(random_state=0), Xc, clv, cv=kf, scoring="r2").mean()), 4),
}
out = {"rows": int(len(d)), "churn": churn, "clv_r2": clv_r2}
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "recheck.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
