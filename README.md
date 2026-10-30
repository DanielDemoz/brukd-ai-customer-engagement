# AI Customer Segmentation and Engagement: a re-check

A pipeline for customer segments, churn risk, CLV and engagement rules, built on a public shopping-behaviour dataset and re-checked in October 2026.

**Case study page:** https://danieldemoz.github.io/brukd-ai-customer-engagement/
**Related:** [customer-segmentation-clv](https://github.com/DanielDemoz/customer-segmentation-clv) has the interactive CLV calculator and ROI scenario.

## What the pipeline does

1. `scripts/01_data_preparation.py` cleans the 3,900 records and builds a CLV value and a churn-risk label.
2. `scripts/02_customer_segmentation.py` groups customers with K-means.
3. `scripts/03_churn_prediction_90_days.py` trains random forest and XGBoost on the churn-risk label.
4. `scripts/04_engagement_recommendations.py` maps segment and risk to an engagement action.

## What the re-check found

`analysis/recheck.py` reproduces the labels and tests the models with 5-fold cross-validation.

| Check | Result |
|---|---|
| Churn: always guess "not at risk" | 90.2% |
| Churn: random forest, all columns | 98.4% |
| Churn: random forest, without the 4 rule columns | 90.2% |
| CLV: gradient boosting R², all columns | 0.9996 |
| CLV: R² without the 2 formula columns | -0.06 |

- The churn label is a rule built from 4 input columns, so the model re-learns the rule.
- CLV is purchase amount x previous purchases, and both were inputs.
- The data has no dates, so a "90-day" churn window cannot be measured.

The earlier README, summary and blog post reported these scores as model performance, along with ROI and re-engagement figures based on assumptions. A blog post written as a client case study, with a testimonial, was also removed, because the project uses a public dataset.

## Responsible AI, following the Reliable AI Adoption Framework

1. **Business problem:** target retention spend at customers likely to leave. That needs a label based on real future behaviour.
2. **Data readiness:** 3,900 records from a public shopping-behaviour dataset, included in `data/`. No dates, and it looks synthetic.
3. **Governance and risk:** leaked labels give confident scores that would misdirect offers, so the re-check is shown first.
4. **Technology and tools:** rules are the honest tool for this data. Models come back once a real label exists.
5. **People and adoption:** marketing staff can read and question a rule.
6. **Outcomes and monitoring:** with real data, compare predicted risk with who actually stopped buying, every quarter.

## How to run

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py      # the original pipeline
python analysis/recheck.py          # the re-check, writes docs/recheck.json
pytest -q
```

## Tech stack

Python, pandas, scikit-learn, XGBoost, imbalanced-learn, Streamlit (`dashboard_app.py`).
