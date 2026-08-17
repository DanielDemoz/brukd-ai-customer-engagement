# Daniel S. Demoz — AI Customer Segmentation & Predictive Engagement

End-to-end customer intelligence framework combining segmentation, CLV prediction, churn modeling, and segment-specific engagement recommendations.

## Problem

Fashion retail clients need actionable customer segments and predictive models to target retention spend—not just static RFM reports.

## Approach

Processed 3,900 anonymized customer records: K-Means segmentation (K=3), XGBoost CLV regression, 90-day churn classification, Random Forest segment assignment for new customers, and a rule-based engagement recommendation engine. Packaged analysis scripts, trained models, Streamlit dashboard, and optional FastAPI serving layer.

## Results

- 3 segments: high-value frequent buyers (~33%), budget-minded occasional (~41%), at-risk churners (~26%)
- CLV model: R² = 0.9996; churn model: 87%+ accuracy (90-day window)
- Segment classifier: 92%+ accuracy for new customers
- Marketing ROI scenario: 29.36× on $250K ad budget; +12% predicted re-engagement lift

## Tech stack

Python, scikit-learn, XGBoost, pandas, Streamlit, FastAPI, Docker, Joblib

## How to run

```bash
git clone https://github.com/DanielDemoz/brukd-ai-customer-engagement.git
cd brukd-ai-customer-engagement
pip install -r requirements.txt
python scripts/run_pipeline.py
streamlit run dashboard_app.py
```

Open http://localhost:8501

## Screenshot / demo

Run the Streamlit dashboard locally or review segment visualizations in `visualizations/`. See `docs/METHODOLOGY.md` for full pipeline documentation.

Built by Daniel S. Demoz as a replicable client engagement showcase.

## Contact

Daniel S. Demoz  
📧 Email: asbdansi9@gmail.com  
📱 Phone: (437) 249-3308  
🔗 LinkedIn: linkedin.com/in/daniel-s-demoz  
💼 GitHub: github.com/DanielDemoz  
🌐 Website: brukdconsultancy.com
