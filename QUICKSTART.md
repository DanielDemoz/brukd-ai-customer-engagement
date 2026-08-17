# ⚡ Quick Start Guide

Get up and running with Daniel S. Demoz AI-Driven Customer Engagement in 5 minutes!

---

## 🚀 For the Impatient

```bash
# 1. Clone and navigate
git clone https://github.com/brukd/ai-customer-engagement.git
cd brukd-ai-customer-engagement

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the complete pipeline
python scripts/run_pipeline.py

# 4. Launch dashboard
streamlit run dashboard_app.py
```

**That's it!** Open your browser to `http://localhost:8501` and explore.

---

## 📋 Detailed Steps

### Step 1: Prerequisites

Make sure you have:
- Python 3.8+ installed
- 4GB RAM available
- 2GB disk space

Check your Python version:
```bash
python --version
```

### Step 2: Clone Repository

```bash
git clone https://github.com/brukd/ai-customer-engagement.git
cd brukd-ai-customer-engagement
```

### Step 3: Set Up Environment

**Option A: Virtual Environment (Recommended)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

**Option B: Conda**
```bash
conda create -n brukd python=3.9
conda activate brukd
```

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**This installs:**
- Pandas, NumPy (data processing)
- Scikit-learn, XGBoost (machine learning)
- Streamlit (dashboard)
- Plotly, Matplotlib, Seaborn (visualization)
- And more...

### Step 5: Run the Analysis

**Option A: Complete Pipeline (Recommended for first time)**
```bash
python scripts/run_pipeline.py
```

This will:
1. Process the data (2 min)
2. Segment customers (1 min)
3. Train churn model (2 min)
4. Generate recommendations (1 min)

**Expected output:**
```
================================================================================
                    DANIEL S. DEMOZ — AI-DRIVEN CUSTOMER ENGAGEMENT
                         Complete Analysis Pipeline
================================================================================

Running: Step 1: Data Preparation & Feature Engineering
...
✓ All steps completed successfully!
Total execution time: 6.34 minutes
```

**Option B: Individual Scripts**
```bash
python scripts/01_data_preparation.py
python scripts/02_customer_segmentation.py
python scripts/03_churn_prediction_90_days.py
python scripts/04_engagement_recommendations.py
```

### Step 6: Launch Dashboard

```bash
streamlit run dashboard_app.py
```

**Browser will automatically open to:**
`http://localhost:8501`

If it doesn't, manually navigate to that URL.

---

## 🎯 What You'll See

### Dashboard Features

1. **Executive Dashboard** 📊
   - Key metrics and KPIs
   - Segment distribution
   - Revenue analysis

2. **Customer Segments** 👥
   - 3 distinct segments
   - Detailed characteristics
   - Behavioral patterns

3. **Predictions** 🔮
   - CLV predictions (99.96% accurate)
   - 90-day churn risk
   - Model performance metrics

4. **Recommendations** 💡
   - Personalized actions per customer
   - Campaign templates
   - Priority rankings

5. **ROI Analysis** 📈
   - Marketing budget optimization
   - Expected returns
   - Sensitivity analysis

6. **Customer Lookup** 🔍
   - Individual customer profiles
   - Specific recommendations
   - Risk assessment

---

## 📁 Project Structure Overview

```
brukd-ai-customer-engagement/
├── data/                           # Datasets and outputs
│   ├── shopping_behavior_updated.csv  # Raw data
│   └── [Generated files after running pipeline]
│
├── models/                         # Trained ML models
│   └── [Generated after running pipeline]
│
├── scripts/                        # Analysis scripts
│   ├── 01_data_preparation.py
│   ├── 02_customer_segmentation.py
│   ├── 03_churn_prediction_90_days.py
│   ├── 04_engagement_recommendations.py
│   └── run_pipeline.py
│
├── visualizations/                 # Generated charts
│   └── [Created during analysis]
│
├── docs/                          # Documentation
│   ├── BLOG_POST.md              # Case study article
│   └── DEPLOYMENT_GUIDE.md       # Production deployment
│
├── dashboard_app.py               # Interactive dashboard
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container configuration
├── docker-compose.yml            # Multi-container setup
└── README.md                     # Full documentation
```

---

## 🎓 Understanding the Output

### Generated Files

After running the pipeline, you'll find:

**Data Files:**
- `data_processed.csv` - Cleaned data with features
- `data_segmented.csv` - With cluster labels
- `data_with_churn_predictions.csv` - Complete predictions
- `customer_engagement_recommendations.csv` - Action plans
- `high_priority_customers.csv` - Urgent actions needed

**Models:**
- `kmeans_model.pkl` - Segmentation model
- `churn_model.pkl` - Churn prediction model
- `scaler.pkl` - Feature scaler

**Visualizations:**
- `cluster_optimization.png` - Elbow & silhouette plots
- `segment_analysis.png` - Segment characteristics
- `churn_prediction_evaluation.png` - Model performance

---

## 🔧 Common Issues & Solutions

### Issue: Module not found
```bash
# Solution: Ensure you activated virtual environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Then reinstall
pip install -r requirements.txt
```

### Issue: Port 8501 already in use
```bash
# Solution: Use a different port
streamlit run dashboard_app.py --server.port 8502
```

### Issue: Out of memory
```bash
# Solution: Close other applications or use subset of data
# Edit scripts to use df.sample(frac=0.5) for testing
```

---

## 📚 Next Steps

### Explore the Results

1. **Review Visualizations**
   ```bash
   # Open folder
   cd visualizations
   ```

2. **Read the Blog Post**
   ```bash
   # Open in text editor
   cat docs/BLOG_POST.md
   ```

3. **Check Recommendations**
   ```bash
   # View high-priority customers
   python -c "import pandas as pd; print(pd.read_csv('data/high_priority_customers.csv').head())"
   ```

### Customize for Your Data

1. **Replace the dataset:**
   - Place your CSV in `data/` folder
   - Update `DATA_PATH` in `01_data_preparation.py`
   - Ensure column names match or update mapping

2. **Adjust parameters:**
   - Number of clusters: Edit `optimal_k` in `02_customer_segmentation.py`
   - Model hyperparameters: Modify XGBoost params in scripts
   - Churn threshold: Change in `01_data_preparation.py`

3. **Customize recommendations:**
   - Edit segment strategies in `04_engagement_recommendations.py`
   - Modify campaign templates
   - Adjust priority rules

---

## 🐳 Docker Quick Start

If you prefer Docker:

```bash
# Build and run
docker-compose up -d

# Access dashboard
open http://localhost:8501

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## 💡 Tips for Success

### For Data Scientists
- Review feature engineering in `01_data_preparation.py`
- Experiment with different K values in segmentation
- Try other ML algorithms (Random Forest, LightGBM)
- Add SHAP values for model interpretability

### For Business Users
- Focus on the dashboard interface
- Review `high_priority_customers.csv` for immediate actions
- Use the blog post for presenting to stakeholders
- Customize campaign templates in recommendations

### For Developers
- Check out the API structure for integration
- Review Docker setup for deployment
- Explore CI/CD options in deployment guide
- Consider adding database integration

---

## 📞 Getting Help

- **Documentation:** Read full [README.md](README.md)
- **Deployment:** See [DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)
- **Issues:** [GitHub Issues](https://github.com/brukd/ai-customer-engagement/issues)
- **Email:** asbdansi9@gmail.com

---

## ✅ Success Checklist

- [ ] Python 3.8+ installed
- [ ] Repository cloned
- [ ] Dependencies installed
- [ ] Pipeline executed successfully
- [ ] Dashboard running locally
- [ ] Explored all dashboard tabs
- [ ] Reviewed generated visualizations
- [ ] Read blog post case study
- [ ] Understood segment characteristics
- [ ] Examined recommendations output

---

## 🎉 You're Ready!

You now have a fully functional AI-driven customer engagement system!

**Next:** Try it with your own data or deploy to production.

**Questions?** Read the full [README.md](README.md) or contact us at asbdansi9@gmail.com

---

*© 2025 Daniel S. Demoz. Made with ❤️ for data-driven businesses.*

