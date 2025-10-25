# 🚀 Deployment Guide

Complete guide for deploying the Brukd AI-Driven Customer Engagement solution to production.

---

## Table of Contents

1. [Local Development Setup](#local-development-setup)
2. [Running the Complete Pipeline](#running-the-complete-pipeline)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Production Considerations](#production-considerations)
6. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Local Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- 4GB+ RAM recommended
- 2GB disk space for data and models

### Step 1: Clone the Repository

```bash
git clone https://github.com/brukd/ai-customer-engagement.git
cd brukd-ai-customer-engagement
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import pandas, sklearn, xgboost, streamlit; print('✓ All dependencies installed')"
```

---

## Running the Complete Pipeline

### Option 1: Run Complete Pipeline

Execute all analysis steps in sequence:

```bash
python scripts/run_pipeline.py
```

This will:
1. Prepare and engineer features
2. Perform customer segmentation
3. Train churn prediction model
4. Generate engagement recommendations
5. Create all visualizations

**Expected Duration:** 5-10 minutes on standard hardware

### Option 2: Run Individual Steps

Run each script separately for debugging or customization:

```bash
# Step 1: Data Preparation
python scripts/01_data_preparation.py

# Step 2: Customer Segmentation
python scripts/02_customer_segmentation.py

# Step 3: Churn Prediction
python scripts/03_churn_prediction_90_days.py

# Step 4: Engagement Recommendations
python scripts/04_engagement_recommendations.py
```

### Step 5: Launch Dashboard

```bash
streamlit run dashboard_app.py
```

Access at: `http://localhost:8501`

---

## Docker Deployment

### Build Docker Image

```bash
docker build -t brukd-customer-engagement .
```

### Run Container

```bash
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/visualizations:/app/visualizations \
  brukd-customer-engagement
```

### Using Docker Compose (Recommended)

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Access dashboard at: `http://localhost:8501`

---

## Cloud Deployment

### AWS Deployment

#### Option 1: AWS ECS (Elastic Container Service)

**1. Install AWS CLI:**
```bash
pip install awscli
aws configure
```

**2. Create ECR Repository:**
```bash
aws ecr create-repository --repository-name brukd-customer-engagement
```

**3. Build and Push Image:**
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag image
docker tag brukd-customer-engagement:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/brukd-customer-engagement:latest

# Push image
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/brukd-customer-engagement:latest
```

**4. Deploy to ECS:**
```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name brukd-cluster

# Create task definition (use provided template)
aws ecs register-task-definition --cli-input-json file://deploy/aws-ecs-task-definition.json

# Create service
aws ecs create-service --cluster brukd-cluster --service-name brukd-dashboard --task-definition brukd-customer-engagement --desired-count 1
```

#### Option 2: AWS EC2

**1. Launch EC2 Instance:**
- AMI: Amazon Linux 2 or Ubuntu 20.04
- Instance Type: t3.medium or larger
- Security Group: Allow inbound on port 8501

**2. SSH and Setup:**
```bash
ssh -i your-key.pem ec2-user@<instance-ip>

# Install Docker
sudo yum update -y
sudo yum install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user

# Clone repo and run
git clone https://github.com/brukd/ai-customer-engagement.git
cd brukd-ai-customer-engagement
docker-compose up -d
```

**3. Access:**
Navigate to `http://<instance-ip>:8501`

---

### Google Cloud Platform (GCP)

#### Deploy to Cloud Run

**1. Install gcloud CLI:**
```bash
curl https://sdk.cloud.google.com | bash
gcloud init
```

**2. Build and Deploy:**
```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Build container
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/brukd-customer-engagement

# Deploy to Cloud Run
gcloud run deploy brukd-dashboard \
  --image gcr.io/YOUR_PROJECT_ID/brukd-customer-engagement \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

**3. Access:**
Your service will be available at the URL provided by Cloud Run.

---

### Microsoft Azure

#### Deploy to Azure Container Instances

**1. Install Azure CLI:**
```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az login
```

**2. Create Resources:**
```bash
# Create resource group
az group create --name brukd-rg --location eastus

# Create container registry
az acr create --resource-group brukd-rg --name brukdacr --sku Basic

# Login to registry
az acr login --name brukdacr
```

**3. Build and Push:**
```bash
# Tag image
docker tag brukd-customer-engagement brukdacr.azurecr.io/brukd-customer-engagement:latest

# Push image
docker push brukdacr.azurecr.io/brukd-customer-engagement:latest
```

**4. Deploy Container:**
```bash
az container create \
  --resource-group brukd-rg \
  --name brukd-dashboard \
  --image brukdacr.azurecr.io/brukd-customer-engagement:latest \
  --cpu 2 \
  --memory 4 \
  --registry-login-server brukdacr.azurecr.io \
  --registry-username $(az acr credential show --name brukdacr --query username -o tsv) \
  --registry-password $(az acr credential show --name brukdacr --query passwords[0].value -o tsv) \
  --dns-name-label brukd-dashboard \
  --ports 8501
```

**5. Access:**
Navigate to `http://brukd-dashboard.eastus.azurecontainer.io:8501`

---

## Production Considerations

### Security

#### 1. Authentication

Add authentication to Streamlit dashboard:

```python
# dashboard_app.py
import streamlit_authenticator as stauth

# Create authentication object
authenticator = stauth.Authenticate(
    ['usernames'],
    ['names'],
    ['passwords'],
    'brukd_dashboard',
    'auth_key',
    cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    # Dashboard content
    pass
elif authentication_status == False:
    st.error('Username/password is incorrect')
```

#### 2. HTTPS/SSL

Use nginx as reverse proxy with SSL:

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

#### 3. Environment Variables

Store sensitive data in environment variables:

```bash
# .env file
DATABASE_URL=postgresql://user:pass@host:5432/db
API_KEY=your_api_key
AWS_ACCESS_KEY=your_key
```

Load in Python:
```python
from dotenv import load_dotenv
import os

load_dotenv()
db_url = os.getenv('DATABASE_URL')
```

### Performance Optimization

#### 1. Model Caching

```python
import streamlit as st
import joblib

@st.cache_resource
def load_model():
    return joblib.load('models/churn_model.pkl')

model = load_model()
```

#### 2. Data Caching

```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_customer_data():
    return pd.read_csv('data/data_with_churn_predictions.csv')
```

#### 3. Database Integration

For production, store data in database:

```python
from sqlalchemy import create_engine

engine = create_engine(os.getenv('DATABASE_URL'))

@st.cache_data(ttl=600)
def load_data_from_db():
    return pd.read_sql("SELECT * FROM customers", engine)
```

### Scalability

#### Load Balancing

Use multiple instances behind load balancer:

```yaml
# docker-compose.yml for multiple instances
version: '3.8'
services:
  dashboard-1:
    build: .
    ports:
      - "8501:8501"
  
  dashboard-2:
    build: .
    ports:
      - "8502:8501"
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - dashboard-1
      - dashboard-2
```

---

## Monitoring & Maintenance

### Application Monitoring

#### 1. Health Checks

Add health check endpoint:

```python
# health.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}
```

#### 2. Logging

Configure comprehensive logging:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info('Dashboard started')
```

#### 3. Error Tracking

Use Sentry for error tracking:

```python
import sentry_sdk

sentry_sdk.init(
    dsn="your-sentry-dsn",
    traces_sample_rate=1.0
)
```

### Model Retraining

Schedule periodic model retraining:

```bash
# cron job example (Linux)
0 2 * * 0 cd /app && python scripts/run_pipeline.py >> /var/log/brukd/retrain.log 2>&1
```

Or use Apache Airflow for workflow management:

```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

dag = DAG('brukd_model_retrain', schedule_interval='@weekly')

retrain_task = PythonOperator(
    task_id='retrain_models',
    python_callable=run_pipeline,
    dag=dag
)
```

### Backup Strategy

```bash
# Backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)

# Backup models
tar -czf backups/models_$DATE.tar.gz models/

# Backup data
tar -czf backups/data_$DATE.tar.gz data/

# Upload to S3
aws s3 cp backups/ s3://brukd-backups/customer-engagement/ --recursive
```

---

## CI/CD Pipeline

### GitHub Actions Example

```yaml
# .github/workflows/deploy.yml
name: Build and Deploy

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest
    
    - name: Run tests
      run: pytest tests/
    
    - name: Build Docker image
      run: docker build -t brukd-customer-engagement .
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push brukd-customer-engagement:latest
    
    - name: Deploy to production
      run: |
        # Your deployment commands
```

---

## Troubleshooting

### Common Issues

#### Issue: Port already in use
```bash
# Find process using port 8501
lsof -i :8501

# Kill process
kill -9 <PID>
```

#### Issue: Out of memory
```bash
# Increase Docker memory limit
docker run --memory=4g brukd-customer-engagement
```

#### Issue: Slow loading
- Enable caching with `@st.cache_data`
- Use smaller data samples for development
- Optimize database queries with indexes

---

## Support & Resources

- **Documentation:** [README.md](../README.md)
- **Blog Post:** [BLOG_POST.md](BLOG_POST.md)
- **Issues:** [GitHub Issues](https://github.com/brukd/ai-customer-engagement/issues)
- **Email:** support@brukd.com

---

## Summary Checklist

- [ ] Local development environment setup
- [ ] Run complete analysis pipeline
- [ ] Test dashboard locally
- [ ] Configure Docker deployment
- [ ] Set up cloud hosting (AWS/GCP/Azure)
- [ ] Implement authentication and security
- [ ] Configure monitoring and logging
- [ ] Set up automated backups
- [ ] Establish CI/CD pipeline
- [ ] Document customizations
- [ ] Train team on usage and maintenance

---

*© 2025 Brukd. For production deployment support, contact our team at deploy@brukd.com*

