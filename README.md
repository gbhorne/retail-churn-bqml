# retail-churn-bqml

Retail customer churn prediction using BigQuery ML. Generates 200,000 synthetic
retail customers across six behavioral segments, trains a logistic regression
classifier in BigQuery ML, and scores all customers with churn probability and
RFM segment labels.

## Results
- Dataset: 200,000 synthetic customers, 53/47 churn split
- Model: BQML Logistic Regression
- ROC-AUC: 0.826
- Precision: 0.812 | Recall: 0.654 | F1: 0.725

## GCP Setup
- Project: customer-churn-492703
- Dataset: customer_intelligence
- Tables: rfm_scores, bqml_churn_scores
- Model: bqml_churn_lr

## Quick Start

### 1. Clone and install
git clone https://github.com/gbhorne/retail-churn-bqml.git
cd retail-churn-bqml
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

### 2. Authenticate
gcloud auth application-default login
gcloud config set project customer-churn-492703
gcloud auth application-default set-quota-project customer-churn-492703

### 3. Generate synthetic data
python generate_synthetic_data.py

### 4. Run SQL pipeline in order
.\bq_run.ps1 sql\01_rfm_features.sql
.\bq_run.ps1 sql\02_create_churn_model.sql
.\bq_run.ps1 sql\03_evaluate_model.sql
.\bq_run.ps1 sql\04_score_customers.sql
.\bq_run.ps1 sql\05_validate_scores.sql

## Pipeline
| File | Purpose |
|------|---------|
| generate_synthetic_data.py | Generates 200k synthetic customers and writes to BigQuery |
| sql/01_rfm_features.sql | Validates rfm_scores table |
| sql/02_create_churn_model.sql | Trains BQML logistic regression |
| sql/03_evaluate_model.sql | Evaluates model — AUC, precision, recall |
| sql/04_score_customers.sql | Scores all customers, writes bqml_churn_scores |
| sql/05_validate_scores.sql | Validates scored output by segment and risk tier |