# retail-churn-bqml

Retail customer churn prediction using BigQuery ML logistic regression.
Generates 200,000 synthetic retail customers across six behavioral segments,
trains a logistic regression classifier in BigQuery ML, and scores all
customers with churn probability and RFM segment labels.

Companion repo: https://github.com/gbhorne/retail-churn-pytorch

---

## Churn label definition

A customer is labeled as churned (1) if their churn probability exceeds the
segment-level threshold assigned during synthetic data generation.
Churn labels are assigned probabilistically per segment, not derived
deterministically from recency alone. This avoids the leakage pattern where
the label is a direct function of a training feature.

| Segment | Assigned churn probability |
|---------|--------------------------|
| Champions | 5% |
| Loyal | 10% |
| Potential | 40% |
| Recent | 25% |
| At-Risk | 65% |
| Hibernating | 85% |

In a production system, features would be computed over a historical window
strictly before the label observation period to prevent leakage.

---

## Results (reference run)

| Model | ROC-AUC | Precision | Recall | F1 | Training time |
|-------|---------|-----------|--------|----|---------------|
| BQML logistic regression | 0.826 | 0.812 | 0.654 | 0.725 | 86s |

These numbers are from a single reference run.
Results may vary across data refreshes and BigQuery slot availability.

---

## Data split note

This pipeline uses RANDOM data split (data_split_method = RANDOM).
In a production churn model, a time-based split should be used to ensure
the model is evaluated on future data it has never seen during training.
RANDOM split is used here for simplicity on a synthetic dataset with no
temporal ordering.

---

## Setup

### 1. Clone and install

```
git clone https://github.com/gbhorne/retail-churn-bqml.git
cd retail-churn-bqml
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Authenticate to GCP

```
gcloud auth application-default login
gcloud auth login
gcloud config set project customer-churn-492703
gcloud auth application-default set-quota-project customer-churn-492703
```

### 3. Generate synthetic data

```
python generate_synthetic_data.py
```

### 4. Run SQL pipeline in order

```
.\bq_run.ps1 sql\01_rfm_features.sql
.\bq_run.ps1 sql\02_create_churn_model.sql
.\bq_run.ps1 sql\03_evaluate_model.sql
.\bq_run.ps1 sql\04_score_customers.sql
.\bq_run.ps1 sql\05_validate_scores.sql
```

---

## Pipeline

| File | Purpose |
|------|---------|
| generate_synthetic_data.py | Generates 200,000 synthetic customers, writes to BigQuery |
| sql/01_rfm_features.sql | Validates rfm_scores table |
| sql/02_create_churn_model.sql | Trains BQML logistic regression |
| sql/03_evaluate_model.sql | Evaluates model: AUC, precision, recall |
| sql/04_score_customers.sql | Scores all customers, writes bqml_churn_scores |
| sql/05_validate_scores.sql | Validates scored output by segment and risk tier |

---

## Reproducible evaluation queries

Run these in BigQuery to verify model performance from scratch:

```sql
SELECT * FROM ML.EVALUATE(
  MODEL `customer-churn-492703.customer_intelligence.bqml_churn_lr`
)
```

```sql
SELECT * FROM ML.ROC_CURVE(
  MODEL `customer-churn-492703.customer_intelligence.bqml_churn_lr`
)
ORDER BY threshold DESC
```

---

## Synthetic data disclaimer

All customer data is synthetically generated using numpy random distributions.
No real customer records, PII, or proprietary retail data was used.
The data simulates realistic RFM behavioral patterns for ML pipeline demonstration.

---

## License

MIT