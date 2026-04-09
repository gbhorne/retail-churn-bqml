# retail-churn-bqml

Retail customer churn prediction using BigQuery ML logistic regression.
Generates 200,000 synthetic retail customers across six behavioral segments,
trains a logistic regression classifier entirely in SQL using BigQuery ML,
and scores all customers with churn probability and RFM segment labels.

Companion repo: https://github.com/gbhorne/retail-churn-pytorch

---

## Churn label definition

| Property | Value |
|----------|-------|
| Observation window | Full synthetic purchase history per customer |
| Prediction horizon | N/A (synthetic labels assigned at generation time) |
| Churn rule | Probabilistic per segment (see table below) |
| Scoring timestamp | End of observation window |

Churn labels are assigned probabilistically per segment during synthetic
data generation, not derived deterministically from any single feature.
This avoids the leakage pattern where recency_days directly determines
the churn label.

| Segment | Churn probability |
|---------|------------------|
| Champions | 5% |
| Loyal | 10% |
| Potential | 40% |
| Recent | 25% |
| At-Risk | 65% |
| Hibernating | 85% |

In a production deployment, churn would be defined as no purchase within
a fixed horizon (e.g. 90 days) after a feature cutoff date. Features must
be computed strictly before that cutoff to prevent leakage:

```sql
-- Production pattern: enforce feature cutoff before label window
-- feature_cutoff = label_date - 90 days
-- All features computed from events WHERE event_date < feature_cutoff
-- Churn label = no purchase between feature_cutoff and label_date
WITH feature_cutoff AS (
  SELECT DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY) AS cutoff
)
SELECT
  user_id,
  COUNT(DISTINCT order_id) AS frequency,
  SUM(sale_price)          AS monetary
FROM orders
WHERE order_date < (SELECT cutoff FROM feature_cutoff)
GROUP BY user_id
```

---

## Results (reference run)

| Model | ROC-AUC | Precision | Recall | F1 | Training time |
|-------|---------|-----------|--------|----|---------------|
| BQML logistic regression | 0.826 | 0.812 | 0.654 | 0.725 | 86s |

Verify from scratch using the reproducible evaluation queries below.

---

## Data split note

This pipeline uses RANDOM data split (data_split_method = RANDOM).
In a production churn model with real transaction data, a time-based
split must be used to ensure the model is evaluated on future data
it has never seen during training:

```sql
-- Production pattern: time-based split
-- Train on data before cutoff, evaluate on data after
CREATE OR REPLACE MODEL `project.dataset.churn_model`
OPTIONS(
  model_type = 'LOGISTIC_REG',
  data_split_method = 'CUSTOM',
  data_split_col = 'is_eval'
)
AS SELECT
  *,
  IF(last_order_date >= '2024-01-01', TRUE, FALSE) AS is_eval
FROM `project.dataset.rfm_scores`
```

RANDOM split is used here because the synthetic dataset has no temporal
ordering that would make time-based splitting meaningful.

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

## Cost profile

| Component | Cost model |
|-----------|-----------|
| Data generation | One-time BigQuery load job, minimal cost |
| Model training | BigQuery on-demand: billed per TB of training data scanned |
| Batch scoring | ML.PREDICT billed per TB scanned |
| Storage | rfm_scores ~6.7MB, bqml_churn_scores ~8MB, negligible cost |
| Infrastructure | None (fully serverless) |

BQML eliminates all infrastructure management. There are no VMs, containers,
or serving layers to maintain. Scoring runs as a SQL query on a schedule.

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

## Synthetic data disclaimer

All customer data is synthetically generated using numpy random distributions.
No real customer records, PII, or proprietary retail data was used.
The data simulates realistic RFM behavioral patterns for ML pipeline demonstration.

---

## License

MIT