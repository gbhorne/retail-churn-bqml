-- 02_create_churn_model.sql
-- Trains a logistic regression churn classifier inside BigQuery using BQML
-- Reads from rfm_scores, stores model artifact in customer_intelligence dataset

CREATE OR REPLACE MODEL `customer-churn-492703.customer_intelligence.bqml_churn_lr`
OPTIONS(
  model_type               = 'LOGISTIC_REG',
  input_label_cols         = ['churned'],
  auto_class_weights       = TRUE,
  data_split_method        = 'RANDOM',
  data_split_eval_fraction = 0.2,
  max_iterations           = 50,
  l1_reg                   = 0.1,
  l2_reg                   = 0.1
) AS
SELECT
  recency_days,
  frequency,
  monetary,
  avg_order_value,
  active_months,
  r_score,
  f_score,
  m_score,
  age,
  traffic_source,
  churned
FROM `customer-churn-492703.customer_intelligence.rfm_scores`