-- 03_evaluate_model.sql
-- Evaluates the trained BQML logistic regression model against the held-out 20% eval set
-- Key metrics: roc_auc, precision, recall, f1_score

SELECT
  ROUND(precision, 4)  AS precision,
  ROUND(recall, 4)     AS recall,
  ROUND(accuracy, 4)   AS accuracy,
  ROUND(f1_score, 4)   AS f1_score,
  ROUND(log_loss, 4)   AS log_loss,
  ROUND(roc_auc, 4)    AS roc_auc
FROM ML.EVALUATE(
  MODEL `customer-churn-492703.customer_intelligence.bqml_churn_lr`
)