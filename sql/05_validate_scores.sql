-- 05_validate_scores.sql
-- Sanity check on bqml_churn_scores output table

SELECT
  rfm_segment,
  churn_risk,
  COUNT(*)                          AS customers,
  ROUND(AVG(churn_probability), 3)  AS avg_churn_prob,
  ROUND(AVG(monetary), 2)           AS avg_spend
FROM `customer-churn-492703.customer_intelligence.bqml_churn_scores`
GROUP BY rfm_segment, churn_risk
ORDER BY rfm_segment, churn_risk