-- 04_score_customers.sql
-- Runs ML.PREDICT to score every customer in rfm_scores
-- Writes churn probability, risk tier, and RFM segment to bqml_churn_scores table

CREATE OR REPLACE TABLE `customer-churn-492703.customer_intelligence.bqml_churn_scores` AS
SELECT
  r.user_id,
  r.recency_days,
  r.frequency,
  r.monetary,
  r.r_score,
  r.f_score,
  r.m_score,
  r.traffic_source,
  r.country,
  r.age,
  p.predicted_churned                                            AS churn_prediction,
  (SELECT prob FROM UNNEST(p.predicted_churned_probs)
   WHERE label = 1)                                             AS churn_probability,
  CASE
    WHEN (SELECT prob FROM UNNEST(p.predicted_churned_probs)
          WHERE label = 1) >= 0.7 THEN 'High'
    WHEN (SELECT prob FROM UNNEST(p.predicted_churned_probs)
          WHERE label = 1) >= 0.4 THEN 'Medium'
    ELSE 'Low'
  END                                                           AS churn_risk,
  CASE
    WHEN r.r_score >= 4 AND r.f_score >= 4 THEN 'Champions'
    WHEN r.r_score >= 3 AND r.f_score >= 3 THEN 'Loyal'
    WHEN r.r_score >= 4 AND r.f_score <= 2 THEN 'Recent'
    WHEN r.r_score <= 2 AND r.f_score >= 3 THEN 'At-Risk'
    WHEN r.r_score <= 2 AND r.f_score <= 2 THEN 'Hibernating'
    ELSE 'Potential'
  END                                                           AS rfm_segment,
  CURRENT_TIMESTAMP()                                           AS scored_at
FROM `customer-churn-492703.customer_intelligence.rfm_scores` r
JOIN ML.PREDICT(
  MODEL `customer-churn-492703.customer_intelligence.bqml_churn_lr`,
  (SELECT * FROM `customer-churn-492703.customer_intelligence.rfm_scores`)
) p
ON r.user_id = p.user_id