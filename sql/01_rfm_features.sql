-- 01_rfm_features.sql
-- Reads TheLook public dataset, computes one row per customer with RFM features
-- and a binary churn label, writes to customer-churn-492703.customer_intelligence.rfm_scores

CREATE OR REPLACE TABLE `customer-churn-492703.customer_intelligence.rfm_scores` AS

WITH order_agg AS (
  SELECT
    o.user_id,
    u.traffic_source,
    u.country,
    u.age,
    MAX(o.created_at)                                             AS last_order_date,
    COUNT(DISTINCT o.order_id)                                    AS frequency,
    SUM(oi.sale_price)                                            AS monetary,
    AVG(oi.sale_price)                                            AS avg_order_value,
    COUNT(DISTINCT DATE_TRUNC(o.created_at, MONTH))               AS active_months
  FROM `bigquery-public-data.thelook_ecommerce.orders` o
  JOIN `bigquery-public-data.thelook_ecommerce.order_items` oi
    ON o.order_id = oi.order_id
  JOIN `bigquery-public-data.thelook_ecommerce.users` u
    ON o.user_id = u.id
  WHERE o.status NOT IN ('Cancelled', 'Returned')
  GROUP BY o.user_id, u.traffic_source, u.country, u.age
)

SELECT
  user_id,
  traffic_source,
  country,
  age,
  DATE_DIFF(CURRENT_DATE(), DATE(last_order_date), DAY)           AS recency_days,
  frequency,
  ROUND(monetary, 2)                                              AS monetary,
  ROUND(avg_order_value, 2)                                       AS avg_order_value,
  active_months,
  NTILE(5) OVER (ORDER BY DATE_DIFF(CURRENT_DATE(), DATE(last_order_date), DAY) ASC) AS r_score,
  NTILE(5) OVER (ORDER BY frequency ASC)                          AS f_score,
  NTILE(5) OVER (ORDER BY monetary ASC)                           AS m_score,
  IF(DATE_DIFF(CURRENT_DATE(), DATE(last_order_date), DAY) > 90, 1, 0) AS churned
FROM order_agg