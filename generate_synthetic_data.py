# generate_synthetic_data.py
# Generates 200,000 synthetic retail customers with realistic RFM features
# and a probabilistic churn label, then writes directly to BigQuery

import numpy as np
import pandas as pd
from google.cloud import bigquery

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

PROJECT   = "customer-churn-492703"
DATASET   = "customer_intelligence"
TABLE     = "rfm_scores"
N         = 200_000

# ── segment definitions ───────────────────────────────────────────────────────
# Each segment has a share of the total population and a churn probability.
# RFM distributions are tuned so the model has to learn real patterns,
# not just read off a single feature.
SEGMENTS = [
    {
        "name":         "Hibernating",
        "share":        0.35,        # 35% of customers
        "churn_prob":   0.85,        # 85% are churned
        "recency_mu":   400,         # avg days since last order
        "recency_sd":   120,
        "freq_mu":      1.5,
        "freq_sd":      0.7,
        "monetary_mu":  85,
        "monetary_sd":  40,
    },
    {
        "name":         "At-Risk",
        "share":        0.20,
        "churn_prob":   0.65,
        "recency_mu":   150,
        "recency_sd":   50,
        "freq_mu":      3.0,
        "freq_sd":      1.2,
        "monetary_mu":  180,
        "monetary_sd":  70,
    },
    {
        "name":         "Potential",
        "share":        0.15,
        "churn_prob":   0.40,
        "recency_mu":   60,
        "recency_sd":   25,
        "freq_mu":      2.0,
        "freq_sd":      0.8,
        "monetary_mu":  120,
        "monetary_sd":  50,
    },
    {
        "name":         "Recent",
        "share":        0.10,
        "churn_prob":   0.25,
        "recency_mu":   20,
        "recency_sd":   12,
        "freq_mu":      1.2,
        "freq_sd":      0.4,
        "monetary_mu":  95,
        "monetary_sd":  35,
    },
    {
        "name":         "Loyal",
        "share":        0.12,
        "churn_prob":   0.10,
        "recency_mu":   30,
        "recency_sd":   15,
        "freq_mu":      8.0,
        "freq_sd":      2.5,
        "monetary_mu":  420,
        "monetary_sd":  130,
    },
    {
        "name":         "Champions",
        "share":        0.08,
        "churn_prob":   0.05,
        "recency_mu":   12,
        "recency_sd":   6,
        "freq_mu":      14.0,
        "freq_sd":      4.0,
        "monetary_mu":  850,
        "monetary_sd":  220,
    },
]

TRAFFIC_SOURCES = ["Organic", "Search", "Email", "Social", "Display"]
COUNTRIES       = ["United States", "United Kingdom", "Canada", "Australia",
                   "Germany", "France", "Brazil", "Japan", "India", "Mexico"]

# ── generate customers ────────────────────────────────────────────────────────
rows = []
user_id = 1

for seg in SEGMENTS:
    n_seg = int(N * seg["share"])

    recency  = np.clip(np.random.normal(seg["recency_mu"],  seg["recency_sd"],  n_seg), 1, 1825).astype(int)
    freq     = np.clip(np.random.normal(seg["freq_mu"],     seg["freq_sd"],     n_seg), 1, 50).astype(int)
    monetary = np.clip(np.random.normal(seg["monetary_mu"], seg["monetary_sd"], n_seg), 1, 5000)
    monetary = np.round(monetary, 2)
    avg_order_value = np.round(monetary / freq, 2)
    active_months   = np.clip(freq // 2 + np.random.randint(1, 4, n_seg), 1, 24).astype(int)
    age             = np.random.randint(18, 75, n_seg)
    traffic         = np.random.choice(TRAFFIC_SOURCES, n_seg)
    country         = np.random.choice(COUNTRIES, n_seg)

    # probabilistic churn label — not deterministic from recency
    churned = (np.random.rand(n_seg) < seg["churn_prob"]).astype(int)

    # RFM scores (1-5): higher recency_days = worse = lower r_score
    r_score = pd.cut(recency, bins=5, labels=[5, 4, 3, 2, 1]).astype(int)
    f_score = pd.cut(freq,    bins=5, labels=[1, 2, 3, 4, 5]).astype(int)
    m_score = pd.cut(monetary,bins=5, labels=[1, 2, 3, 4, 5]).astype(int)

    for i in range(n_seg):
        rows.append({
            "user_id":         user_id,
            "segment":         seg["name"],
            "traffic_source":  traffic[i],
            "country":         country[i],
            "age":             int(age[i]),
            "recency_days":    int(recency[i]),
            "frequency":       int(freq[i]),
            "monetary":        float(monetary[i]),
            "avg_order_value": float(avg_order_value[i]),
            "active_months":   int(active_months[i]),
            "r_score":         int(r_score[i]),
            "f_score":         int(f_score[i]),
            "m_score":         int(m_score[i]),
            "churned":         int(churned[i]),
        })
        user_id += 1

df = pd.DataFrame(rows)

# ── report churn split ────────────────────────────────────────────────────────
churn_counts = df["churned"].value_counts()
total        = len(df)
print(f"Generated {total:,} customers")
print(f"  Churned:     {churn_counts[1]:,} ({churn_counts[1]/total*100:.1f}%)")
print(f"  Not churned: {churn_counts[0]:,} ({churn_counts[0]/total*100:.1f}%)")
print(f"\nSegment breakdown:")
print(df.groupby("segment")["churned"].agg(["count", "mean"]).round(2))

# ── write to BigQuery ─────────────────────────────────────────────────────────
client = bigquery.Client(project=PROJECT)
table_ref = f"{PROJECT}.{DATASET}.{TABLE}"

job_config = bigquery.LoadJobConfig(
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    schema=[
        bigquery.SchemaField("user_id",         "INTEGER"),
        bigquery.SchemaField("segment",          "STRING"),
        bigquery.SchemaField("traffic_source",   "STRING"),
        bigquery.SchemaField("country",          "STRING"),
        bigquery.SchemaField("age",              "INTEGER"),
        bigquery.SchemaField("recency_days",     "INTEGER"),
        bigquery.SchemaField("frequency",        "INTEGER"),
        bigquery.SchemaField("monetary",         "FLOAT"),
        bigquery.SchemaField("avg_order_value",  "FLOAT"),
        bigquery.SchemaField("active_months",    "INTEGER"),
        bigquery.SchemaField("r_score",          "INTEGER"),
        bigquery.SchemaField("f_score",          "INTEGER"),
        bigquery.SchemaField("m_score",          "INTEGER"),
        bigquery.SchemaField("churned",          "INTEGER"),
    ]
)

print(f"\nWriting to BigQuery: {table_ref} ...")
job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
job.result()
print(f"Done. {table_ref} now has {len(df):,} rows.")