"""Split the raw customer churn CSV into normalized seed files for dbt.

In production, these tables would arrive from separate source systems
(CRM, provisioning, billing, retention). Here we normalize the single
denormalized CSV to mirror that structure.

Output: dbt_churn/seeds/raw_customers.csv
        dbt_churn/seeds/raw_services.csv
        dbt_churn/seeds/raw_billing.csv
        dbt_churn/seeds/raw_churn_labels.csv
"""

import pandas as pd
from pathlib import Path

RAW_PATH = "data/raw/Customer-Churn.csv"
SEED_DIR = Path("dbt_churn/seeds")
SEED_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW_PATH)
print(f"Loaded {len(df):,} rows from {RAW_PATH}")

# Normalize column names to snake_case for dbt convention
df.columns = (
    df.columns
    .str.strip()
    .str.replace(r"(?<=[a-z])(?=[A-Z])", "_", regex=True)
    .str.lower()
)

customers = df[["customer_id", "gender", "senior_citizen", "partner", "dependents", "tenure"]]
customers.to_csv(SEED_DIR / "raw_customers.csv", index=False)
print(f"  raw_customers.csv  — {len(customers)} rows, {len(customers.columns)} cols")

service_cols = [
    "customer_id", "phone_service", "multiple_lines", "internet_service",
    "online_security", "online_backup", "device_protection", "tech_support",
    "streaming_tv", "streaming_movies",
]
services = df[service_cols]
services.to_csv(SEED_DIR / "raw_services.csv", index=False)
print(f"  raw_services.csv   — {len(services)} rows, {len(services.columns)} cols")

billing = df[["customer_id", "contract", "paperless_billing", "payment_method", "monthly_charges", "total_charges"]]
billing.to_csv(SEED_DIR / "raw_billing.csv", index=False)
print(f"  raw_billing.csv    — {len(billing)} rows, {len(billing.columns)} cols")

churn_labels = df[["customer_id", "churn"]]
churn_labels.to_csv(SEED_DIR / "raw_churn_labels.csv", index=False)
print(f"  raw_churn_labels.csv — {len(churn_labels)} rows, {len(churn_labels.columns)} cols")

print(f"\nDone. Seeds written to {SEED_DIR}/")
