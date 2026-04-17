# Customer Churn Dataset — Data Card

What the dataset is, where it came from, what's faithful about it, and where a reader should be skeptical. Read this before taking any metric in this repo at face value.

## What this is

The public IBM customer churn dataset. One CSV, 7,043 rows, one row per customer, one binary target (`Churn`). Commonly distributed on Kaggle under names like "Telco Customer Churn" or "WA_Fn-UseC_-Telco-Customer-Churn" — search that phrase on Kaggle to locate the raw file. It's IBM sample data, not a raw extract from a live billing system.

The project uses this dataset as a **public benchmark** — well-characterized, reproducible, and small enough to iterate on. Nothing about the project assumes the data is live, current, or representative of any specific operator.

## Shape and schema

- **Rows:** 7,043 customers (one snapshot, no time dimension)
- **Columns:** 21 total — 1 ID, 1 target (`Churn`), 19 features
- **Target balance:** ~26.5% churn (`Churn = Yes`), ~73.5% retained
- **Feature groups:**
  - Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - Account tenure and billing: `tenure` (months), `MonthlyCharges`, `TotalCharges`, `Contract`, `PaperlessBilling`, `PaymentMethod`
  - Services subscribed: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`

## What's faithful

- **Schema.** The column set looks like what a subscription operator's customer table would expose to an analyst — demographics, services, billing, contract type. Semantically plausible.
- **Within-feature distributions.** Tenure skews short (a lot of new customers), `MonthlyCharges` has the bimodal shape you'd expect from basic vs. bundled plans, and churn concentrates on month-to-month contracts and customers with short tenure. An analyst in the industry would recognize these patterns.
- **The core business signal.** Contract type, tenure, and service bundle really are the dominant churn drivers in subscription businesses. That part of the story transfers.

## What's synthetic or curated

- **This is IBM sample data, not a raw dump.** It's been cleaned, sampled, and balanced well enough to teach with. Edge cases, data-entry noise, and the ugly parts of a real warehouse aren't in it.
- **No customer ID lineage.** Each row is a standalone snapshot. No way to trace a customer's journey (plan changes, upgrades, complaint history) over time.
- **Dollar amounts are as-of whenever IBM published this.** `MonthlyCharges` and `TotalCharges` are plausible subscription-service numbers, not inflation-adjusted, not representative of any 2026 operator's pricing.

## Known quirks worth calling out

### `TotalCharges` is derived from `tenure` × `MonthlyCharges`

For the vast majority of rows, `TotalCharges ≈ tenure × MonthlyCharges`. Using all three as independent features invites multicollinearity and — depending on how you frame the target — mild leakage. The training pipeline in this repo keeps all three because tree-based models tolerate redundancy, but a linear model would want one dropped.

### `TotalCharges` has blank strings for tenure-0 customers

A small number of rows have an empty-string `TotalCharges`. These are brand-new customers (`tenure = 0`) who haven't been billed yet. The preprocessing step in `src/data/preprocess.py` converts this to `NaN` and imputes. Worth knowing if you ever load the raw CSV yourself.

### Some categorical levels overlap semantically

`PhoneService = No` implies `MultipleLines = "No phone service"`; `InternetService = No` implies every internet add-on (`OnlineSecurity`, `OnlineBackup`, etc.) is `"No internet service"`. These are redundant encodings of the same fact. The one-hot features derived from them are correlated by construction.

### Class imbalance is mild, not severe

26.5% positive rate means the majority class isn't overwhelming. `scale_pos_weight` in XGBoost handles it cleanly; you don't need SMOTE or aggressive resampling for this dataset. On a real operator's data, churn rates are often much lower (1-5% annualized), which would change the modeling approach.

## What does NOT reproduce on real operational data

If you lifted this project and pointed it at a real operator's warehouse, these things would be different and would need addressing:

- **Temporal drift.** The snapshot has no time structure. Real churn models need time-split training/validation (train on pre-date, evaluate on post-date) because feature distributions and churn base rates drift. Cross-validation on a static snapshot is optimistic.
- **Seasonality.** Real churn spikes around contract renewal windows, holiday periods, price increases, and competitor promotions. None of that signal is in this dataset.
- **Campaign history.** Real retention teams send offers. Those interventions change who churns. The dataset has no treatment column, no campaign history, no "who got a discount last quarter." This is the main reason genuine **uplift / causal** modeling on this dataset requires synthesizing a treatment effect.
- **Lifecycle events.** Customers upgrade, downgrade, pause, resume, transfer accounts. All of that is flattened into a single snapshot here.
- **Missingness patterns.** Real billing data has systematic missingness (feature-not-collected-for-older-customers, partner data gaps, etc.). This dataset has almost none.
- **Label definition.** Real "churn" is operationally fuzzy: voluntary disconnect vs. involuntary (non-payment) vs. downgrade-to-nothing vs. port-out. IBM's label is a clean binary. Real teams spend weeks arguing about the definition.
- **Revenue figures.** The "revenue at risk" numbers in `notebooks/01_business_analysis.ipynb` multiply `MonthlyCharges × 12` to get annual revenue per customer. That's gross, not margin. On a real P&L you'd subtract cost-to-serve, support overhead, and other variable costs. The dollar numbers in this repo are directional, not finance-ready.

## Assumptions baked into specific artifacts

Some claims in the notebooks and README depend on assumptions that aren't in the dataset itself:

| Artifact | Number | Source |
|---|---|---|
| `notebooks/02_model_analysis.ipynb` | $50 retention offer cost | Industry benchmark, not in data |
| `notebooks/02_model_analysis.ipynb` | $1,554 LTV loss per churn | Derived from dataset (avg `MonthlyCharges × 12 × retention factor`), not a finance system |
| `notebooks/02_model_analysis.ipynb` | 40% save rate given an offer | Industry benchmark, not estimated from data |
| `notebooks/01_business_analysis.ipynb` | "$1.67M annualized revenue at risk" | `MonthlyCharges × 12 × P(churn)`, pre-margin |
| `notebooks/01_business_analysis.ipynb` | Top-100 at-risk customer list | Out-of-fold XGBoost predictions on the dataset's own customers |

If any of these assumptions change, the dollar figures move. The methodology is the load-bearing part, not the specific numbers.

## Reproducing the dataset

- Official source: IBM sample data, available on Kaggle. See the opening section for the search term that locates the raw file.
- The raw file belongs in `data/raw/` under a consistent filename (the training script and notebooks default to `data/raw/Customer-Churn.csv`).
- `data/raw/` is gitignored — the dataset isn't checked in.

## When to stop trusting this card

This card describes the dataset as of the last time these notebooks were rerun. If you retrain on a different vintage, a filtered subset, or a different labeling convention, update the relevant sections before shipping new numbers in the README or notebooks.

Rule of thumb: if a claim in the repo has a dollar sign or a percentage in it, it ultimately traces back to something in this file. When the claim changes, the card changes too.
