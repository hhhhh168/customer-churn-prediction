# Customer Churn Prediction

XGBoost churn model for subscription-service customers, served behind a FastAPI + Gradio app on AWS ECS Fargate. Built as a personal project to get an end-to-end ML system into production rather than just a notebook.

## Analysis

Three notebooks, written up as case studies rather than scratchpads. They walk through the descriptive → predictive → prescriptive decision stack a retention team would actually use.

**[`notebooks/01_business_analysis.ipynb`](notebooks/01_business_analysis.ipynb)** — descriptive:

- Segmentation by contract, tenure bucket, payment method, and service bundle, each tied to annual revenue-at-risk in dollars
- Unified revenue-at-risk ranking across segmentation axes
- Ranked list of the top 100 currently-active customers the retention team should call first, scored with honest out-of-fold XGBoost predictions
- Retention action recommendations mapped to the segments that surface from the analysis

**[`notebooks/02_model_analysis.ipynb`](notebooks/02_model_analysis.ipynb)** — predictive:

- Three-model comparison (logistic regression, random forest, XGBoost) with 5-fold stratified cross-validation
- Calibration curve against observed frequencies
- SHAP global feature importance plus individual explanations for the highest- and lowest-risk customers in the test set
- Business-grounded threshold tuning: instead of picking 0.5 or 0.35 by intuition, the classification threshold is chosen to maximize expected net savings under a simple cost/benefit model ($50 retention offer, ~$1,550 lifetime value loss per churn, 40% save rate)
- Results, recommendations for the retention team, and limitations

**[`notebooks/03_uplift_modeling.ipynb`](notebooks/03_uplift_modeling.ipynb)** — prescriptive:

- Frames retention as a treatment-effect problem: who's actually *persuadable* by an offer, not just who's most at risk
- Dataset has no treatment column, so the notebook is explicit about being a methodology demonstration with a synthesized treatment effect on top of the observed data. See the first markdown cell and `docs/data_card.md` for the honest framing
- T-learner and X-learner uplift models via `causalml`, evaluated with decile analysis, Qini curve, and AUUC against risk-based and random targeting baselines
- Budget-constrained comparison: given a fixed-size retention campaign, how much better is uplift targeting than ranking by churn risk?

**[Tableau Public dashboard](https://public.tableau.com/views/CustomerChurnPrediction_1/Dashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)** — interactive version of the business analysis, built for stakeholder consumption. Segmentation charts cross-filter on click, and a budget parameter lets you adjust how many at-risk customers the retention team can reach.

The methodology generalizes to any subscription or recurring-revenue retention problem — SaaS churn, neobank attrition, subscription-fintech app churn, B2B payment platform retention, membership businesses, and customer revenue analytics more broadly. The data here is public telco churn, but the segmentation axes, revenue-at-risk framing, and top-N outreach pipeline aren't telco-specific. `notebooks/01_business_analysis.ipynb` section 9 has a direct feature-by-feature mapping, and `notebooks/02_model_analysis.ipynb` section 5 includes an adverse-action-style reason-code treatment for regulated explainability contexts.

## Stack

- **Model**: XGBoost, trained on a public customer churn dataset (see `docs/data_card.md`)
- **Tracking**: MLflow (file-based, runs stored under `mlruns/`)
- **Validation**: Great Expectations on the raw input
- **API**: FastAPI (`/predict`, `/`) + a Gradio UI mounted at `/ui`
- **Container**: Docker, uvicorn on port 8000
- **CI**: GitHub Actions builds the image and pushes to Docker Hub on every push to `main`
- **Hosting**: AWS ECS Fargate behind an Application Load Balancer

## How requests flow in production

```
client -> ALB :80  ->  ECS task :8000  ->  FastAPI -> XGBoost model -> JSON
```

The model artifacts (`model.pkl`, feature column list, preprocessing metadata) are baked into the image at build time, so the container starts up self-contained — no S3 pull, no model registry call.

## Run it locally

```bash
# train
python scripts/run_pipeline.py --input data/raw/Customer-Churn.csv --target Churn

# serve
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000/ui
```

Or with Docker:

```bash
docker build -t customer-churn-app .
docker run -p 8000:8000 customer-churn-app
```

## Hit the API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "Partner": "No", "Dependents": "No",
    "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "tenure": 1, "MonthlyCharges": 85.0, "TotalCharges": 85.0
  }'
```

Returns `{"prediction": "Likely to churn"}` or `{"prediction": "Not likely to churn"}`.

## Notes

- The classification threshold is tuned via cost/benefit analysis rather than picked by intuition — see `notebooks/02_model_analysis.ipynb` for the derivation. The serving code currently ships a fixed threshold as a starting point; the notebook is the source of truth for how it should be chosen.
- Class imbalance is handled with `scale_pos_weight` in XGBoost rather than resampling.
- Training and serving use the same feature transformations — see `src/features/build_features.py` and `src/serving/inference.py`. If you change one, change the other.
