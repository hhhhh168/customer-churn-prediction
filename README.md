# Customer Churn Prediction

XGBoost churn model for telecom customers, served behind a FastAPI + Gradio app on AWS ECS Fargate. Built as a personal project to get an end-to-end ML system into production rather than just a notebook.

## Analysis

Two analytical notebooks, written up as case studies rather than scratchpads:

**[`notebooks/02_model_analysis.ipynb`](notebooks/02_model_analysis.ipynb)** — modeling:

- Three-model comparison (logistic regression, random forest, XGBoost) with 5-fold stratified cross-validation
- Calibration curve against observed frequencies
- SHAP global feature importance plus individual explanations for the highest- and lowest-risk customers in the test set
- Business-grounded threshold tuning: instead of picking 0.5 or 0.35 by intuition, the classification threshold is chosen to maximize expected net savings under a simple cost/benefit model ($50 retention offer, ~$1,550 lifetime value loss per churn, 40% save rate)
- Results, recommendations for the retention team, and limitations

**[`notebooks/01_business_analysis.ipynb`](notebooks/01_business_analysis.ipynb)** — business framing:

- Segmentation by contract, tenure bucket, payment method, and service bundle, each tied to annual revenue-at-risk in dollars
- Unified revenue-at-risk ranking across segmentation axes
- Ranked list of the top 100 currently-active customers the retention team should call first, scored with honest out-of-fold XGBoost predictions
- Retention action recommendations mapped to the segments that surface from the analysis

**[Tableau Public dashboard](https://public.tableau.com/views/CustomerChurnPrediction_1/Dashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)** — interactive version of the business analysis, built for stakeholder consumption. Segmentation charts cross-filter on click, and a budget parameter lets you adjust how many at-risk customers the retention team can reach.

## Stack

- **Model**: XGBoost, trained on the public IBM Telco churn dataset
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
