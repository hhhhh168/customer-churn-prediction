# Customer Churn Prediction

XGBoost churn model for telecom customers, served behind a FastAPI + Gradio app on AWS ECS Fargate. Built as a personal project to get an end-to-end ML system into production rather than stop at a notebook.

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

- The classification threshold is 0.35, picked to favor recall over precision since missing a churner is generally more expensive than a false positive in this domain.
- Class imbalance is handled with `scale_pos_weight` in XGBoost rather than resampling.
- Training and serving use the same feature transformations — see `src/features/build_features.py` and `src/serving/inference.py`. If you change one, change the other.
