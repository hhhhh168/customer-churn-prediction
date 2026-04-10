## Customer Churn Prediction - End-to-End ML Project

### Purpose

A full machine-learning solution for predicting customer churn in a telecom setting -- from data prep and modeling to an API + web UI deployed on AWS.

### What this project does

- **Predicts churn**: Identifies which customers are likely to leave so teams can act before they do.
- **Operationalized ML**: Model is accessible via a REST API and a Gradio web UI; anyone can test it without notebooks.
- **Repeatable delivery**: CI/CD + Docker containers mean every change is rebuilt, tested, and redeployed consistently.
- **Traceable experiments**: MLflow tracks runs, metrics, and artifacts for reproducibility and auditing.

### What I built

- **Data & Modeling**: Feature engineering + XGBoost classifier; experiments logged to MLflow.
- **Inference service**: FastAPI app exposing `/predict` (POST) and a root health check `/`.
- **Web UI**: Gradio interface mounted at `/ui` for quick manual testing.
- **Containerization**: Docker image with uvicorn entrypoint listening on port 8000.
- **CI/CD**: GitHub Actions builds the image and pushes to Docker Hub; triggers an ECS service update.
- **Orchestration**: AWS ECS Fargate runs the container (serverless).
- **Networking**: Application Load Balancer (ALB) on HTTP:80 forwarding to a Target Group on HTTP:8000.
- **Security**: Security groups scoped to allow ALB inbound 80 from the internet, and task inbound 8000 from the ALB SG.
- **Observability**: CloudWatch Logs for container stdout/stderr and ECS service events.

### Deployment flow

1. Push to `main` -> GitHub Actions builds the Docker image and pushes it to Docker Hub.
2. ECS service is updated to force a new deployment.
3. ALB health checks hit `/` on port 8000; once healthy, traffic is routed to the new task.
4. Users call `POST /predict` or open the Gradio UI at `/ui` via the ALB DNS.

### Running locally

```bash
# Train the model
python scripts/run_pipeline.py --input data/raw/Customer-Churn.csv --target Churn

# Run the API
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000

# Docker
docker build -t customer-churn-app .
docker run -p 8000:8000 customer-churn-app
```
