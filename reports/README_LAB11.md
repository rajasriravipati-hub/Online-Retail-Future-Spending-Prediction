# Lab 11: Docker and Kubernetes Deployment

## 1. Objective
To containerize the Online Retail Future Spending Prediction API using Docker and deploy it using Kubernetes.

## 2. Docker Implementation
- Created a Dockerfile using Python 3.13-slim.
- Installed API dependencies using requirements-api.txt.
- Copied the API source code and Lab 9 recovery model into the image.
- Successfully built the Docker image: `online-retail-api:v1`.
- Started the container on port 8001.

## 3. Docker Testing
- Docker container status: Running.
- Health endpoint: Passed.
- Single prediction endpoint (`POST /predict`): Passed.
- Batch prediction endpoint (`POST /predict/batch`): Passed.
- Model loaded successfully inside the container.

## 4. Kubernetes Deployment
Created the following configuration files:
- `k8s/deployment.yaml`
- `k8s/service.yaml`

Deployed the application using Kubernetes with 2 replicas initially.

Service details:
- Service name: online-retail-service
- Service type: NodePort
- NodePort: 30081
- Container port: 8000

## 5. Kubernetes Testing
- Kubernetes node status: Ready.
- Both initial pods reached Running status.
- Health endpoint: Passed.
- Single prediction endpoint: Passed.
- Batch prediction endpoint: Passed.
- Scaled deployment from 2 to 3 replicas successfully.
- All 3 pods reached Running status.

## 6. Self-Healing Test
Deleted one Online Retail API pod manually.

Kubernetes automatically created a replacement pod. The deployment returned to 3 running pods successfully.

## 7. Logs Verification
Checked application logs using:

`kubectl logs deployment/online-retail-deployment --tail=30`

Observed:
- Application startup complete.
- Uvicorn running on `0.0.0.0:8000`.
- Swagger documentation and OpenAPI requests returned HTTP 200 OK.

## 8. Conclusion
Successfully containerized and deployed the Online Retail Future Spending Prediction API using Docker and Kubernetes. Verified single and batch predictions, health checks, scaling, application logs, and Kubernetes self-healing.