# Optional Google Cloud Deployment Guide

Google Cloud deployment is **completely optional**. SpectrumBloom runs locally via Ollama / llama.cpp and Docker Compose with zero cloud accounts.

If you choose to deploy to Google Cloud Run (scales to zero when idle):

> **Sandbox security note:** Serverless Cloud Run containers cannot spawn isolated Docker sibling containers safely, so [`cloudrun-api.yaml`](cloudrun-api.yaml) explicitly sets `SANDBOX_BACKEND=disabled` to run deterministic EDA, ML Lab, and deterministic copilot tools without executing arbitrary agent-generated Python. If you need dynamic Python execution in a cloud environment, deploy a separate private `sandbox-executor` service (using `Dockerfile.executor`) inside an isolated VPC boundary and configure `SANDBOX_BACKEND=remote`, `SANDBOX_EXECUTOR_URL`, and `SANDBOX_EXECUTOR_TOKEN` via Secret Manager — never `strict_subprocess`.

1. Build and push the standard container image (no privileged flags needed):

   ```bash
   gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT_ID/spectrumbloom/api:latest .
   ```

2. Create a Secret Manager secret `spectrumbloom-api-auth-token` (required because `APP_ENV=production` refuses unauthenticated startup), update `PROJECT_ID` and `REGION` in `cloudrun-api.yaml`, inject any optional LLM provider keys via Secret Manager, and deploy:

   ```bash
   gcloud run services replace deploy/gcp/cloudrun-api.yaml --region REGION
   ```

3. For multi-instance persistence without AWS S3, attach an optional GCS bucket (via S3 interoperability endpoint `https://storage.googleapis.com` with HMAC keys) or Memorystore Redis by setting `REDIS_URL` and `S3_ENDPOINT_URL` in environment secrets.
