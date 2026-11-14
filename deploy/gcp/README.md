# Optional Google Cloud Deployment Guide

Google Cloud deployment is **completely optional**. SpectrumBloom runs locally via Ollama / llama.cpp and Docker Compose with zero cloud accounts.

If you choose to deploy to Google Cloud Run (scales to zero when idle):

1. Build and push the standard container image (no privileged flags needed):

   ```bash
   gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT_ID/spectrumbloom/api:latest .
   ```

2. Create a Secret Manager secret `spectrumbloom-api-auth-token` (required because `APP_ENV=production` refuses unauthenticated startup), update `PROJECT_ID` and `REGION` in `cloudrun-api.yaml`, inject any optional LLM provider keys via Secret Manager, and deploy:

   ```bash
   gcloud run services replace deploy/gcp/cloudrun-api.yaml --region REGION
   ```

3. For multi-instance persistence without AWS S3, attach an optional GCS bucket (via S3 interoperability endpoint `https://storage.googleapis.com` with HMAC keys) or Memorystore Redis by setting `REDIS_URL` and `S3_ENDPOINT_URL` in environment secrets.
