#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-esp32cam-472912}"
REGION="${REGION:-asia-east1}"
SERVICE_NAME="${SERVICE_NAME:-faces-recognizer}"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"
INDEX_NAME="${INDEX_NAME:?INDEX_NAME is required}"
INDEX_ENDPOINT_ID="${INDEX_ENDPOINT_ID:?INDEX_ENDPOINT_ID is required}"
DEPLOYED_INDEX_ID="${DEPLOYED_INDEX_ID:?DEPLOYED_INDEX_ID is required}"
BUCKET_DATA="${BUCKET_DATA:-esp32cam-472912-vertex-data}"
THRESHOLD="${THRESHOLD:-0.6}"

pushd "$(dirname "$0")" >/dev/null

echo "[INFO] Building container ${IMAGE}"
gcloud builds submit --tag "${IMAGE}" ..

echo "[INFO] Deploying Cloud Run service ${SERVICE_NAME}"
gcloud run deploy "${SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${IMAGE}" \
  --platform=managed \
  --allow-unauthenticated \
  --set-env-vars "PROJECT_ID=${PROJECT_ID},REGION=${REGION},INDEX_NAME=${INDEX_NAME},INDEX_ENDPOINT_ID=${INDEX_ENDPOINT_ID},DEPLOYED_INDEX_ID=${DEPLOYED_INDEX_ID},BUCKET_DATA=${BUCKET_DATA},THRESHOLD=${THRESHOLD}"

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --project="${PROJECT_ID}" --region="${REGION}" --format='value(status.url)')"
echo "[INFO] Service URL: ${SERVICE_URL}"

echo "[INFO] Creating Eventarc trigger faces-index-trigger"
gcloud eventarc triggers create faces-index-trigger \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --destination-run-service="${SERVICE_NAME}" \
  --destination-run-region="${REGION}" \
  --event-filters="type=google.cloud.storage.object.v1.finalized" \
  --event-filters="bucket=${BUCKET_DATA}" \
  --event-filters-path-pattern="subject=projects/_/buckets/${BUCKET_DATA}/objects/faces/inbox/*" \
  --service-account="${SERVICE_ACCOUNT:-faces-sa@${PROJECT_ID}.iam.gserviceaccount.com}" \
  --transport-topic="${SERVICE_NAME}-eventarc" \
  --destination-run-path="/events" || true

echo "[INFO] To inspect logs: gcloud logs read --project=${PROJECT_ID} --region=${REGION} --limit=50 --format='value(textPayload)' "

popd >/dev/null
