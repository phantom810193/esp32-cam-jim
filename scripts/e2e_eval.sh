#!/usr/bin/env bash
set -euo pipefail

REGION=${REGION:-asia-east1}
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "[ERROR] gcloud project is not set. Run 'gcloud config set project <PROJECT_ID>'." >&2
  exit 1
fi
ENDPOINT_ID=${ENDPOINT_ID:-projects/665759721336/locations/asia-east1/indexEndpoints/9005598365810950144}
IDX_DISPLAY=${IDX_DISPLAY:-faces-index-stream-bf}

BUCKET_DATA=${BUCKET_DATA:-esp32cam-472912-vertex-data}
BUCKET_STAGING=${BUCKET_STAGING:-esp32cam-472912-vertex-staging}
BUCKET_OUTPUT=${BUCKET_OUTPUT:-esp32cam-472912-vertex-output}

RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
LOCAL_DATA_DIR=${LOCAL_DATA_DIR:-$HOME/faces_raw}
STAGING_GCS=${STAGING_GCS:-gs://${BUCKET_STAGING}/embeddings_delta/${RUN_ID}}
OUTPUT_LOCAL=${OUTPUT_LOCAL:-vertex_eval_results/${RUN_ID}}
OUTPUT_GCS=${OUTPUT_GCS:-gs://${BUCKET_OUTPUT}/eval_results/${RUN_ID}}

THRESHOLD=${THRESHOLD:-0.25}
NEIGHBORS=${NEIGHBORS:-3}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

OUTPUT_LOCAL_DIR="${REPO_ROOT}/${OUTPUT_LOCAL}"
mkdir -p "${OUTPUT_LOCAL_DIR}"
export RUN_DIR="${OUTPUT_LOCAL_DIR}"
DEBUG_LOG="${OUTPUT_LOCAL_DIR}/debug.log"
touch "${DEBUG_LOG}"

log_debug() {
  printf '%s\n' "$1" >>"${DEBUG_LOG}"
}

EMBEDDING_LOCAL_PATH="${OUTPUT_LOCAL_DIR}/datapoints-00001-of-00001.json"
STAGING_GCS_PATH="${STAGING_GCS}/datapoints-00001-of-00001.json"

printf '[INFO] Using project %s in region %s\n' "${PROJECT_ID}" "${REGION}"
printf '[INFO] Run ID: %s\n' "${RUN_ID}"
printf '[INFO] Output directory: %s\n' "${OUTPUT_LOCAL_DIR}"

python3 -m pip install -q google-cloud-aiplatform google-cloud-storage \
  insightface onnxruntime onnx pillow numpy opencv-python

if [[ ! -d "${LOCAL_DATA_DIR}" || -z "$(find "${LOCAL_DATA_DIR}" -type f -print -quit 2>/dev/null)" ]]; then
  echo "[INFO] Local image directory missing or empty. Syncing from GCS..."
  if ! gcloud storage rsync -r "gs://${BUCKET_DATA}/faces_raw" "${LOCAL_DATA_DIR}"; then
    echo "[WARN] Unable to sync images from gs://${BUCKET_DATA}/faces_raw. Continuing." >&2
  fi
fi

HAS_IMAGES=0
if [[ -d "${LOCAL_DATA_DIR}" && -n "$(find "${LOCAL_DATA_DIR}" -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) -print -quit 2>/dev/null)" ]]; then
  HAS_IMAGES=1
fi

if [[ "${HAS_IMAGES}" -eq 1 ]]; then
  echo "[INFO] Generating embeddings with insightface..."
  export LOCAL_DATA_DIR EMBEDDING_LOCAL_PATH BUCKET_DATA
  export OUTPUT_LOCAL_DIR
  python3 <<'PY'
import json
import os
from pathlib import Path
import sys
import cv2
from insightface.app import FaceAnalysis

local_dir = Path(os.environ["LOCAL_DATA_DIR"]).expanduser()
out_path = Path(os.environ["EMBEDDING_LOCAL_PATH"])
out_path.parent.mkdir(parents=True, exist_ok=True)
source_bucket = os.environ.get("BUCKET_DATA", "")

image_paths = sorted([
    p for p in local_dir.rglob('*')
    if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
])

if not image_paths:
    print("[WARN] No images found for embedding generation.")
    sys.exit(0)

app = FaceAnalysis(name='buffalo_l')
try:
    app.prepare(ctx_id=0, det_size=(640, 640))
except Exception as exc:  # pragma: no cover - fallback for CPU environments
    print(f"[WARN] Failed to initialize FaceAnalysis on ctx_id=0 ({exc}). Falling back to CPU (ctx_id=-1).")
    app.prepare(ctx_id=-1, det_size=(640, 640))

written = 0
skipped = 0

with out_path.open('w', encoding='utf-8') as f:
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            skipped += 1
            continue
        faces = app.get(img)
        if not faces:
            skipped += 1
            continue
        embedding = faces[0].normed_embedding.astype(float).tolist()
        rel_path = path.relative_to(local_dir)
        rel_parts = rel_path.as_posix()
        gcs_uri = f"gs://{source_bucket}/faces_raw/{rel_parts}" if source_bucket else rel_parts
        record = {
            "id": gcs_uri,
            "embedding": embedding,
            "local_path": rel_parts,
        }
        f.write(json.dumps(record) + "\n")
        written += 1

print(f"[INFO] Embeddings written: {written}, skipped images: {skipped}")
PY
else
  echo "[WARN] No images available; skipping embedding generation."
fi

HAS_EMBEDDINGS=0
if [[ -s "${EMBEDDING_LOCAL_PATH}" ]]; then
  HAS_EMBEDDINGS=1
else
  echo "[WARN] No embeddings generated; downstream steps will be skipped." >&2
fi

if [[ "${HAS_EMBEDDINGS}" -eq 1 ]]; then
  echo "[INFO] Uploading embeddings to ${STAGING_GCS_PATH}"
  gcloud storage cp "${EMBEDDING_LOCAL_PATH}" "${STAGING_GCS_PATH}"
else
  echo "[INFO] Skipping upload because no embeddings are available."
fi

if [[ -z "${INDEX_STREAM:-}" ]]; then
  echo "[INFO] Creating STREAM_UPDATE brute-force index..."
  INDEX_DEF_FILE="${OUTPUT_LOCAL_DIR}/index_stream_bf.json"
  cat > "${INDEX_DEF_FILE}" <<'JSON'
{
  "config": {
    "dimensions": 512,
    "distanceMeasureType": "COSINE_DISTANCE",
    "featureNormType": "UNIT_L2_NORM",
    "algorithmConfig": { "bruteForceConfig": {} }
  }
}
JSON
  gcloud ai indexes create \
    --region="${REGION}" \
    --display-name="${IDX_DISPLAY}" \
    --metadata-file="${INDEX_DEF_FILE}" \
    --index-update-method=STREAM_UPDATE \
    >/dev/null
  INDEX_STREAM=$(gcloud ai indexes list \
    --region="${REGION}" \
    --filter="displayName=\"${IDX_DISPLAY}\"" \
    --format='value(name)' | tail -1)
  if [[ -z "${INDEX_STREAM}" ]]; then
    echo "[ERROR] Failed to retrieve index resource for displayName=${IDX_DISPLAY}." >&2
    exit 1
  fi
  if [[ "${INDEX_STREAM}" != projects/* ]]; then
    echo "[ERROR] Unexpected index identifier returned: ${INDEX_STREAM}" >&2
    exit 1
  fi
  export INDEX_STREAM
  echo "[INFO] Created index: ${INDEX_STREAM}"
else
  if [[ "${INDEX_STREAM}" != projects/* ]]; then
    INDEX_STREAM=$(gcloud ai indexes list \
      --region="${REGION}" \
      --filter="displayName=\"${IDX_DISPLAY}\"" \
      --format='value(name)' | tail -1)
    if [[ -z "${INDEX_STREAM}" ]]; then
      echo "[ERROR] Unable to resolve existing index via displayName=${IDX_DISPLAY}." >&2
      exit 1
    fi
    if [[ "${INDEX_STREAM}" != projects/* ]]; then
      echo "[ERROR] Resolved index identifier is invalid: ${INDEX_STREAM}" >&2
      exit 1
    fi
  fi
  export INDEX_STREAM
  echo "[INFO] Using existing index: ${INDEX_STREAM}"
fi

if [[ "${HAS_EMBEDDINGS}" -eq 1 ]]; then
  echo "[INFO] Upserting datapoints into ${INDEX_STREAM}"
  export INDEX_STREAM REGION STAGING_GCS_PATH
  python3 <<'PY'
import json
import os
import time
from pathlib import Path

from google.api_core import exceptions as gax_exceptions
from google.api_core import retry as gax_retry
from google.cloud import aiplatform_v1
from google.cloud import storage

index_name = os.environ["INDEX_STREAM"]
region = os.environ["REGION"]
uri = os.environ["STAGING_GCS_PATH"]

if not uri.startswith("gs://"):
    raise SystemExit("STAGING_GCS_PATH must be a gs:// URI")

bucket_name, _, object_path = uri[5:].partition('/')
if not bucket_name:
    raise SystemExit("Invalid GCS URI")

storage_client = storage.Client()
blob = storage_client.bucket(bucket_name).blob(object_path)
if not blob.exists():
    raise SystemExit(f"Embedding file {uri} not found in GCS.")

lines = []
with blob.open("r") as reader:
    for line in reader:
        line = line.strip()
        if not line:
            continue
        lines.append(json.loads(line))

if not lines:
    print("[WARN] No datapoints found in embedding file; skipping upsert.")
    raise SystemExit(0)

client = aiplatform_v1.IndexServiceClient(
    client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
)

def chunk(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

chunk_size = 100
total = 0
retry = gax_retry.Retry(initial=1.0, maximum=30.0, multiplier=2.0, deadline=300.0)

for batch in chunk(lines, chunk_size):
    datapoints = []
    for item in batch:
        embedding = item.get("embedding")
        if not embedding:
            continue
        dp = aiplatform_v1.IndexDatapoint(
            datapoint_id=str(item.get("id")),
            feature_vector=list(map(float, embedding)),
        )
        datapoints.append(dp)
    if not datapoints:
        continue
    request = aiplatform_v1.UpsertDatapointsRequest(
        index=index_name,
        datapoints=datapoints,
    )
    attempt = 0
    while True:
        attempt += 1
        try:
            retry(client.upsert_datapoints)(request=request)
            total += len(datapoints)
            break
        except (gax_exceptions.GoogleAPICallError, gax_exceptions.RetryError, gax_exceptions.ServiceUnavailable) as exc:
            if attempt >= 5:
                raise
            sleep_for = min(2 ** attempt, 30)
            print(f"[WARN] Upsert attempt {attempt} failed: {exc}. Retrying in {sleep_for}s...")
            time.sleep(sleep_for)

print(f"[INFO] Upserted {total} datapoints into {index_name}")
PY
else
  echo "[INFO] Skipping upsert because no embeddings were generated."
fi

DEPLOY_ID="faces_stream_${RUN_ID//[-:]/}"
ENDPOINT_SHORT=$(basename "${ENDPOINT_ID}")

if [[ "${HAS_EMBEDDINGS}" -eq 1 ]]; then
  echo "[INFO] Deploying index ${INDEX_STREAM} to endpoint ${ENDPOINT_ID}"
  attempt=0
  deployed=0
  while [[ ${attempt} -lt 3 && ${deployed} -eq 0 ]]; do
    attempt=$((attempt+1))
    candidate="${DEPLOY_ID}"
    if [[ ${attempt} -gt 1 ]]; then
      candidate="${DEPLOY_ID}_${attempt}"
    fi
    echo "[INFO] Deploy attempt ${attempt} with deployed-index-id=${candidate}"
    OUT="$(gcloud ai index-endpoints deploy-index "${ENDPOINT_ID}" \
      --region="${REGION}" \
      --index="${INDEX_STREAM}" \
      --deployed-index-id="${candidate}" \
      --display-name="${candidate}" 2>&1 || true)"
    if echo "${OUT}" | grep -q "ALREADY_EXISTS"; then
      echo "[WARN] Deployment ID already exists. Retrying with a new ID."
      log_debug "[WARN] deploy attempt ${attempt} failed: ALREADY_EXISTS"
      log_debug "[DETAIL] ${OUT}"
      continue
    fi
    if echo "${OUT}" | grep -q "INTERNAL"; then
      echo "[WARN] Deployment encountered INTERNAL error. Retrying with a new ID."
      log_debug "[WARN] deploy attempt ${attempt} failed: INTERNAL"
      log_debug "[DETAIL] ${OUT}"
      continue
    fi
    OP_ID="$(sed -n 's/.*operations\/\([0-9]\{6,\}\).*/\1/p' <<<"${OUT}" | tail -1)"
    if [[ -z "${OP_ID}" ]]; then
      echo "[ERROR] cannot parse operation id"
      log_debug "[ERROR] cannot parse operation id"
      log_debug "[DETAIL] ${OUT}"
      exit 1
    fi
    echo "[INFO] Waiting for operation ${OP_ID}"
    poll_error=""
    poll_success=0
    for i in {1..120}; do
      LINE=$(gcloud ai operations describe "${OP_ID}" \
        --index-endpoint="${ENDPOINT_SHORT}" \
        --region="${REGION}" \
        --format='value(done,error.message)' 2>&1)
      status=$?
      if [[ ${status} -ne 0 ]]; then
        echo "[WARN] Failed to poll operation ${OP_ID}: ${LINE}" >&2
        log_debug "[WARN] poll attempt ${i} failed for ${OP_ID}: ${LINE}"
        sleep 10
        continue
      fi
      LINE="${LINE//$'\t'/ }"
      DONE="${LINE%% *}"
      if [[ "${DONE}" == "${LINE}" ]]; then
        ERR=""
      else
        ERR="${LINE#* }"
      fi
      if [[ "${DONE}" == "True" || "${DONE}" == "true" ]]; then
        if [[ -n "${ERR}" && "${ERR}" != "None" ]]; then
          poll_error="${ERR}"
        fi
        poll_success=1
        break
      fi
      if [[ -n "${ERR}" && "${ERR}" != "None" ]]; then
        poll_error="${ERR}"
        break
      fi
      sleep 10
    done
    if [[ ${poll_success} -eq 1 && -z "${poll_error}" ]]; then
      DEPLOY_ID="${candidate}"
      deployed=1
      echo "[INFO] Deployment completed with deployed-index-id=${DEPLOY_ID}"
      gcloud ai index-endpoints describe "${ENDPOINT_ID}" --region="${REGION}" \
        --format="table(deployedIndexes.id,deployedIndexes.index,deployedIndexes.createTime)"
    else
      if [[ -n "${poll_error}" ]]; then
        echo "[ERROR] ${poll_error}"
        log_debug "[ERROR] deployment error for ${candidate}: ${poll_error}"
        if [[ ${attempt} -lt 3 ]]; then
          if [[ "${poll_error}" == *ALREADY_EXISTS* || "${poll_error}" == *already\ exists* || "${poll_error}" == *INTERNAL* ]]; then
            echo "[INFO] Retrying deployment with a new ID."
            continue
          fi
        fi
        exit 1
      else
        echo "[ERROR] Deployment timed out waiting for operation ${OP_ID}" >&2
        log_debug "[ERROR] deployment timed out for ${candidate} (operation ${OP_ID})"
        exit 1
      fi
    fi
  done
  if [[ ${deployed} -eq 0 ]]; then
    echo "[ERROR] Unable to deploy index after multiple attempts." >&2
    log_debug "[ERROR] Unable to deploy index after multiple attempts"
    exit 1
  fi
else
  echo "[INFO] Skipping deployment because no embeddings were generated."
fi

if [[ "${HAS_EMBEDDINGS}" -eq 1 ]]; then
  echo "[INFO] Running smoke test via quick_match_test.py"
  python3 scripts/quick_match_test.py \
    --staging-json "${STAGING_GCS_PATH}" \
    --endpoint "${ENDPOINT_ID}" \
    --deploy-id "${DEPLOY_ID}" \
    --out "${OUTPUT_LOCAL_DIR}/smoke_test.json"
else
  echo "[INFO] Skipping smoke test because no embeddings were generated."
fi

if [[ "${HAS_IMAGES}" -eq 1 && "${HAS_EMBEDDINGS}" -eq 1 ]]; then
  echo "[INFO] Running full evaluation via run_vertex_face_eval.py"
  python3 scripts/run_vertex_face_eval.py \
    --endpoint "${ENDPOINT_ID}" \
    --deploy-id "${DEPLOY_ID}" \
    --images-dir "${LOCAL_DATA_DIR}" \
    --out-dir "${OUTPUT_LOCAL_DIR}" \
    --threshold "${THRESHOLD}" \
    --neighbors "${NEIGHBORS}"
else
  echo "[INFO] Skipping full evaluation (images or embeddings not available)."
fi

if [[ "${UPLOAD_TO_GCS:-0}" == "1" && -d "${OUTPUT_LOCAL_DIR}" ]]; then
  echo "[INFO] Uploading results to ${OUTPUT_GCS}"
  gcloud storage cp -r "${OUTPUT_LOCAL_DIR}" "${OUTPUT_GCS}"
fi

echo "[INFO] Local results available at: ${OUTPUT_LOCAL_DIR}"
if [[ "${UPLOAD_TO_GCS:-0}" == "1" ]]; then
  echo "[INFO] GCS results uploaded to: ${OUTPUT_GCS}"
fi
