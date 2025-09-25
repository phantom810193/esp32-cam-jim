# Streaming Index & Endpoint Provisioning

下列步驟會在 `asia-east1` 建立新的 Vertex AI Streaming Index，並部署到 Index Endpoint。所有指令均可直接貼到 Cloud Shell（需先 `gcloud auth login` 並確認 `gcloud config set project esp32cam-472912`）。

```bash
#!/usr/bin/env bash
set -euo pipefail

REGION="asia-east1"
PROJECT_ID="${PROJECT_ID:-esp32cam-472912}"
INDEX_DISPLAY_NAME="${INDEX_DISPLAY_NAME:-faces-index-stream-512}"
INDEX_CFG="index_cfg.json"

# 1. 建立 index metadata
cat >"${INDEX_CFG}" <<'JSON'
{
  "config": {
    "dimensions": 512,
    "distanceMeasureType": "COSINE_DISTANCE",
    "algorithmConfig": {
      "bruteForceConfig": {}
    }
  }
}
JSON

# 2. 建立 Streaming Index
CREATE_OUT="$(gcloud ai indexes create \
  --region="${REGION}" \
  --display-name="${INDEX_DISPLAY_NAME}" \
  --metadata-file="${INDEX_CFG}" \
  --index-update-method=STREAM_UPDATE \
  --format="json(name,createTime)" 2>&1 || true)"
echo "${CREATE_OUT}"
INDEX_OPERATION="$(sed -n 's/.*operations\/\([^" ]\+\).*/\1/p' <<<"${CREATE_OUT}" | tail -1)"
[[ -z "${INDEX_OPERATION}" ]] && { echo "[ERROR] 無法解析 index operation id"; exit 1; }

# 3. 等待 Index 建立完成
API_ENDPOINT="${REGION}-aiplatform.googleapis.com"
for attempt in {1..120}; do
  STATUS="$(gcloud ai operations describe "${INDEX_OPERATION}" \
    --region="${REGION}" \
    --format='value(done,error.message)')"
  DONE="${STATUS%% *}"; ERR="${STATUS#* }"
  if [[ "${DONE}" == "True" ]]; then
    [[ -n "${ERR}" && "${ERR}" != "None" ]] && { echo "[ERROR] ${ERR}"; exit 1; }
    break
  fi
  sleep 10
  [[ ${attempt} -eq 120 ]] && { echo "[ERROR] Index operation timeout"; exit 1; }
done

INDEX_NAME="$(gcloud ai indexes list --region="${REGION}" \
  --filter="displayName=${INDEX_DISPLAY_NAME}" \
  --format='value(name)' | tail -1)"
[[ -z "${INDEX_NAME}" ]] && { echo "[ERROR] 取得 INDEX_NAME 失敗"; exit 1; }
export INDEX_NAME
INDEX_ID="$(basename "${INDEX_NAME}")"
export INDEX_ID

# 4. 建立 Index Endpoint
ENDPOINT_DISPLAY="${INDEX_DISPLAY_NAME}-endpoint"
ENDPOINT_ID="$(gcloud ai index-endpoints create \
  --region="${REGION}" \
  --display-name="${ENDPOINT_DISPLAY}" \
  --format='value(name)' )"
export ENDPOINT_ID

# 5. 部署 Index 至 Endpoint
DEPLOY_ID="faces-stream-$(date +%Y%m%d%H%M%S)"
DEPLOY_OUT="$(gcloud ai index-endpoints deploy-index "${ENDPOINT_ID}" \
  --region="${REGION}" \
  --index="${INDEX_NAME}" \
  --deployed-index-id="${DEPLOY_ID}" \
  --display-name="${DEPLOY_ID}" 2>&1 || true)"
OP_ID="$(sed -n 's/.*operations\/\([0-9]\{6,\}\).*/\1/p' <<<"${DEPLOY_OUT}" | tail -1)"
[[ -z "${OP_ID}" ]] && { echo "[ERROR] 部署 operation id 解析失敗"; exit 1; }

echo "[INFO] 等待部署 operation ${OP_ID} 完成"
for attempt in {1..120}; do
  STATUS_LINE="$(gcloud ai operations describe "${OP_ID}" \
    --index-endpoint="$(basename "${ENDPOINT_ID}")" \
    --region="${REGION}" \
    --format='value(done,error.message)')"
  DONE="${STATUS_LINE%% *}"; ERR="${STATUS_LINE#* }"
  if [[ "${DONE}" == "True" ]]; then
    [[ -n "${ERR}" && "${ERR}" != "None" ]] && { echo "[ERROR] ${ERR}"; exit 1; }
    break
  fi
  sleep 10
  [[ ${attempt} -eq 120 ]] && { echo "[ERROR] Deploy operation timeout"; exit 1; }
done

# 6. 查詢部署結果
printf '\nCurrent deployments:\n'
gcloud ai index-endpoints describe "${ENDPOINT_ID}" --region="${REGION}" \
  --format='table(deployedIndexes.id,deployedIndexes.index,deployedIndexes.createTime)'

echo "\nexport INDEX_NAME=${INDEX_NAME}"
echo "export DEPLOYED_INDEX_ID=${DEPLOY_ID}"
echo "export INDEX_ENDPOINT_ID=${ENDPOINT_ID}"
```

> **驗收提示**：完成後請執行 `echo $INDEX_NAME`、`echo $DEPLOYED_INDEX_ID`、`echo $INDEX_ENDPOINT_ID` 確保環境變數可被後續步驟讀取。
```
