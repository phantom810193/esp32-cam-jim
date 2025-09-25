# Vertex AI Pipeline 與 Search 指南

本文件說明如何使用專案提供的 GCS bucket 與 Python 腳本，部署人臉辨識訓練 Pipeline 與 Vertex AI Search 顧客資料庫。以下範例皆以專案 ID `esp32cam-472912` 及使用者提供的 bucket：

- `gs://esp32cam-472912-vertex-data`
- `gs://esp32cam-472912-vertex-staging`
- `gs://esp32cam-472912-vertex-output`
- `gs://esp32cam-472912-search-docs`

## 1. 服務啟用與權限

```bash
gcloud config set project esp32cam-472912

# Vertex AI Pipelines + Datastore
 gcloud services enable aiplatform.googleapis.com discoveryengine.googleapis.com
# 需要 GCS 與 Compute Engine
 gcloud services enable storage.googleapis.com compute.googleapis.com cloudbuild.googleapis.com

# 建議建立專用 Service Account
 gcloud iam service-accounts create vertex-pipeline-sa \
    --display-name "Vertex Pipeline SA"

# 賦予必要角色（Vertex AI Admin、Storage Object Admin、Service Account User）
 gcloud projects add-iam-policy-binding esp32cam-472912 \
    --member=serviceAccount:vertex-pipeline-sa@esp32cam-472912.iam.gserviceaccount.com \
    --role=roles/aiplatform.admin
 gcloud projects add-iam-policy-binding esp32cam-472912 \
    --member=serviceAccount:vertex-pipeline-sa@esp32cam-472912.iam.gserviceaccount.com \
    --role=roles/storage.objectAdmin
 gcloud projects add-iam-policy-binding esp32cam-472912 \
    --member=serviceAccount:vertex-pipeline-sa@esp32cam-472912.iam.gserviceaccount.com \
    --role=roles/iam.serviceAccountUser
```

> 若要交由 Cloud Build 觸發 Pipeline，請在建置 trigger 中使用此 Service Account。

## 2. 準備資料集

確保照片依照 `faces/<person_id>/<image>.jpg` 結構存放於 `gs://esp32cam-472912-vertex-data`。建議同時準備 `metadata.json` 以包含 personId、時間戳、備註等資訊，供 pipeline 與 Vertex AI Search 同步使用。

## 3. 編譯與上傳 Pipeline

專案提供 `vertex_ai/pipeline.py` 作為 Kubeflow Pipeline 範本。此檔案會：

1. 下載 `vertex-data` bucket 內的影像與 metadata。
2. 使用 Cloud Vision API 產生 embedding，並於 `vertex-output` 存放模型/向量檔。
3. 於評估節點計算辨識率，確認達到 80% 門檻。

### 3.1 安裝工具

```bash
pip install google-cloud-aiplatform>=1.48 kfp>=2.6
```

### 3.2 編譯 Pipeline JSON

```bash
python vertex_ai/pipeline.py \
  --project esp32cam-472912 \
  --region asia-east1 \
  --data-bucket gs://esp32cam-472912-vertex-data \
  --output-bucket gs://esp32cam-472912-vertex-output \
  --pipeline face_pipeline.json
```

### 3.3 觸發 Pipeline

可在 CLI 或 Console 觸發：

```bash
gcloud ai pipelines run \
  --project=esp32cam-472912 \
  --region=asia-east1 \
  --display-name=esp32cam-face-recognition \
  --pipeline-definition-file=face_pipeline.json \
  --pipeline-root=gs://esp32cam-472912-vertex-staging/pipeline-root \
  --service-account=vertex-pipeline-sa@esp32cam-472912.iam.gserviceaccount.com \
  --parameter-values=\
      project_id=esp32cam-472912,\
      location=asia-east1,\
      data_bucket=gs://esp32cam-472912-vertex-data,\
      output_bucket=gs://esp32cam-472912-vertex-output
```

Pipeline 完成後，模型與評估報告會寫入 `vertex-output` bucket，供 Cloud Run API 或 ESP32 裝置下載使用。

## 4. Vertex AI Search 顧客資料庫

### 4.1 準備文件

於 `gs://esp32cam-472912-search-docs` 建立 `customers/` 目錄，每位顧客一個 JSON：

```json
{
  "id": "personA",
  "displayName": "王小明",
  "lastVisit": "2025-09-17T12:30:00Z",
  "purchases": ["Milk", "Bread"],
  "promotion": "Milk 10% off",
  "embeddingUri": "gs://esp32cam-472912-vertex-output/embeddings/personA.npy"
}
```

### 4.2 建立 Data Store

1. Console：Vertex AI Search → Data Stores → Create。
2. 選擇 **Content Search** 類型，資料來源為 GCS，指向 `gs://esp32cam-472912-search-docs`。
3. 允許自動推論 schema，或上傳 JSON Schema，其中包含 `id`, `displayName`, `purchases`, `promotion`, `embeddingUri` 等欄位。

CLI 建立資料匯入：

```bash
gcloud alpha discovery-engine data-stores import documents \
  --project=esp32cam-472912 \
  --location=global \
  --data-store=<DATA_STORE_ID> \
  --gcs-source=gs://esp32cam-472912-search-docs/customers/*.json
```

### 4.3 查詢整合

`api.py` 的 `/detect_face` 回應可加入搜尋結果：

```python
from google.cloud import discoveryengine_v1beta

client = discoveryengine_v1beta.SearchServiceClient()
serving_config = (
    "projects/esp32cam-472912/locations/global/collections/default_collection/"
    "dataStores/<DATA_STORE_ID>/servingConfigs/default_config"
)
request = discoveryengine_v1beta.SearchRequest(
    serving_config=serving_config,
    query=f"{person_id} promotion",
    user_pseudo_id=person_id,
)
response = client.search(request)
```

把 `response` 中的 `summary` 或 `results` 轉換成推播內容，即可於 LCD/`display.py` 顯示個人化優惠。

### 4.4 自動同步

- 在 Cloud Build 或 GitHub Actions 完成 Pipeline 後，於 `main.py` 中呼叫 Search API 新增/更新顧客檔案。
- Pipeline 輸出的最新 embedding 路徑可寫入 JSON（放在 `vertex-output/embeddings/`）並在 Search 文件中更新，達成模型與顧客資料同步。

## 5. Cloud Run 部署建議

1. Docker 映像透過 Cloud Build 預先下載模型 (`gsutil cp` from `vertex-output`) 與 `sample_data`。
2. 於部署指令加入：

```bash
gcloud run deploy face-api \
  --source=. \
  --service-account=vertex-pipeline-sa@esp32cam-472912.iam.gserviceaccount.com \
  --set-env-vars=VERTEX_MODEL_URI=gs://esp32cam-472912-vertex-output/exported_model
```

3. ESP32 端即可呼叫 `/enroll` / `/detect_face`，CI 仍由 GitHub Actions `pytest` 驗證。

## 6. 故障排除

- **403 權限不足**：確認 Pipeline 所用 Service Account 具備 Storage 與 Vertex AI 權限，並在 bucket 加入對應 IAM。
- **Search 無結果**：請檢查 GCS JSON 是否包含 `id` 欄位，並確認資料匯入後 Data Store 的 Document count > 0。
- **Pipeline 執行失敗**：於 Vertex AI Pipelines 詳細頁面查看各 component logs，必要時在 `vertex_ai/pipeline.py` 開啟更多 log 輸出。

## 7. 下一步

- 將 Pipeline 成功部署後的模型 URI 與 Search 查詢結果回寫至 SQLite 或 Firestore，以供 ESP32 顯示。
- 可透過 Cloud Scheduler 定期觸發 Pipeline 重新訓練、並呼叫 Search API 更新顧客資料。

如需進一步整合，請參考 `README.md` 內的 CI 測試流程與 API 撰寫範例。
