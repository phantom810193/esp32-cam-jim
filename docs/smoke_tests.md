# Vertex AI 臉部辨識端對端驗證指南

## 1. Pipeline 成功驗證
使用 Vertex AI Python SDK 查詢 Index 內的資料或做隨機查詢。

```python
from google.cloud import aiplatform_v1

PROJECT_ID = "esp32cam-472912"
REGION = "asia-east1"
INDEX_NAME = os.environ["INDEX_NAME"]  # 先 export

client = aiplatform_v1.IndexServiceClient(client_options={
    "api_endpoint": f"{REGION}-aiplatform.googleapis.com"
})
response = client.list_datapoints(request={
    "index": INDEX_NAME,
    "page_size": 5,
})
print(f"Found {len(list(response))} datapoints")
```

若想抽樣比對，可改用 `MatchServiceClient.find_neighbors` 對任意 embedding 查詢，確認能回傳最近鄰。

## 2. Cloud Run HTTP 測試

```bash
SERVICE_URL="$(gcloud run services describe faces-recognizer \
  --project=esp32cam-472912 --region=asia-east1 --format='value(status.url)')"

curl -X POST "${SERVICE_URL}/recognize" \
  -H "Content-Type: application/json" \
  -d '{"gcs_uri": "gs://esp32cam-472912-vertex-data/faces/personA/sample.jpg"}'
```

回傳示例：

```json
{
  "topK": [
    {"datapoint_id": "personA#sample.jpg", "score": 0.93, "distance": 0.07},
    ...
  ],
  "best_match": {"datapoint_id": "personA#sample.jpg", "score": 0.93, "distance": 0.07},
  "match": true,
  "threshold": 0.6
}
```

score 越接近 1 越相似；當 `match=true` 代表相似度 >= 門檻。

## 3. Eventarc 觸發驗證

```bash
gsutil cp local_test.jpg gs://esp32cam-472912-vertex-data/faces/inbox/local_test.jpg
```

接著使用：

```bash
gcloud logs read "projects/esp32cam-472912/logs/run.googleapis.com%2Fstdout" \
  --region=asia-east1 --limit=50
```

檢查日誌是否出現：
- `Processing GCS object gs://.../faces/inbox/local_test.jpg`
- `Event outcome: {'status': 'existing', ...}` 或 `enrolled`
- 若是新用戶會看到 upsert 成功的 INFO。

## 4. 多人辨識示例

```python
import base64
import io
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))

image = Image.open("group_photo.jpg").convert("RGB")
arr = np.array(image)[:, :, ::-1]
faces = app.get(arr)

for idx, face in enumerate(faces):
    crop = arr[int(face.bbox[1]):int(face.bbox[3]), int(face.bbox[0]):int(face.bbox[2])][:, :, ::-1]
    embedding = face.normed_embedding.astype(float).tolist()
    # 呼叫 /recognize 或 MatchServiceClient 查詢
```

針對每張臉送查詢即可取得 person_id 與 score。

## 5. 調整門檻
- 若誤判率高（太容易誤識），將 `THRESHOLD` 環境變數提高到 0.65–0.7 後重新 `gcloud run deploy`。
- 若漏判多（難以匹配），可降低到 0.55，但建議搭配混淆矩陣分析。
- 每次調整後重新測量 `/recognize` 以及 Eventarc 流程，確保日誌顯示符合預期。
