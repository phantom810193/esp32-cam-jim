# 常見錯誤與修復建議

- **pip 安裝失敗（numpy 升級到 2.x 或輪子缺失）**：在指令中加入 `--only-binary=:all:` 並鎖版本 `numpy==1.26.4`、`onnxruntime==1.17.3`、`opencv-python-headless==4.10.0.84`，必要時重試 2–3 次。
- **onnxruntime / opencv 缺少動態庫**：部署映像請 `apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1`，確保執行期可載入。
- **InsightFace 匯入失敗**：使用 `pip install --no-deps insightface==0.7.3`，再手動安裝 numpy、onnxruntime、opencv 等依賴的指定版本。
- **Vertex SDK 使用錯誤**：
  - `IndexServiceClient` 用於 `upsert_datapoints`、`list_datapoints`，請帶 `index=<INDEX_NAME>`。
  - `MatchServiceClient` 用於 `find_neighbors`，需提供 `index_endpoint=<INDEX_ENDPOINT_ID>` 與 `deployed_index_id=<DEPLOYED_INDEX_ID>`。
- **Streaming Index 查不到新資料**：確認 upsert 時使用的是 `INDEX_NAME`（非 operation 路徑），並確認最新的 Index 已部署到 Endpoint（`gcloud ai index-endpoints describe`）。
- **Eventarc 未觸發**：檢查 `faces/inbox/` 前綴是否符合；確定 Eventarc 觸發器與 Cloud Run 服務在 `asia-east1`；服務帳號需具備 `roles/eventarc.eventReceiver` 與 Cloud Run Invoker。
- **Cloud Run 記憶體不足**：下載 GCS 檔案時改用串流（`blob.download_as_bytes()` 後即釋放）；Pillow 轉換時可 `thumbnail` 限制大小；調整 FastAPI workers 為 `--workers 1`。
- **相似度門檻不合理**：若誤判多，提升 `THRESHOLD`；若漏判，降低；調整後重新進行 smoke test。
- **Eventarc upsert 無法寫入**：確認 Cloud Run 使用的 service account 擁有 `roles/aiplatform.user` 以便呼叫 Index API。
- **Pipeline 執行報錯 PermissionDenied**：確保 pipeline job 使用的 service account 具備 Storage 讀寫、Aiplatform Admin 權限。
