# ESP32 Cloud Vision Demo (CI Ready)

本專案提供一套可在 GitHub Actions 上自動驗證的 ESP32 零售體驗流程，涵蓋攝影機、Cloud Vision 臉部辨識、SQLite 客製化行銷、Flask API 與終端機/LCD 文字輸出。所有二進位資產（ArcFace ONNX 模型、測試圖片、影片等）都移出版本控制，改由 Cloud Build 於建置時自動自 Google Cloud Storage (GCS) 下載，符合「方案A」的要求。

## 專案結構

```
.
├── api.py                # Flask /enroll 與 /detect_face 端點
├── camera.py             # 攝影機模擬與 cam.log
├── stability.py          # 30 分鐘穩定度報告 cam_stable.log
├── vision.py             # Cloud Vision + 向量嵌入模擬與 id_test.log
├── display.py            # 終端機輸出與 text_test.log
├── main.py               # 任務4 串接流程 e2e.log
├── admin_dashboard.py    # 任務5(1) 管理員後台 HTML
├── promo_ui.py           # 任務5(2) 推播 UI
├── data.sql              # SQLite users + visits 預設資料建置腳本
├── sample_data/          # 臉部向量測試資料（純文字）
├── tests/                # PyTest 測試覆蓋任務1~5
└── .github/workflows/    # CI 腳本
```

## 快速開始

1. **安裝依賴**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **執行測試**
   ```bash
   pytest
   ```
   測試流程會自動建立所有日誌 (`cam.log`, `id_test.log`, `cam_stable.log`, `text_test.log`, `api_test.log`,
   `promo_display.log(.json)`, `admin_dashboard.html`, `e2e.log`) 並檢查 FPS/準確率/回應時間門檻。
3. **端到端流程**
   ```bash
   python main.py
   ```
   於專案根目錄產生最新 `cam.log`, `id_test.log`, `cam_stable.log`, `api_test.log`,
   `text_test.log`, `promo_display.log(.json)`, `admin_dashboard.html`, `e2e.log` 與 `users.db`。

## Cloud Build / GCS 模型注入

1. 將 `arcface_r100.onnx` 上傳至專案專用的 GCS 私有桶，例如 `gs://<PROJECT>-models/arcface_r100.onnx`。
2. 在 `cloudbuild.yaml` 中於 Docker build 前加入：
   ```yaml
   - name: gcr.io/google.com/cloudsdktool/cloud-sdk
     entrypoint: gsutil
     args: ["cp", "gs://<PROJECT>-models/arcface_r100.onnx", "embed/models/arcface_r100.onnx"]
   ```
3. Docker 映像中即可取得模型，仍維持 Git repo 無二進位檔案。

## GitHub Actions

`.github/workflows/ci.yml` 會在 push / PR 自動：
1. 安裝 Python 3.11 與依賴。
2. 執行 `pytest` 驗證任務1~5指標。
3. 生成測試報告與日誌檔供下載。

> **合併前請確認**：GitHub Actions 的 "Lint GitHub Workflows / actionlint (validate workflow syntax)" 工作需成功通過，以確保 workflow 語法正確。

## 貢獻指南

維運須知：
- 到 Settings → Branches → Branch protection rules（主分支）：
  - 勾 **Require status checks to pass before merging**
  - 加入必須通過的檢查：`Lint GitHub Workflows / actionlint (validate workflow syntax)`
- 儲存設定。

## 任務說明與驗證指標

| 任務 | 自動化腳本 | 驗證項目 | 對應 log |
|------|------------|----------|----------|
| 任務1(1) | `camera.py` | FPS > 10 | `cam.log` |
| 任務1(2) | `vision.py` | 臉辨 5 張 >= 80% | `id_test.log` |
| 任務2 | `stability.py` | 30 分穩定、FPS>10、Accuracy>0.8 | `cam_stable.log` |
| 任務3(1) | `api.py` | `/enroll` + `/detect_face` < 1 秒、記錄訪客 | `api_test.log` |
| 任務3(2) | `data.sql` + `database_utils.py` | SQLite `users` 與 `visits` 表、首訪/返店邏輯 | `users.db` |
| 任務3(3) | `display.py` | 5 筆訊息 < 5 秒 | `text_test.log` |
| 任務4 | `main.py` | 攝影機→辨識→建檔→返店推播 < 30 秒 | `e2e.log` |
| 任務5 | `admin_dashboard.py`, `promo_ui.py` | 管理 UI + 推播 UI | `admin_dashboard.html`, `promo_display.log` |
| 任務6 | `README.md` | 專案說明、執行步驟 | N/A |

## 首訪與返店流程

1. **首次建檔 `/enroll`**：
   - 對 Cloud Run 送多張照片或 embedding 陣列。
   - 回傳訊息為「新用戶已建檔」，SQLite `users` 與 `visits` 新增紀錄。
   - CI 測試會確認 `visit_count == 1` 並寫入 `api_test.log`。
2. **返店辨識 `/detect_face`**：
   - ESP32 或測試腳本送入 embedding；API 0.2 秒上下回應。
   - 回傳「老朋友歡迎回來」，`visits` 增加一筆並根據上一筆消費產生優惠（例：「上次買牛奶，現9折！」）。
   - `text_test.log` 與 LCD/Led display 會顯示上一筆消費 + 推播。
3. **效能觀察**：
   - Cloud Run 約 200 ms、Vision 約 200 ms；Firestore/SQLite 首訪 1 秒內屬正常（冷啟或憑證交換）。
   - 若後續仍慢，可考慮連線重用、批次寫入或選擇接近客戶的區域資料庫。

## 實體 ESP32 測試建議

1. 於 Cloud Run 部署 Flask API (`api.py`) 與 Cloud Vision 服務金鑰。
2. ESP32 端程式先呼叫 `/enroll` 建檔，再呼叫 `/detect_face` 驗證返店推播。
3. 透過 `cam_stable.log` 與 `e2e.log` 指標確認 30 分穩定度與端到端延遲。
4. LCD/終端機可重用 `display.py` 與 `promo_ui.py` 輸出格式。

## Demo 影片製作

`demo_plan.md`（可自行新增）中列出錄製建議腳本：
1. 開啟管理員後台畫面。
2. ESP32 進行實際拍攝辨識（含 `/enroll` → `/detect_face` 流程）。
3. LCD 推播顯示優惠訊息。
4. 於 1 分鐘內剪輯完成。

## 後續擴充

- 將 `FaceRecognitionService` 接到實際 GCS 模型 + Cloud Vision API。
- 以 Pub/Sub 或 Firebase Cloud Messaging 建立即時推播。
- 將 `admin_dashboard.py` 轉成 Flask 頁面或前端框架。
- [Vertex AI Pipeline 與 Search 指南](vertex_ai/README.md) 示範如何以 `esp32cam-472912-vertex-*` bucket 建立訓練 Pipeline 與顧客搜尋資料庫。

歡迎依實際硬體狀況調整參數與資料表結構。GitHub Actions 會提供第一層自動驗證，接著即可移至實體 ESP32 進行驗證。

## 如何執行

在 Cloud Shell 進行 Vertex Matching Engine 評測時，可依下列步驟啟動端到端流程：

```bash
cd ~/esp32-cam-jim

# 可選：自動將評測結果上傳至 gs://esp32cam-472912-vertex-output
export UPLOAD_TO_GCS=1

# 產生 embeddings、上傳索引、部署並完成評測
bash scripts/e2e_eval.sh

# 查看本地評測輸出
ls -la vertex_eval_results/*

# （若有開啟 UPLOAD_TO_GCS）檢視雲端產物
gcloud storage ls gs://esp32cam-472912-vertex-output/eval_results/
```

執行腳本後，`vertex_eval_results/<run_id>/` 會包含 embeddings JSONL、`smoke_test.json`、`matches.csv`、`summary.json`、`confusion_matrix.csv` 與 `report.html` 等成果，方便進一步除錯或上傳到指定 GCS bucket。
