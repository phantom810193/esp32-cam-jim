# ESP32 Cloud Vision Demo (CI Ready)

本專案提供一套可在 GitHub Actions 上自動驗證的 ESP32 零售體驗流程，涵蓋攝影機、Cloud Vision 臉部辨識、SQLite 客製化行銷、Flask API 與終端機/LCD 文字輸出。所有二進位資產（ArcFace ONNX 模型、測試圖片、影片等）都移出版本控制，改由 Cloud Build 於建置時自動自 Google Cloud Storage (GCS) 下載，符合「方案 A」。

## 專案結構

```
.
├── api.py                # Flask /enroll 與 /detect_face 端點
├── camera.py             # 攝影機模擬，輸出 cam.log
├── stability.py          # 30 分鐘穩定度報告，輸出 cam_stable.log
├── vision.py             # 向量嵌入/臉辨模擬，輸出 id_test.log
├── display.py            # 終端機輸出，輸出 text_test.log
├── main.py               # 端到端流程（e2e），輸出 e2e.log
├── admin_dashboard.py    # 管理員後台 HTML
├── promo_ui.py           # 推播 UI，輸出 promo_display.log(.json)
├── data.sql              # SQLite users + visits 預設資料建置腳本
├── sample_data/          # 臉部向量測試資料（純文字）
├── tests/                # PyTest 覆蓋任務 1~5
└── .github/workflows/    # CI 腳本
```

## 快速開始

1) 安裝依賴
```bash
python -m venv .venv
# Windows: .venv\Scriptsctivate
source .venv/bin/activate
pip install -r requirements.txt
```

2) 執行測試（自動建立並驗證各 log）
```bash
pytest
```
測試流程會建立並檢查：`cam.log`, `id_test.log`, `cam_stable.log`, `text_test.log`, `api_test.log`, `promo_display.log(.json)`, `admin_dashboard.html`, `e2e.log`。

3) 端到端流程
```bash
python main.py
```
將於專案根目錄產生最新 `cam.log`, `id_test.log`, `cam_stable.log`, `api_test.log`, `text_test.log`, `promo_display.log(.json)`, `admin_dashboard.html`, `e2e.log`, `users.db`。

## Cloud Build / GCS 模型注入

1. 將 `arcface_r100.onnx` 上傳至專案專用 GCS 私有桶（例如：`gs://<PROJECT>-models/arcface_r100.onnx`）。
2. 在 `cloudbuild.yaml` 於 Docker build 前加入：
```yaml
- name: gcr.io/google.com/cloudsdktool/cloud-sdk
  entrypoint: gsutil
  args: ["cp", "gs://<PROJECT>-models/arcface_r100.onnx", "embed/models/arcface_r100.onnx"]
```
3. Docker 映像建置時即會把模型帶入，Git repo 保持無二進位檔案。

## GitHub Actions（CI）

`.github/workflows/ci.yml` 於 push / PR / 手動觸發時會：
1. 安裝 Python 3.11 與依賴並執行 `pytest`。
2. 上傳測試日誌（artifact）。
3. **可選**：若設定了 `GCP_SA_KEY` 等 Secrets/Variables，會額外執行 GCP 連線與端點延遲/嵌入測試。

## 任務與驗證指標

| 任務 | 自動化腳本 | 驗證項目 | 對應 log |
|------|------------|----------|----------|
| 任務1(1) | `camera.py` | FPS > 10 | `cam.log` |
| 任務1(2) | `vision.py` | 臉辨 5 張 >= 80% | `id_test.log` |
| 任務2 | `stability.py` | 30 分穩定、FPS>10、Accuracy>0.8 | `cam_stable.log` |
| 任務3(1) | `api.py` | `/enroll` + `/detect_face` < 1 秒、記錄訪客 | `api_test.log` |
| 任務3(2) | `data.sql` + `database_utils.py` | SQLite `users` 與 `visits` 表、首訪/返店邏輯 | `users.db` |
| 任務3(3) | `display.py` | 5 筆訊息 < 5 秒 | `text_test.log` |
| 任務4 | `main.py` | 攝影機→辨識→建檔→返店推播 < 30 秒 | `e2e.log` |
| 任務5 | `admin_dashboard.py`, `promo_ui.py` | 管理 UI + 推播 UI | `admin_dashboard.html`, `promo_display.log(.json)` |
| 任務6 | `README.md` | 專案說明、執行步驟 | N/A |

## 首訪與返店流程

**1) 首次建檔 `/enroll`**  
- 送入多張圖片或多筆 embedding（亦支援 base64 `images` / 網址 `urls` → 伺服器側轉 embedding）。  
- 回傳「新用戶已建檔」，`users` 與 `visits` 會新增紀錄。CI 會驗證 `visit_count == 1` 並紀錄 `api_test.log`。

**2) 返店辨識 `/detect_face`**  
- 送入單筆 embedding；API 約 0.2 秒回應（冷啟除外）。  
- 回傳「老朋友歡迎回來」，`visits` 增加一筆並依上一筆消費產生優惠（例：「上次買牛奶，現 9 折！」）。  
- `text_test.log` 與 LCD/LED display 會顯示上一筆消費 + 推播。

**3) 效能觀察**  
- Cloud Run ≈ 200 ms、Vision ≈ 200 ms；Firestore/SQLite 首訪 ~1s 常見（冷啟/憑證交換）。  
- 若仍慢：考慮連線重用、批次寫入、或選用接近客戶的資料庫區域。

## 實體 ESP32 測試建議

1. 於 Cloud Run 部署 Flask API（`api.py`）並配置服務金鑰。  
2. ESP32 端先呼叫 `/enroll` 建檔，再呼叫 `/detect_face` 驗證返店推播。  
3. 透過 `cam_stable.log` 與 `e2e.log` 觀察 30 分穩定度與端到端延遲。  
4. LCD/終端機可重用 `display.py` 與 `promo_ui.py` 的輸出格式。

## Demo 影片製作建議

1. 開啟管理員後台畫面。  
2. 示範 ESP32 實拍辨識（含 `/enroll` → `/detect_face` 流程）。  
3. LCD 推播顯示優惠訊息。  
4. 全片控制於 1 分鐘內。

## 後續擴充

- 將 `FaceRecognitionService` 接上實際 GCS 模型 + Cloud Vision API。  
- 以 Pub/Sub 或 Firebase Cloud Messaging 建立即時推播。  
- 將 `admin_dashboard.py` 轉成 Flask 頁面或前端框架。

---

> 依實際硬體調整參數與資料表結構即可。GitHub Actions 提供第一層自動驗證，接著移至實體 ESP32 進一步驗證。
