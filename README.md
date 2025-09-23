# ESP32 Cloud Vision Demo (CI Ready)

本專案提供一套可在 GitHub Actions 上自動驗證的 ESP32 零售體驗流程，涵蓋攝影機、Cloud Vision 臉部辨識、SQLite 客製化行銷、Flask API 與終端機/LCD 文字輸出。所有二進位資產（ArcFace ONNX 模型、測試圖片、影片等）都移出版本控制，改由 Cloud Build 於建置時自動自 Google Cloud Storage (GCS) 下載，符合「方案A」的要求。

## 專案結構

.
├── api.py # Flask /enroll 與 /detect_face 端點
├── camera.py # 攝影機模擬與 cam.log
├── stability.py # 30 分鐘穩定度報告 cam_stable.log
├── vision.py # Cloud Vision + 向量嵌入模擬與 id_test.log
├── display.py # 終端機輸出與 text_test.log
├── main.py # 任務4 串接流程 e2e.log
├── admin_dashboard.py # 任務5(1) 管理員後台 HTML
├── promo_ui.py # 任務5(2) 推播 UI
├── data.sql # SQLite users + visits 預設資料建置腳本
├── sample_data/ # 臉部向量測試資料（純文字）
├── tests/ # PyTest 測試覆蓋任務1~5
└── .github/workflows/ # CI 腳本


## 快速開始

1. **安裝依賴**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows 用 .venv\Scripts\activate
   pip install -r requirements.txt

    執行測試

pytest

測試流程會自動建立所有日誌（cam.log, id_test.log, cam_stable.log, text_test.log, api_test.log,
promo_display.log(.json), admin_dashboard.html, e2e.log）並檢查 FPS/準確率/回應時間門檻。

端到端流程

    python main.py

    於專案根目錄產生最新 cam.log, id_test.log, cam_stable.log, api_test.log,
    text_test.log, promo_display.log(.json), admin_dashboard.html, e2e.log 與 users.db。

Cloud Build / GCS 模型注入

    將 arcface_r100.onnx 上傳至專案專用的 GCS 私有桶，例如 gs://<PROJECT>-models/arcface_r100.onnx。

    在 cloudbuild.yaml 中於 Docker build 前加入：

    - name: gcr.io/google.com/cloudsdktool/cloud-sdk
      entrypoint: gsutil
      args: ["cp", "gs://<PROJECT>-models/arcface_r100.onnx", "embed/models/arcface_r100.onnx"]

    Docker 映像中即可取得模型，仍維持 Git repo 無二進位檔案。

GitHub Actions

.github/workflows/ci.yml 會在 push / PR / 手動觸發時自動：

    安裝 Python 3.11 與依賴，執行 pytest 驗證任務1~5指標。

    產出並上傳測試日誌（artifact）。

    可選：若設定 GCP_SA_KEY 等 Secrets/Variables，會額外執行 GCP 連線與端點延遲測試。

任務說明與驗證指標
任務	自動化腳本	驗證項目	對應 log
任務1(1)	camera.py	FPS > 10	cam.log
任務1(2)	vision.py	臉辨 5 張 >= 80%	id_test.log
任務2	stability.py	30 分穩定、FPS>10、Accuracy>0.8	cam_stable.log
任務3(1)	api.py	/enroll + /detect_face < 1 秒、記錄訪客	api_test.log
任務3(2)	data.sql + database_utils.py	SQLite users 與 visits 表、首訪/返店邏輯	users.db
任務3(3)	display.py	5 筆訊息 < 5 秒	text_test.log
任務4	main.py	攝影機→辨識→建檔→返店推播 < 30 秒	e2e.log
任務5	admin_dashboard.py, promo_ui.py	管理 UI + 推播 UI	admin_dashboard.html, promo_display.log(.json)
任務6	README.md	專案說明、執行步驟	N/A
首訪與返店流程

    首次建檔 /enroll

        對 Cloud Run 送多張照片或 embedding 陣列（也支援 base64 images / 網址 urls → 伺服器側轉 embedding）。

        回傳訊息為「新用戶已建檔」，SQLite users 與 visits 新增紀錄。

        CI 測試會確認 visit_count == 1 並寫入 api_test.log。

    返店辨識 /detect_face

        ESP32 或測試腳本送入單一 embedding；API 約 0.2 秒回應（雲端冷啟除外）。

        回傳「老朋友歡迎回來」，visits 增加一筆並根據上一筆消費產生優惠（例：「上次買牛奶，現9折！」）。

        text_test.log 與 LCD/LED display 顯示上一筆消費 + 推播。

    效能觀察

        Cloud Run 約 200 ms、Vision 約 200 ms；Firestore/SQLite 首訪 ~1 秒屬常見（冷啟或憑證交換）。

        若後續仍慢，可考慮連線重用、批次寫入或選擇接近客戶的區域資料庫。

實體 ESP32 測試建議

    於 Cloud Run 部署 Flask API（api.py）與 Cloud Vision 服務金鑰。

    ESP32 端程式先呼叫 /enroll 建檔，再呼叫 /detect_face 驗證返店推播。

    透過 cam_stable.log 與 e2e.log 指標確認 30 分穩定度與端到端延遲。

    LCD/終端機可重用 display.py 與 promo_ui.py 輸出格式。

Demo 影片製作

demo_plan.md（可自行新增）建議腳本：

    開啟管理員後台畫面。

    ESP32 進行實際拍攝辨識（含 /enroll → /detect_face 流程）。

    LCD 推播顯示優惠訊息。

    全片 1 分鐘內完成。

後續擴充

    將 FaceRecognitionService 接到實際 GCS 模型 + Cloud Vision API。

    以 Pub/Sub 或 Firebase Cloud Messaging 建立即時推播。

    將 admin_dashboard.py 轉成 Flask 頁面或前端框架。

    歡迎依實際硬體狀況調整參數與資料表結構。GitHub Actions 提供第一層自動驗證，接著即可移至實體 ESP32 進行驗證。


提交指令：
```bash
git add README.md
git commit -m "Resolve README conflict: unify /enroll + /detect_face, users+visits, stability, UI, and CI docs"
# 若正進行 merge/rebase：
# git merge --continue
# 或
# git rebase --continue