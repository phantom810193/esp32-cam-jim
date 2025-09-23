# syntax=docker/dockerfile:1
# ---- Dockerfile (Cloud Run) ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 系統相依：OpenCV/ONNX Runtime 需要的動態庫，curl 用來在 build 時抓模型
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        ca-certificates \
        curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先安裝依賴（保留快取層）
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY . .

# 確保 ArcFace 模型存在；若 repo 沒帶檔案，build 時會自動下載
RUN mkdir -p /app/models && \
    if [ ! -f /app/models/arcface.onnx ]; then \
      echo "Downloading ArcFace ONNX model..." && \
      curl -L -o /app/models/arcface.onnx \
        https://huggingface.co/garavv/arcface-onnx/resolve/main/arcface.onnx ; \
    fi

# 你的程式用到的預設環境變數
ENV MODEL_PATH=/app/models/arcface.onnx \
    USE_VISION=1 \
    ORT_PROVIDERS=CPUExecutionProvider \
    ORT_NUM_THREADS=1 \
    PORT=8080

# 用非 root 運行更安全
RUN useradd -m -u 1001 appuser
USER appuser

# Cloud Run 入口（gunicorn 請在 requirements.txt 中安裝）
CMD exec gunicorn -b :$PORT api:app --workers 2 --threads 4 --timeout 120