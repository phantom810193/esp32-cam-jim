# 文件名: Dockerfile.vertex-insightface
FROM python:3.10-slim

# 基本系統依賴（避免編譯期缺 lib）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 先安裝與 ABI 相關的底層套件與相容版本（numpy<2）
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "onnx==1.15.0" \
    "onnxruntime==1.18.1" \
    "opencv-python-headless==4.10.0.84" \
    "scikit-image==0.22.0" \
    "pillow==10.3.0" \
    "tqdm==4.66.5"

# 再裝 insightface（0.7.3 與上面版本組合穩）
RUN pip install --no-cache-dir "insightface==0.7.3"

# 可選：預先下載人臉模型（減少首次冷啟）
RUN python - <<'PY'
import insightface, os
insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']).prepare(ctx_id=0)
print("insightface ready")
PY

WORKDIR /app
COPY . /app
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
