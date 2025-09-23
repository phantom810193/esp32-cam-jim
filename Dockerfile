# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# OpenCV 需要的系統套件
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 你的程式碼
COPY . .

# Cloud Run 入口
ENV PORT=8080
CMD exec gunicorn -b :$PORT api:app --workers 2 --threads 4 --timeout 120
