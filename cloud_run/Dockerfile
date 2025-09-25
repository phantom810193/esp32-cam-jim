FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --upgrade --only-binary=:all: -r requirements.txt && \
    pip install --only-binary=:all: --no-deps insightface==0.7.3

COPY app /app/app

ENV HOST=0.0.0.0 PORT=8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
