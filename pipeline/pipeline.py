from kfp import dsl
from kfp.dsl import component

@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        # 只裝 GCP SDK，其他用函式內動態安裝（先解決 g++）
        "google-cloud-storage",
        "google-cloud-aiplatform>=1.48.0",
    ],
)
def embed_and_upsert_op(project_id: str,
                        region: str,
                        index_name: str,        # projects/.../indexes/...
                        gcs_faces_root: str):   # gs://.../faces/<person_id>/*.jpg
    import subprocess, sys, io
    # ---- 先安裝系統編譯與影像相關套件（提供 g++ 等）----
    subprocess.run(
        ["bash","-lc",
         "apt-get update && apt-get install -y --no-install-recommends "
         "build-essential libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 "
         "&& rm -rf /var/lib/apt/lists/*"],
        check=True)
    # ---- 再安裝 Python 相依（這時候有 g++ 可編 C/Cython）----
    subprocess.run([sys.executable,"-m","pip","install","--no-cache-dir",
                    "insightface==0.7.3",
                    "onnxruntime==1.18.0",
                    "opencv-python-headless==4.10.0.84",
                    "numpy>=1.24,<3",
                    "Pillow"], check=True)

    # 之後再 import，確保套件可用
    from google.cloud import storage, aiplatform_v1 as gapic
    from PIL import Image
    import numpy as np
    import insightface

    # 解析 GCS 路徑
    parts = gcs_faces_root.split("/")
    bucket_name = parts[2]
    prefix = "/".join(parts[3:]).rstrip("/") + "/"

    # 列檔案
    storage_client = storage.Client(project=project_id)
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    # InsightFace：偵測/對齊/嵌入（CPU）
    app = insightface.app.FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))

    # Vector Search 串流 upsert
    api_endpoint = f"{region}-aiplatform.googleapis.com"
    index_client = gapic.IndexServiceClient(
        client_options={"api_endpoint": api_endpoint}
    )

    batch, sent = [], 0
    for blob in blobs:
        name = blob.name.lower()
        if not (name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".png")):
            continue

        # 依 <person_id>/file.jpg 推導 person_id
        path_parts = blob.name.split("/")
        if len(path_parts) < 2:
            continue
        person_id = path_parts[-2]
        file_name = path_parts[-1]

        try:
            img = Image.open(io.BytesIO(blob.download_as_bytes())).convert("RGB")
        except Exception as e:
            print(f"[skip] open {blob.name} failed: {e}")
            continue

        arr = np.array(img)[:, :, ::-1]  # RGB -> BGR
        faces = app.get(arr)
        if not faces:
            print(f"[skip] no face: {blob.name}")
            continue

        # 取最大臉
        def area(f):
            x1, y1, x2, y2 = f.bbox.astype(float)
            return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        face = max(faces, key=area)

        emb = face.normed_embedding.astype(float).tolist()  # 512D

        dp = gapic.IndexDatapoint(
            datapoint_id=f"{person_id}#{file_name}",
            feature_vector=emb,
            restricts=[gapic.IndexDatapoint.Restriction(
                namespace="person", allow_list=[person_id]
            )],
        )
        batch.append(dp)
        if len(batch) >= 128:
            index_client.upsert_datapoints(index=index_name, datapoints=batch)
            sent += len(batch)
            batch = []

    if batch:
        index_client.upsert_datapoints(index=index_name, datapoints=batch)
        sent += len(batch)

    print(f"Upserted datapoints: {sent}")

@dsl.pipeline(name="faces-embed-upsert")
def pipeline(project_id: str = "esp32cam-472912",
             region: str = "asia-east1",
             index_name: str = "projects/esp32cam-472912/locations/asia-east1/indexes/xxxxxxxx",
             gcs_faces_root: str = "gs://esp32cam-472912-vertex-data/faces"):
    embed_and_upsert_op(
        project_id=project_id,
        region=region,
        index_name=index_name,
        gcs_faces_root=gcs_faces_root,
    )
