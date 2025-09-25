from kfp import dsl
from kfp.dsl import component, Dataset, Output
import json

@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "google-cloud-storage",
        "google-cloud-aiplatform>=1.48.0",
        "insightface==0.7.3",        # 會自動下載模型（MIT；預訓練模型非商用）
        "onnxruntime==1.18.0",
        "opencv-python-headless==4.10.0.84",
        "numpy>=1.24,<3",
        "Pillow"
    ],
)
def embed_and_upsert_op(
    project_id: str,
    region: str,
    index_name: str,        # projects/.../locations/.../indexes/...
    gcs_faces_root: str,    # gs://esp32cam-472912-vertex-data/faces
    out_log: Output[Dataset],
):
    import os, io
    from google.cloud import storage, aiplatform_v1 as gapic
    from PIL import Image
    import numpy as np
    import insightface

    # 1) 準備 GCS
    client = storage.Client(project=project_id)
    bucket_name = gcs_faces_root.split("/")[2]
    prefix = "/".join(gcs_faces_root.split("/")[3:])
    bucket = client.bucket(bucket_name)

    # 2) 讀全部臉照清單（每人資料夾）
    blobs = client.list_blobs(bucket, prefix=prefix)
    paths = [b.name for b in blobs if b.name.lower().endswith((".jpg",".jpeg",".png"))]

    # 3) 載入 InsightFace（含偵測與 512D embedding）
    app = insightface.app.FaceAnalysis(name="buffalo_l")  # 內含檢測/對齊/embedding
    app.prepare(ctx_id=0, det_size=(640,640))

    # 4) 準備 upsert client（low-level，確保相容性）
    api_endpoint = f"{region}-aiplatform.googleapis.com"
    index_client = gapic.IndexServiceClient(client_options={"api_endpoint": api_endpoint})

    # 5) 逐檔處理
    upserts = []
    for p in paths:
        person_id = p.split("/")[-2] if "/" in p else "unknown"
        blob = bucket.blob(p)
        img = Image.open(io.BytesIO(blob.download_as_bytes())).convert("RGB")
        img = np.array(img)[:,:,::-1]   # to BGR

        faces = app.get(img)
        if not faces:
            continue
        # 取最大臉
        face = max(faces, key=lambda f: f.bbox[2]*f.bbox[3])
        emb = face.normed_embedding.astype(float).tolist()  # 512 維

        # datapoint_id 可以用 personId#檔名
        datapoint_id = f"{person_id}#{os.path.basename(p)}"
        upserts.append(
            gapic.IndexDatapoint(
                datapoint_id=datapoint_id,
                feature_vector=emb,
                restricts=[gapic.IndexDatapoint.Restriction(namespace="person", allow_list=[person_id])]
            )
        )

        # 批次分批送，避免 payload 太大
        if len(upserts) >= 128:
            index_client.upsert_datapoints(index=index_name, datapoints=upserts)
            upserts = []

    if upserts:
        index_client.upsert_datapoints(index=index_name, datapoints=upserts)

    # 簡單寫個輸出日誌
    with open(out_log.path, "w") as f:
        f.write(json.dumps({"processed": len(paths)}))

@dsl.pipeline(name="faces-embed-upsert")
def pipeline(project_id: str, region: str, index_name: str, gcs_faces_root: str):
    embed_and_upsert_op(project_id, region, index_name, gcs_faces_root)
