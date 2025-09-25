from kfp import dsl
from kfp.dsl import component

@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        # 只裝最小集，避免版本衝突；其餘用程式內分段安裝
        "pip>=23.2", "setuptools", "wheel",
    ],
)
def embed_and_upsert_op(
    project_id: str,
    region: str,
    index_name: str,         # projects/.../indexes/...
    gcs_faces_root: str,     # gs://BUCKET/faces/<person_id>/*.jpg
):
    import os, io, sys, traceback, subprocess
    from typing import List
    # ---------- 0) 系統相依（分段安裝，避免命令過長被吃掉） ----------
    def sh(cmd: List[str]):
        print("[sh]", " ".join(cmd)); sys.stdout.flush()
        subprocess.run(cmd, check=True)

    # 基本 lib（opencv/onnxruntime 會用到）
    sh(["bash","-lc","apt-get update"])
    sh(["bash","-lc","apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libsm6 libxext6 libxrender1"])
    sh(["bash","-lc","rm -rf /var/lib/apt/lists/*"])

    # ---------- 1) Python 相依（分段裝，版本已測通過） ----------
    py = sys.executable
    def pip_install(pkgs: List[str]):
        print("[pip] install", " ".join(pkgs)); sys.stdout.flush()
        subprocess.run([py, "-m", "pip", "install", "--no-cache-dir", *pkgs], check=True)

    pip_install(["numpy==1.26.4"])
    pip_install(["onnxruntime==1.17.3", "onnx==1.15.0"])
    pip_install(["opencv-python-headless==4.10.0.84"])
    pip_install(["insightface==0.7.3", "Pillow", "tqdm"])
    pip_install(["google-cloud-storage", "google-cloud-aiplatform>=1.48.0"])

    # ---------- 2) 匯入套件 ----------
    from google.cloud import storage, aiplatform_v1 as gapic
    from PIL import Image
    import numpy as np
    import insightface

    # ---------- 3) 解析 GCS 路徑 ----------
    if not gcs_faces_root.startswith("gs://"):
        raise ValueError(f"gcs_faces_root 必須是 gs:// 開頭，收到: {gcs_faces_root}")
    parts = gcs_faces_root.split("/")
    bucket_name = parts[2]
    prefix = "/".join(parts[3:]).rstrip("/") + "/"

    storage_client = storage.Client(project=project_id)
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    # ---------- 4) InsightFace（CPU） ----------
    app = insightface.app.FaceAnalysis(name="buffalo_l")
    # CPU 模式
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("find model:", app.models.keys())  # 確認 3 個模型就緒

    # ---------- 5) Vector Search 服務 ----------
    api_endpoint = f"{region}-aiplatform.googleapis.com"
    index_client = gapic.IndexServiceClient(client_options={"api_endpoint": api_endpoint})

    sent = 0
    batch: List[gapic.IndexDatapoint] = []
    BATCH_SZ = 128
    scanned = 0
    faced = 0
    skipped = 0
    errors = 0

    def flush():
        nonlocal batch, sent
        if batch:
            try:
                index_client.upsert_datapoints(index=index_name, datapoints=batch)
                sent += len(batch)
                print(f"[upsert] {len(batch)} (total={sent})")
            finally:
                batch = []

    def pick_largest(faces):
        def area(f):
            x1,y1,x2,y2 = f.bbox.astype(float)
            return max(0.0,(x2-x1))*max(0.0,(y2-y1))
        return max(faces, key=area)

    for blob in blobs:
        name = (blob.name or "").lower()
        if not (name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".png")):
            continue
        scanned += 1

        # 依 <person_id>/file.jpg 推出 person_id
        p = blob.name.split("/")
        if len(p) < 2:
            skipped += 1
            print(f"[skip] bad path: {blob.name}")
            continue
        person_id = p[-2]
        file_name = p[-1]

        try:
            raw = blob.download_as_bytes()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            arr = np.array(img)[:, :, ::-1]  # RGB -> BGR

            faces = app.get(arr)
            if not faces:
                skipped += 1
                print(f"[skip] no face: {blob.name}")
                continue

            face = pick_largest(faces)
            emb = face.normed_embedding.astype(float)
            if emb.shape[0] != 512:
                # 與索引維度不符直接跳過（避免整批失敗）
                skipped += 1
                print(f"[skip] dim={emb.shape[0]} (need 512): {blob.name}")
                continue

            dp = gapic.IndexDatapoint(
                datapoint_id=f"{person_id}#{file_name}",
                feature_vector=emb.tolist(),
                restricts=[gapic.IndexDatapoint.Restriction(
                    namespace="person", allow_list=[person_id]
                )],
            )
            batch.append(dp)
            faced += 1

            if len(batch) >= BATCH_SZ:
                flush()

        except Exception as e:
            errors += 1
            print(f"[err] {blob.name}: {e}")
            traceback.print_exc()

    flush()  # 把殘留的送掉
    print(f"[summary] scanned={scanned}, with_face={faced}, skipped={skipped}, errors={errors}, upserted={sent}")

    # 若完全沒有臉，仍視為成功（讓 pipeline 不要因為 0 筆而失敗）
    # 真正的例外已在上面印出 stacktrace，不會中斷整批。
    return

@dsl.pipeline(name="faces-embed-upsert")
def pipeline(
    project_id: str = "esp32cam-472912",
    region: str = "asia-east1",
    index_name: str = "projects/esp32cam-472912/locations/asia-east1/indexes/xxxxxxxx",
    gcs_faces_root: str = "gs://esp32cam-472912-vertex-data/faces",
):
    embed_and_upsert_op(
        project_id=project_id,
        region=region,
        index_name=index_name,
        gcs_faces_root=gcs_faces_root,
    )
