"""Kubeflow Pipeline：批次嵌入與 Streaming Index upsert。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import List

from kfp import dsl


APT_PACKAGES = [
    "libgl1",
    "libglib2.0-0",
    "libsm6",
    "libxext6",
    "libxrender1",
]

PIP_PACKAGES = [
    "numpy==1.26.4",
    "onnxruntime==1.17.3",
    "opencv-python-headless==4.10.0.84",
    "scikit-image==0.22.0",
    "Pillow==10.3.0",
    "tqdm==4.66.4",
    "requests==2.32.3",
    "google-cloud-storage==2.16.0",
    "google-cloud-aiplatform>=1.48.0",
]

INSIGHTFACE_WHEEL_URI = os.environ.get(
    "INSIGHTFACE_WHEEL_URI",
    "https://github.com/deepinsight/insightface/releases/download/v0.7.3/"
    "insightface-0.7.3-cp310-cp310-manylinux2014_x86_64.whl",
)


def _run_cmd(cmd: List[str], retries: int = 3, sleep_s: int = 5) -> None:
    """執行指令並在失敗時重試。"""
    attempt = 0
    while True:
        attempt += 1
        try:
            print(f"[CMD] {' '.join(cmd)} (attempt {attempt})")
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as exc:
            if attempt >= retries:
                raise
            print(f"[WARN] command failed: {exc}. retrying in {sleep_s}s")
            time.sleep(sleep_s)


def _ensure_dependencies() -> None:
    _run_cmd(["apt-get", "update"])
    _run_cmd(["apt-get", "install", "-y", *APT_PACKAGES])

    pip_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--only-binary=:all:",
        "--no-cache-dir",
    ]
    for package in PIP_PACKAGES:
        _run_cmd(pip_cmd + [package])

    insightface_target = INSIGHTFACE_WHEEL_URI
    if not insightface_target.endswith(".whl"):
        insightface_target = "insightface==0.7.3"

    try:
        _run_cmd(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--only-binary=:all:",
                insightface_target,
            ]
        )
    except subprocess.CalledProcessError as err:
        print(
            "[WARN] Failed to install insightface wheel directly: "
            f"{err}. Falling back to --no-deps without only-binary."
        )
        _run_cmd(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "insightface==0.7.3",
            ]
        )


@dsl.component(base_image="python:3.10-slim")
def embed_and_upsert_op(
    project_id: str,
    region: str,
    index_name: str,
    gcs_faces_root: str,
) -> dsl.Output[dsl.Artifact]:  # type: ignore[name-defined]
    """下載 GCS 影像、計算 embedding 並 upsert 到 Streaming Index。"""

    import io
    import math
    from pathlib import Path

    import numpy as np
    from google.cloud import aiplatform_v1, storage
    from google.cloud.aiplatform_v1.types import IndexDatapoint
    from insightface.app import FaceAnalysis
    from PIL import Image
    from tqdm import tqdm

    _ensure_dependencies()

    print("[INFO] Initialising clients and model")
    storage_client = storage.Client(project=project_id)
    index_client = aiplatform_v1.IndexServiceClient(
        client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
    )

    parsed = storage.urlsplit(gcs_faces_root)
    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip("/")
    bucket = storage_client.bucket(bucket_name)

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))

    blobs = list(storage_client.list_blobs(bucket_name, prefix=prefix))
    image_blobs = [b for b in blobs if b.name.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"[INFO] Found {len(image_blobs)} candidate images under {gcs_faces_root}")

    datapoints: List[IndexDatapoint] = []
    processed = 0
    skipped = 0

    def flush_batch(batch: List[IndexDatapoint]) -> None:
        if not batch:
            return
        print(f"[INFO] Upserting batch with {len(batch)} datapoints")
        request = aiplatform_v1.UpsertDatapointsRequest(index=index_name, datapoints=batch)
        response = index_client.upsert_datapoints(request=request)
        if response is not None:
            print(f"[INFO] Upsert response: {response}")

    for blob in tqdm(image_blobs, desc="Embedding faces"):
        person_id = Path(blob.name).parent.name
        try:
            content = blob.download_as_bytes()
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            print(f"[WARN] Failed to load image {blob.name}: {exc}")
            continue

        np_img = np.array(image)[:, :, ::-1]  # RGB -> BGR
        faces = app.get(np_img)
        if not faces:
            skipped += 1
            print(f"[WARN] No faces detected in {blob.name}")
            continue

        # 取面積最大的臉
        selected = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        embedding = selected.normed_embedding.astype(float).tolist()
        if len(embedding) != 512:
            skipped += 1
            print(f"[WARN] Unexpected embedding length in {blob.name}: {len(embedding)}")
            continue

        datapoint_id = f"{person_id}#{Path(blob.name).name}"
        datapoints.append(
            IndexDatapoint(
                datapoint_id=datapoint_id,
                feature_vector=embedding,
                restricts=[
                    aiplatform_v1.IndexDatapoint.Restrict(
                        namespace="person",
                        allow_list=[person_id],
                    )
                ],
            )
        )
        processed += 1

        if len(datapoints) >= 128:
            flush_batch(datapoints)
            datapoints.clear()

    flush_batch(datapoints)

    print(
        json.dumps(
            {
                "processed": processed,
                "skipped": skipped,
                "bucket": bucket_name,
                "prefix": prefix,
                "index_name": index_name,
            },
            indent=2,
        )
    )

    marker = os.path.join(os.getcwd(), "upsert_summary.json")
    with open(marker, "w", encoding="utf-8") as fp:
        json.dump({"processed": processed, "skipped": skipped}, fp)

    artifact = dsl.Artifact()
    artifact.uri = marker
    return artifact


@dsl.pipeline(name="faces-embed-upsert")
def faces_embed_upsert_pipeline(
    project_id: str,
    region: str,
    index_name: str,
    gcs_faces_root: str,
):
    embed_and_upsert_op(
        project_id=project_id,
        region=region,
        index_name=index_name,
        gcs_faces_root=gcs_faces_root,
    )
