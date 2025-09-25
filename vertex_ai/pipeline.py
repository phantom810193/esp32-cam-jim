"""Kubeflow Pipeline 定義：讀取 GCS 影像、產生訓練 manifest、模擬訓練與評估。

此腳本僅提供範例，實際部署時請依照模型需求替換 component 內容。
"""
from __future__ import annotations

import argparse
from typing import Dict, List

from kfp import compiler, dsl
from kfp.dsl import InputPath, OutputPath


def parse_gcs_uri(uri: str) -> Dict[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Unsupported GCS URI: {uri}")
    bucket, _, prefix = uri[5:].partition("/")
    return {"bucket": bucket, "prefix": prefix}


@dsl.component(
    base_image="python:3.11-slim",
    packages=["google-cloud-storage>=2.16"],
)
def prepare_dataset(
    data_bucket: str,
    manifest_path: OutputPath(str),
) -> int:
    """列出 `faces/<person_id>/<image>` 影像，輸出 manifest JSON。"""
    import json
    from collections import defaultdict
    from google.cloud import storage

    uri = parse_gcs_uri(data_bucket)
    client = storage.Client()
    blobs = client.list_blobs(uri["bucket"], prefix=uri["prefix"] + "faces/")

    manifest: Dict[str, List[str]] = defaultdict(list)
    for blob in blobs:
        if not blob.name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        parts = blob.name.split("/")
        if len(parts) < 3:
            # 應符合 faces/<person>/<file>
            continue
        manifest[parts[1]].append(f"gs://{uri['bucket']}/{blob.name}")

    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)
    return sum(len(v) for v in manifest.values())


@dsl.component(
    base_image="python:3.11-slim",
    packages=["google-cloud-storage>=2.16", "numpy>=1.26"],
)
def train_embeddings(
    manifest_path: InputPath(str),
    output_bucket: str,
    model_artifact: OutputPath(str),
) -> float:
    """模擬生成 embedding 並將結果寫回 GCS。"""
    import json
    import os
    import uuid
    from google.cloud import storage
    import numpy as np

    with open(manifest_path, "r", encoding="utf-8") as fp:
        manifest = json.load(fp)

    rng = np.random.default_rng(42)
    embeddings = {}
    for person_id, images in manifest.items():
        # 模擬每人 5 維向量
        embeddings[person_id] = rng.normal(size=5).tolist()

    # 計算模擬準確率（至少 0.85，象徵通過門檻）
    accuracy = 0.85 if embeddings else 0.0

    model_payload = {
        "accuracy": accuracy,
        "embeddings": embeddings,
        "image_count": sum(len(v) for v in manifest.values()),
    }
    with open(model_artifact, "w", encoding="utf-8") as fp:
        json.dump(model_payload, fp, indent=2, ensure_ascii=False)

    # 將 embeddings 輸出到 output bucket，供 API / Search 使用
    gcs = parse_gcs_uri(output_bucket)
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs["bucket"])
    blob = bucket.blob(f"embeddings/{uuid.uuid4().hex}.json")
    blob.upload_from_string(json.dumps(model_payload, ensure_ascii=False))

    return accuracy


@dsl.component(base_image="python:3.11-slim")
def evaluate_model(
    accuracy: float,
    report: OutputPath(str),
) -> float:
    """將準確率寫入評估報告。"""
    with open(report, "w", encoding="utf-8") as fp:
        fp.write(f"accuracy: {accuracy}\n")
    if accuracy < 0.8:
        raise ValueError("Accuracy below threshold")
    return accuracy


@dsl.pipeline(name="esp32cam-face-recognition")
def pipeline(
    project_id: str,
    location: str,
    data_bucket: str,
    output_bucket: str,
):
    manifest_task = prepare_dataset(data_bucket=data_bucket)

    train_task = train_embeddings(
        manifest_path=manifest_task.outputs["manifest_path"],
        output_bucket=output_bucket,
    )

    evaluate_model(
        accuracy=train_task.output,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Vertex AI pipeline")
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--data-bucket", required=True)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument(
        "--pipeline",
        default="face_pipeline.json",
        help="輸出的 Pipeline JSON 路徑",
    )
    args = parser.parse_args()

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=args.pipeline,
    )

    print(
        "Pipeline 已編譯。請於 Vertex AI Pipelines console 上傳 JSON 並設定參數："
        f"project_id={args.project}, location={args.region}, data_bucket={args.data_bucket}, output_bucket={args.output_bucket}."
    )


if __name__ == "__main__":
    main()
