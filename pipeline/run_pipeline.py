"""Submit the faces-embed-upsert pipeline to Vertex AI Pipelines."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from google.cloud import aiplatform


DEFAULT_PROJECT = "esp32cam-472912"
DEFAULT_REGION = "asia-east1"
DEFAULT_BUCKET = "esp32cam-472912-vertex-data"
DEFAULT_STAGING = "gs://esp32cam-472912-vertex-staging/kfp-artifacts"


def main() -> None:
    project_id = os.environ.get("PROJECT_ID", DEFAULT_PROJECT)
    region = os.environ.get("REGION", DEFAULT_REGION)
    index_name = os.environ["INDEX_NAME"]
    gcs_faces_root = f"gs://{os.environ.get('BUCKET_DATA', DEFAULT_BUCKET)}/faces"
    service_account = os.environ.get(
        "SERVICE_ACCOUNT", "faces-sa@esp32cam-472912.iam.gserviceaccount.com"
    )

    pipeline_path = os.path.join(os.path.dirname(__file__), "pipeline.json")
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(
            f"Pipeline definition not found at {pipeline_path}. Did you run compile.py?"
        )

    aiplatform.init(project=project_id, location=region)

    params: Dict[str, Any] = {
        "project_id": project_id,
        "region": region,
        "index_name": index_name,
        "gcs_faces_root": gcs_faces_root,
    }

    print(json.dumps({"job_parameters": params}, indent=2))

    job = aiplatform.PipelineJob(
        display_name="faces-embed-upsert",
        template_path=pipeline_path,
        pipeline_root=os.environ.get("PIPELINE_ROOT", DEFAULT_STAGING),
        parameter_values=params,
        enable_caching=False,
        service_account=service_account,
    )
    job.run(sync=False)
    print(f"[INFO] Submitted pipeline job: {job.resource_name}")


if __name__ == "__main__":
    main()
