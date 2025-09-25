"""End-to-end verification for Vertex pipeline + Cloud Run face workflow."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
from google.api_core import exceptions
from google.cloud import aiplatform_v1, logging_v2, storage
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw


def configure_logger(log_dir: Path) -> Tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "e2e_verify.log"
    logger = logging.getLogger("e2e_verify")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_file


def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def discover_sample_images() -> List[Path]:
    candidates = [
        Path("sample_data/faces"),
        Path("samples/faces"),
        Path("samples"),
        Path("tests/data/faces"),
    ]
    exts = {".jpg", ".jpeg", ".png"}
    images: List[Path] = []
    for base in candidates:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in exts:
                images.append(path)
    return images


def generate_synthetic_samples(count: int = 2) -> List[Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="vertex_faces_"))
    paths: List[Path] = []
    for idx in range(count):
        image = Image.new("RGB", (320, 320), color=(240, 224, 200))
        draw = ImageDraw.Draw(image)
        draw.ellipse((40, 40, 280, 280), fill=(255, 230, 200), outline=(180, 120, 80), width=3)
        eye_offset = 20 * (-1) ** idx
        draw.ellipse((120 + eye_offset, 140, 150 + eye_offset, 170), fill=(0, 0, 0))
        draw.ellipse((190 + eye_offset, 140, 220 + eye_offset, 170), fill=(0, 0, 0))
        draw.arc((140, 200, 220, 250), start=0, end=180, fill=(150, 40, 40), width=6)
        path = tmp_dir / f"synthetic_face_{idx + 1}.png"
        image.save(path)
        paths.append(path)
    return paths


def upload_samples(
    project: str,
    bucket: str,
    prefix: str,
    images: Sequence[Path],
    logger: logging.Logger,
) -> List[str]:
    storage_client = storage.Client(project=project)
    uploaded: List[str] = []
    for image in images:
        destination = f"{prefix.rstrip('/')}/tests/{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}/{image.name}" if prefix else f"tests/{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}/{image.name}"
        blob_path = destination
        logger.info("Uploading %s to gs://%s/%s", image, bucket, blob_path)
        blob = storage_client.bucket(bucket).blob(blob_path)
        blob.upload_from_filename(str(image))
        uploaded.append(f"gs://{bucket}/{blob_path}")
    return uploaded


def wait_for_eventarc(
    project: str,
    region: str,
    service_name: str,
    start_time: datetime,
    timeout_s: int,
    logger: logging.Logger,
) -> bool:
    client = logging_v2.Client(project=project)
    deadline = time.time() + timeout_s
    filter_str = (
        'resource.type="cloud_run_revision" '
        f'AND resource.labels.service_name="{service_name}" '
        f'AND resource.labels.location="{region}" '
        f'AND timestamp>="{start_time.isoformat()}"'
    )
    logger.info("Waiting up to %ss for Eventarc trigger logs", timeout_s)
    while time.time() < deadline:
        try:
            entries = list(
                client.list_entries(
                    filter_=filter_str,
                    order_by=logging_v2.DESCENDING,
                    page_size=20,
                )
            )
        except exceptions.GoogleAPICallError as exc:
            logger.warning("Logging API error while polling Eventarc: %s", exc)
            time.sleep(5)
            continue

        for entry in entries:
            payload = entry.text_payload or ""
            if "/events" in payload:
                logger.info("Detected Eventarc invocation log entry: %s", payload)
                return True
            try:
                if entry.json_payload:
                    if "/events" in json.dumps(entry.json_payload):
                        logger.info("Detected Eventarc invocation via jsonPayload")
                        return True
            except AttributeError:
                continue
        time.sleep(5)
    logger.warning("Eventarc invocation not observed within timeout")
    return False


def fetch_service_url(service_name: str, region: str) -> str:
    result = subprocess.run(
        [
            "gcloud",
            "run",
            "services",
            "describe",
            service_name,
            f"--region={region}",
            "--format=value(status.url)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    url = result.stdout.strip()
    if not url:
        raise RuntimeError(f"Unable to resolve Cloud Run URL for {service_name}")
    return url


def invoke_detect_face(url: str, image: Path, logger: logging.Logger) -> Dict[str, object]:
    with image.open("rb") as handle:
        payload = base64.b64encode(handle.read()).decode("utf-8")
    endpoint = url.rstrip("/") + "/detect_face"
    logger.info("Calling %s", endpoint)
    start = time.perf_counter()
    response = requests.post(endpoint, json={"image_b64": payload}, timeout=120)
    latency_ms = (time.perf_counter() - start) * 1000
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        body = {"raw": response.text}
    logger.info("/detect_face status=%s latency=%.1fms", response.status_code, latency_ms)
    return {
        "status_code": response.status_code,
        "latency_ms": latency_ms,
        "body": body,
    }


def load_face_analysis(logger: logging.Logger) -> FaceAnalysis:
    logger.info("Initialising InsightFace (buffalo_l)")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def compute_embedding(app: FaceAnalysis, image_path: Path, logger: logging.Logger) -> List[float]:
    image = Image.open(image_path).convert("RGB")
    array = np.array(image)[:, :, ::-1]  # RGB -> BGR
    faces = app.get(array)
    if not faces:
        raise RuntimeError(f"No face detected in {image_path}")
    logger.info("Detected %d faces in %s", len(faces), image_path)
    embedding = faces[0].normed_embedding.astype(float).tolist()
    if len(embedding) != 512:
        raise RuntimeError("Unexpected embedding dimension")
    return embedding


def resolve_deploy_id(
    project: str,
    region: str,
    endpoint_name: str,
    logger: logging.Logger,
) -> Tuple[str, str]:
    client = aiplatform_v1.IndexEndpointServiceClient(
        client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
    )
    endpoint = client.get_index_endpoint(name=endpoint_name)
    if not endpoint.deployed_indexes:
        raise RuntimeError(f"Index endpoint {endpoint_name} has no deployed indexes")
    deployed = endpoint.deployed_indexes[0]
    logger.info(
        "Using deployed index %s (index=%s)", deployed.id, deployed.index
    )
    return deployed.id, deployed.index


def find_neighbors(
    region: str,
    endpoint_name: str,
    deploy_id: str,
    embedding: Sequence[float],
    neighbor_count: int,
    logger: logging.Logger,
) -> List[Dict[str, object]]:
    client = aiplatform_v1.MatchServiceClient(
        client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
    )
    query = aiplatform_v1.FindNeighborsRequest.Query()
    query.neighbor_count = neighbor_count
    query.datapoint = aiplatform_v1.IndexDatapoint(
        feature_vector=list(embedding),
    )
    request = aiplatform_v1.FindNeighborsRequest(
        index_endpoint=endpoint_name,
        deployed_index_id=deploy_id,
        queries=[query],
    )
    response = client.find_neighbors(request=request)
    results: List[Dict[str, object]] = []
    if response.nearest_neighbors:
        for neighbor in response.nearest_neighbors[0].neighbors:
            results.append(
                {
                    "datapoint_id": neighbor.datapoint.datapoint_id,
                    "distance": neighbor.distance,
                }
            )
    logger.info("Nearest neighbors: %s", results)
    return results


def list_datapoints(
    region: str,
    index_name: str,
    logger: logging.Logger,
    page_size: int = 50,
) -> int:
    client = aiplatform_v1.IndexServiceClient(
        client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
    )
    try:
        pager = client.list_datapoints(index=index_name, page_size=page_size)
    except exceptions.PermissionDenied as exc:  # pragma: no cover - environment dependent
        logger.warning("Permission denied listing datapoints: %s", exc)
        return -1
    count = 0
    for count, _ in enumerate(pager, start=1):
        if count >= page_size:
            break
    logger.info(
        "Observed at least %d datapoints from index listing (page_size=%d)",
        count,
        page_size,
    )
    return count


def load_pipeline_summary(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    summary_path = Path(path)
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Vertex face workflow verification")
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--gcs-inbox", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--index", default="")
    parser.add_argument("--service-name", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--pipeline-summary", default="")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--deploy-id", default="")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    logger, _ = configure_logger(log_dir)

    summary = {
        "project": args.project,
        "region": args.region,
        "gcs_inbox": args.gcs_inbox,
    }

    pipeline_summary = load_pipeline_summary(args.pipeline_summary)
    if pipeline_summary:
        summary["pipeline"] = pipeline_summary
        logger.info(
            "Pipeline summary loaded (state=%s)", pipeline_summary.get("state")
        )

    try:
        bucket, prefix = parse_gcs_uri(args.gcs_inbox)
    except ValueError as exc:
        logger.error("Invalid inbox URI: %s", exc)
        raise

    samples = discover_sample_images()
    if not samples:
        logger.warning("No sample images found in repo. Generating synthetic faces.")
        samples = generate_synthetic_samples()
    else:
        logger.info("Discovered %d local sample images", len(samples))

    samples = samples[:2] if samples else samples
    uploaded_paths = upload_samples(args.project, bucket, prefix, samples, logger)
    summary["uploaded_samples"] = uploaded_paths

    start_time = datetime.now(timezone.utc)
    eventarc_triggered = wait_for_eventarc(
        args.project, args.region, args.service_name, start_time, args.timeout, logger
    )
    summary["eventarc_triggered"] = eventarc_triggered

    service_url = fetch_service_url(args.service_name, args.region)
    summary["service_url"] = service_url

    detect_results: List[Dict[str, object]] = []
    for image in samples[:1]:
        detect_results.append(invoke_detect_face(service_url, image, logger))
    summary["detect_face"] = detect_results

    face_app = load_face_analysis(logger)
    embedding = compute_embedding(face_app, samples[0], logger)

    deploy_id = args.deploy_id
    index_name = args.index
    if not deploy_id or not index_name:
        resolved_deploy_id, resolved_index = resolve_deploy_id(
            args.project, args.region, args.endpoint, logger
        )
        if not deploy_id:
            deploy_id = resolved_deploy_id
        if not index_name:
            index_name = resolved_index

    neighbor_results = find_neighbors(
        args.region, args.endpoint, deploy_id, embedding, args.neighbors, logger
    )
    summary["neighbors"] = neighbor_results
    summary["threshold"] = args.threshold

    if index_name:
        datapoint_count = list_datapoints(args.region, index_name, logger)
        summary["listed_datapoints"] = datapoint_count

    responses_path = log_dir / "responses.json"
    responses_path.write_text(json.dumps(detect_results, indent=2), encoding="utf-8")
    summary_path = log_dir / "e2e_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Summary written to %s", summary_path)

    detect_ok = bool(detect_results and detect_results[0]["status_code"] == 200)
    neighbors_ok = bool(neighbor_results)
    pipeline_ok = pipeline_summary.get("state") == "SUCCEEDED" if pipeline_summary else True
    if not (detect_ok and neighbors_ok and pipeline_ok):
        raise SystemExit("Verification failed – check logs for details")


if __name__ == "__main__":
    main()
