"""Cloud Run service for Vertex AI face recognition."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from google.cloud import aiplatform_v1, storage
from google.cloud.aiplatform_v1 import IndexDatapoint
from insightface.app import FaceAnalysis
from PIL import Image

LOGGER = logging.getLogger("vertex-face-service")
logging.basicConfig(level=logging.INFO)

REGION = os.environ.get("REGION", "asia-east1")
PROJECT_ID = os.environ.get("PROJECT_ID", "esp32cam-472912")
INDEX_NAME = os.environ["INDEX_NAME"]
INDEX_ENDPOINT_ID = os.environ["INDEX_ENDPOINT_ID"]
DEPLOYED_INDEX_ID = os.environ["DEPLOYED_INDEX_ID"]
BUCKET_DATA = os.environ.get("BUCKET_DATA", "esp32cam-472912-vertex-data")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.6"))

API_ENDPOINT = f"{REGION}-aiplatform.googleapis.com"

storage_client = storage.Client(project=PROJECT_ID)
index_client = aiplatform_v1.IndexServiceClient(
    client_options={"api_endpoint": API_ENDPOINT}
)
match_client = aiplatform_v1.MatchServiceClient(
    client_options={"api_endpoint": API_ENDPOINT}
)

app = FastAPI()
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=-1, det_size=(640, 640))


def _load_image_from_gcs(gcs_uri: str) -> Image.Image:
    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs_uri must start with gs://")
    path = gcs_uri[5:]
    bucket_name, blob_name = path.split("/", 1)
    blob = storage_client.bucket(bucket_name).blob(blob_name)
    data = blob.download_as_bytes()
    return Image.open(io.BytesIO(data)).convert("RGB")


def _load_image_from_b64(image_b64: str) -> Image.Image:
    data = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _embed_image(image: Image.Image) -> List[float]:
    import numpy as np

    array = np.array(image)[:, :, ::-1]
    faces = face_app.get(array)
    if not faces:
        raise ValueError("No face detected")
    face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
    embedding = face.normed_embedding.astype(float).tolist()
    if len(embedding) != 512:
        raise ValueError("Unexpected embedding dimension")
    return embedding


def _to_similarity(distance: float) -> float:
    return max(0.0, 1.0 - distance)


def _find_neighbors(embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    query = aiplatform_v1.FindNeighborsRequest.Query(
        datapoint=IndexDatapoint(datapoint_id="query", feature_vector=embedding),
        neighbor_count=top_k,
    )
    request = aiplatform_v1.FindNeighborsRequest(
        index_endpoint=INDEX_ENDPOINT_ID,
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[query],
    )
    response = match_client.find_neighbors(request=request)
    results: List[Dict[str, Any]] = []
    for neighbor in response.nearest_neighbors[0].neighbors:
        score = _to_similarity(neighbor.distance)
        results.append(
            {
                "datapoint_id": neighbor.datapoint.datapoint_id,
                "distance": neighbor.distance,
                "score": score,
                "restricts": [r.allow_list for r in neighbor.datapoint.restricts],
            }
        )
    return results


def _upsert_new_person(person_id: str, embedding: List[float], source_uri: str) -> None:
    datapoint = IndexDatapoint(
        datapoint_id=f"{person_id}#{os.path.basename(source_uri)}",
        feature_vector=embedding,
        restricts=[
            IndexDatapoint.Restrict(namespace="person", allow_list=[person_id])
        ],
    )
    request = aiplatform_v1.UpsertDatapointsRequest(
        index=INDEX_NAME,
        datapoints=[datapoint],
    )
    index_client.upsert_datapoints(request=request)


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/recognize")
async def recognize(payload: Dict[str, Any]) -> Dict[str, Any]:
    image: Optional[Image.Image] = None
    source = ""

    if "gcs_uri" in payload:
        source = payload["gcs_uri"]
        image = _load_image_from_gcs(source)
    elif "image_b64" in payload:
        source = "inline"
        image = _load_image_from_b64(payload["image_b64"])
    else:
        raise HTTPException(status_code=400, detail="gcs_uri or image_b64 required")

    embedding = _embed_image(image)
    neighbors = _find_neighbors(embedding)
    response = {
        "topK": neighbors,
        "threshold": THRESHOLD,
    }
    if neighbors:
        top1 = neighbors[0]
        response["best_match"] = top1
        response["match"] = top1["score"] >= THRESHOLD
    else:
        response["best_match"] = None
        response["match"] = False
    return response


@app.post("/events")
async def events(request: Request) -> Dict[str, Any]:
    event = json.loads(await request.body())
    data = event.get("data", {})
    bucket = data.get("bucket")
    name = data.get("name")
    if not bucket or not name:
        raise HTTPException(status_code=400, detail="Invalid CloudEvent payload")

    if not name.startswith("faces/inbox/"):
        return {"status": "ignored", "object": name}

    gcs_uri = f"gs://{bucket}/{name}"
    LOGGER.info("Processing GCS object %s", gcs_uri)
    image = _load_image_from_gcs(gcs_uri)
    embedding = _embed_image(image)
    neighbors = _find_neighbors(embedding)

    outcome: Dict[str, Any]
    if neighbors and neighbors[0]["score"] >= THRESHOLD:
        outcome = {
            "status": "existing",
            "person_id": neighbors[0]["datapoint_id"].split("#", 1)[0],
            "score": neighbors[0]["score"],
        }
    else:
        new_person_id = uuid.uuid4().hex
        _upsert_new_person(new_person_id, embedding, name)
        outcome = {
            "status": "enrolled",
            "person_id": new_person_id,
            "score": 1.0,
        }

    LOGGER.info("Event outcome: %s", outcome)
    return {"object": name, **outcome}
