#!/usr/bin/env python3
"""Quick smoke test for Vertex Matching Engine deployments."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from google.api_core import retry as ga_retry
from google.cloud import aiplatform_v1
from google.cloud import storage


@dataclass
class Datapoint:
    identifier: str
    embedding: List[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal neighbor query against a deployed index.")
    parser.add_argument("--staging-json", required=True, help="GCS (gs://) or local path to the JSONL embeddings file.")
    parser.add_argument("--endpoint", required=True, help="Full resource name of the index endpoint.")
    parser.add_argument("--deploy-id", required=True, help="Deployed index ID to query.")
    parser.add_argument("--out", required=True, help="Path to store the smoke test JSON results.")
    parser.add_argument("--region", help="Vertex AI region (defaults to the region extracted from the endpoint name).")
    return parser.parse_args()


def extract_region(endpoint: str) -> str:
    if "/locations/" not in endpoint:
        raise ValueError(f"Unable to determine region from endpoint: {endpoint}")
    return endpoint.split("/locations/")[1].split("/")[0]


def read_first_datapoint(uri: str) -> Datapoint:
    if uri.startswith("gs://"):
        bucket, _, blob_path = uri[5:].partition("/")
        if not bucket or not blob_path:
            raise ValueError(f"Invalid GCS URI: {uri}")
        client = storage.Client()
        blob = client.bucket(bucket).blob(blob_path)
        if not blob.exists():
            raise FileNotFoundError(f"GCS file not found: {uri}")
        with blob.open("r") as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                return Datapoint(payload["id"], list(map(float, payload["embedding"])))
    else:
        path = Path(uri)
        if not path.exists():
            raise FileNotFoundError(f"Embedding file not found: {uri}")
        with path.open("r", encoding="utf-8") as reader:
            for raw in reader:
                raw = raw.strip()
                if not raw:
                    continue
                payload = json.loads(raw)
                return Datapoint(payload["id"], list(map(float, payload["embedding"])))
    raise RuntimeError("No datapoint could be read from the provided JSONL file.")


def run_smoke_test(datapoint: Datapoint, endpoint: str, deploy_id: str, region: str, out_path: Path) -> None:
    client = aiplatform_v1.MatchServiceClient(
        client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
    )

    request = aiplatform_v1.FindNeighborsRequest(
        index_endpoint=endpoint,
        deployed_index_id=deploy_id,
        queries=[
            aiplatform_v1.FindNeighborsRequest.Query(
                datapoint=aiplatform_v1.IndexDatapoint(feature_vector=datapoint.embedding),
                neighbor_count=3,
            )
        ],
    )

    response = ga_retry.Retry(initial=1.0, maximum=10.0, multiplier=2.0, deadline=60.0)(
        client.find_neighbors
    )(request=request)

    neighbors = []
    if response.nearest_neighbors:
        for neighbor in response.nearest_neighbors[0].neighbors:
            neighbors.append(
                {
                    "id": neighbor.datapoint.datapoint_id,
                    "distance": neighbor.distance,
                }
            )

    payload = {
        "endpoint": endpoint,
        "deployed_index_id": deploy_id,
        "query_id": datapoint.identifier,
        "neighbors": neighbors,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(neighbors, indent=2))


def main() -> None:
    args = parse_args()
    region = args.region or extract_region(args.endpoint)
    datapoint = read_first_datapoint(args.staging_json)
    run_smoke_test(datapoint, args.endpoint, args.deploy_id, region, Path(args.out))


if __name__ == "__main__":
    main()
