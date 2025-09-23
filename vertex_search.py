# vertex_search.py
from __future__ import annotations
import os
from typing import List, Tuple
from google.cloud import aiplatform_v1 as gapic

PROJECT = os.getenv("GCP_PROJECT", "esp32cam-472912")
REGION  = os.getenv("VERTEX_REGION", "asia-east1")
INDEX_RESOURCE = os.getenv("VERTEX_INDEX")  # projects/../indexes/123
ENDPOINT_RESOURCE = os.getenv("VERTEX_INDEX_ENDPOINT")  # projects/../indexEndpoints/456
DEPLOYED_ID = os.getenv("VERTEX_DEPLOYED_INDEX_ID", "faces-deployed")

_api = f"{REGION}-aiplatform.googleapis.com"
_index = gapic.IndexServiceClient(client_options={"api_endpoint": _api})
_match = gapic.MatchServiceClient(client_options={"api_endpoint": _api})

def upsert(user_id: str, vectors: List[List[float]]) -> None:
    """把同一人的多個向量寫入索引（同一 datapoint id 可放多個向量 => 建議平均化後只放1個亦可）"""
    # 這裡採用「只放一個代表向量」：多張圖先在上層平均後再丟進來
    dp = gapic.IndexDatapoint(datapoint_id=user_id, feature_vector=vectors[0])
    _index.upsert_datapoints(index=INDEX_RESOURCE, datapoints=[dp])

def query(vec: List[float], k: int = 5) -> List[Tuple[str, float]]:
    """回傳 [(id, score)]，score 越大越相似（Cosine 相似度）"""
    q = gapic.FindNeighborsRequest.Query(
        datapoint=gapic.IndexDatapoint(feature_vector=vec),
        neighbor_count=k,
    )
    resp = _match.find_neighbors(
        index_endpoint=ENDPOINT_RESOURCE,
        deployed_index_id=DEPLOYED_ID,
        queries=[q],
    )
    out = []
    for n in resp.nearest_neighbors[0].neighbors:
        out.append((n.datapoint.datapoint_id, n.distance))  # 對 Cosine：這裡是「相似度」，越大越像
    return out