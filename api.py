# api.py
from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, request

# ArcFace(512-d) server-side embedding + Vertex AI Vector Search
from embedding_arcface import embed as arcface_embed
from vertex_search import upsert as vs_upsert, query as vs_query

THRESHOLD: float = float(os.getenv("THRESHOLD", "0.80"))
API_LOG: Path = Path(os.getenv("API_LOG_PATH", "api_test.log"))
ARCFACE_DIM: int = 512


# --------------- small helpers ---------------
def _append_json_line(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as h:
        h.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _avg(vectors: List[List[float]]) -> List[float]:
    import numpy as np
    arr = np.array(vectors, dtype="float32")
    v = arr.mean(axis=0)
    n = float((v**2).sum() ** 0.5) + 1e-9
    return (v / n).astype("float32").tolist()


def _ensure_dim(vec: List[float], dim: int = ARCFACE_DIM) -> Tuple[bool, str]:
    if not isinstance(vec, (list, tuple)):
        return False, "embedding must be a list"
    if len(vec) != dim:
        return False, f"embedding length must be {dim}, got {len(vec)}"
    return True, ""


def _to_arcface_vector_from_image_payload(img_payload: Any) -> List[float]:
    """
    Accepts:
      - data URI string (data:image/...;base64,...)
      - raw base64 string (…==)
      - bytes (already decoded)
    Returns ArcFace(512) as list[float]
    """
    vec = arcface_embed(img_payload)
    # arcface_embed 會回 numpy.ndarray；轉為 list 並檢核維度
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    ok, msg = _ensure_dim(vec, ARCFACE_DIM)
    if not ok:
        raise ValueError(msg)
    return [float(x) for x in vec]


# --------------- Flask app ---------------
def create_app(config: Dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    if config:
        app.config.update(config)

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "time": datetime.now(UTC).isoformat()})

    @app.post("/enroll")
    def enroll():
        """
        首次：images[]=dataURI（建議 3~5 張，多角度）+ purchase(optional)
        - 忽略任何 client 端的 embedding，統一由伺服器 ArcFace(512) 計算
        - 代表向量採 L2-normalized average
        """
        t0 = time.perf_counter()
        try:
            p = request.get_json(force=True) or {}
            images = p.get("images") or []
            if not images:
                return jsonify({"error": "no images"}), 400

            embs: List[List[float]] = []
            for img in images:
                e = _to_arcface_vector_from_image_payload(img)
                embs.append(e)

            user_id = p.get("id") or f"ID-{int(time.time())}"
            rep = _avg(embs)  # 代表向量
            vs_upsert(user_id, [rep])  # Vertex AI Vector Search

            payload = {
                "id": user_id,
                "message": "新用戶已建檔",
                "promotion": f"首次來店，{p.get('purchase','Milk')} 9折！",
                "embeddings": len(embs),
                "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }
            _append_json_line(API_LOG, {"endpoint": "/enroll", "status": "ok", **payload})
            return jsonify(payload)

        except Exception as e:
            err = {
                "endpoint": "/enroll",
                "status": "error",
                "error": str(e),
                "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }
            _append_json_line(API_LOG, err)
            return jsonify({"error": str(e)}), 400

    @app.post("/detect_face")
    def detect_face():
        """
        返店：
          - 建議送 image（單張 dataURI）→ 伺服器端 ArcFace(512) 計算
          - 亦可送 embedding（list[float]，長度必須 512）
        回傳：
          id / confidence / new_user / message / promotion
        """
        t0 = time.perf_counter()
        try:
            p = request.get_json(force=True) or {}

            if "embedding" in p:
                vec = p["embedding"]
                ok, msg = _ensure_dim(vec, ARCFACE_DIM)
                if not ok:
                    return jsonify({"error": msg}), 400
                vec = [float(x) for x in vec]
            else:
                # 單張影像（或 images 第一張）
                img = p.get("image") or (p.get("images") or [None])[0]
                if img is None:
                    return jsonify({"error": "no embedding or image"}), 400
                vec = _to_arcface_vector_from_image_payload(img)

            # 近鄰查詢
            neighbors = vs_query(vec, k=3)  # → List[Tuple[id, score]]
            if neighbors:
                best_id, best_score = neighbors[0]
                best_score = float(best_score)
            else:
                best_id, best_score = "", 0.0

            is_new = (not neighbors) or (best_score < THRESHOLD)
            if is_new:
                uid = p.get("id") or f"ID-{int(time.time())}"
                message = "新用戶已建檔"
                promo = f"首次來店，{p.get('purchase','Milk')} 9折！"
                # 直接把當前向量上到向量庫，後續就能匹配回來
                vs_upsert(uid, [vec])
            else:
                uid = str(best_id)
                message = "老朋友歡迎回來"
                promo = f"上次買{p.get('purchase','Milk')}，現9折！"

            resp = {
                "id": uid,
                "confidence": round(best_score, 4),
                "new_user": is_new,
                "message": message,
                "promotion": promo,
                "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }
            _append_json_line(API_LOG, {"endpoint": "/detect_face", "status": "ok", **resp})
            return jsonify(resp)

        except Exception as e:
            err = {
                "endpoint": "/detect_face",
                "status": "error",
                "error": str(e),
                "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }
            _append_json_line(API_LOG, err)
            # 以 400 告知可修復的輸入錯誤；非輸入問題可改 500
            return jsonify({"error": str(e)}), 400

    return app


app = create_app()
__all__ = ["create_app", "app"]