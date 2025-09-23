# api.py
from __future__ import annotations
import json, time, os, base64
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, request

from embedding_arcface import embed
from vertex_search import upsert as vs_upsert, query as vs_query

THRESHOLD = float(os.getenv("THRESHOLD", "0.80"))
API_LOG   = Path(os.getenv("API_LOG_PATH", "api_test.log"))

def _append_json_line(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as h:
        h.write(json.dumps(payload, ensure_ascii=False) + "\n")

def _avg(vectors: List[List[float]]) -> List[float]:
    import numpy as np
    arr = np.array(vectors, dtype="float32")
    v = arr.mean(axis=0)
    n = (v**2).sum()**0.5 + 1e-9
    return (v / n).astype("float32").tolist()

def create_app(config: Dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "time": datetime.now(UTC).isoformat()})

    @app.post("/enroll")
    def enroll():
        """首次：images[]=dataURI（3~5張）+ purchase(optional)"""
        start = time.perf_counter()
        p = request.get_json(force=True) or {}
        images = p.get("images") or []
        if not images:
            return jsonify({"error": "no images"}), 400

        embs: List[List[float]] = []
        for img in images:
            e = embed(img).tolist()
            embs.append(e)

        # 產生 user_id（可以改成你的人臉 ID 規則）
        user_id = p.get("id") or f"ID-{int(time.time())}"
        rep = _avg(embs)
        vs_upsert(user_id, [rep])

        payload = {
            "id": user_id,
            "message": "新用戶已建檔",
            "promotion": f"首次來店，{p.get('purchase','Milk')} 9折！",
            "embeddings": len(embs),
            "duration_ms": (time.perf_counter() - start) * 1000.0,
        }
        _append_json_line(API_LOG, {"endpoint": "/enroll", "status": "ok", **payload})
        return jsonify(payload)

    @app.post("/detect_face")
    def detect_face():
        """返店：image(單張 dataURI) 或 embedding（512長度）"""
        start = time.perf_counter()
        p = request.get_json(force=True) or {}

        if "embedding" in p:
            vec = p["embedding"]
        else:
            img = p.get("image") or (p.get("images") or [None])[0]
            if img is None:
                return jsonify({"error": "no embedding or image"}), 400
            vec = embed(img).tolist()

        # 近鄰查詢
        nn = vs_query(vec, k=3)
        best_id, best_score = (nn[0] if nn else ("", 0.0))

        is_new = not nn or (best_score < THRESHOLD)
        if is_new:
            uid = f"ID-{int(time.time())}"
            message = "新用戶已建檔"
            promo   = f"首次來店，{p.get('purchase','Milk')} 9折！"
            vs_upsert(uid, [vec])  # 建檔 + 放代表向量
        else:
            uid = best_id
            message = "老朋友歡迎回來"
            promo   = f"上次買{p.get('purchase','Milk')}，現9折！"  # 可改為查歷史紀錄後動態組文案

        resp = {
            "id": uid,
            "confidence": round(float(best_score), 4),
            "new_user": is_new,
            "message": message,
            "promotion": promo,
            "duration_ms": (time.perf_counter() - start) * 1000.0,
        }
        _append_json_line(API_LOG, {"endpoint": "/detect_face", "status": "ok", **resp})
        return jsonify(resp)

    return app

app = create_app()
__all__ = ["create_app", "app"]