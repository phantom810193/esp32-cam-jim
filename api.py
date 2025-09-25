"""Flask application exposing enrollment and recognition endpoints."""
from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from flask import Flask, jsonify, request

from database_utils import (
    connect,
    create_user_record,
    fetch_all_users,
    fetch_recent_visits,
    get_user,
    initialize_database,
    record_visit,
)
from vision import FaceRecognitionService

# ---- Optional ArcFace image embedding (server-side) ----
_ARCFACE_AVAILABLE = False
_ARCFACE_DIM = 512
try:
    from embedding_arcface import embed as arcface_embed  # type: ignore
    _ARCFACE_AVAILABLE = True
except Exception:
    _ARCFACE_AVAILABLE = False


def _append_json_line(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _avg(vectors: List[List[float]]) -> List[float]:
    import numpy as np

    arr = np.asarray(vectors, dtype="float32")
    v = arr.mean(axis=0)
    n = float((v**2).sum() ** 0.5) + 1e-9
    return (v / n).astype("float32").tolist()


def _to_arcface_vector_from_image_payload(img_payload: Any) -> List[float]:
    if not _ARCFACE_AVAILABLE:
        raise RuntimeError("arcface embedding not available")
    vec = arcface_embed(img_payload)
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    if not isinstance(vec, (list, tuple)) or len(vec) != _ARCFACE_DIM:
        raise ValueError(f"embedding length must be {_ARCFACE_DIM}")
    return [float(x) for x in vec]


def _resolve_embeddings(payload: Dict[str, Any]) -> List[Iterable[float]]:
    embeddings = payload.get("embeddings")
    if embeddings is None:
        single = payload.get("embedding")
        if single is not None:
            embeddings = [single]
        else:
            images = payload.get("images")
            if images:
                # Compute embeddings on server if available
                embs = [_to_arcface_vector_from_image_payload(img) for img in images]
                embeddings = embs
    if not embeddings:
        raise ValueError("missing embeddings")
    return list(embeddings)


def _resolve_single_embedding(payload: Dict[str, Any]) -> List[float]:
    """For /detect_face: accept 'embedding' or single 'image'/'images'."""
    if "embedding" in payload:
        return list(payload["embedding"])
    img = payload.get("image")
    if img is None:
        imgs = payload.get("images") or []
        img = imgs[0] if imgs else None
    if img is not None:
        return _to_arcface_vector_from_image_payload(img)
    raise ValueError("missing embedding")


def create_app(config: Dict[str, Any] | None = None) -> Flask:
    config = config or {}
    app = Flask(__name__)

    db_path = Path(config.get("DB_PATH", "users.db"))
    sql_path = Path(config.get("SQL_PATH", Path(__file__).resolve().parent / "data.sql"))
    api_log_path = Path(config.get("API_LOG_PATH", "api_test.log"))
    recognition_threshold = float(config.get("THRESHOLD", os.getenv("THRESHOLD", 0.8)))

    app.config.update(
        DB_PATH=str(db_path),
        SQL_PATH=str(sql_path),
        API_LOG_PATH=str(api_log_path),
        THRESHOLD=recognition_threshold,
    )

    initialize_database(db_path, sql_path, reset=config.get("RESET_DB", False))
    face_service = FaceRecognitionService(threshold=recognition_threshold)

    def _log_and_response(payload: Dict[str, Any], *, endpoint: str, status_code: int = 200):
        log_entry = {"endpoint": endpoint, "status": "ok", **payload}
        _append_json_line(Path(app.config["API_LOG_PATH"]), log_entry)
        return jsonify(payload), status_code

    @app.get("/health")
    def health() -> Any:
        return jsonify({"ok": True, "time": datetime.now(UTC).isoformat()})

    @app.post("/enroll")
    def enroll() -> Any:
        start = time.perf_counter()
        try:
            payload = request.get_json(force=True) or {}
            embeddings = _resolve_embeddings(payload)
        except (TypeError, ValueError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 400

        purchase = payload.get("purchase", "Milk")
        timestamp = payload.get("timestamp") or datetime.now(UTC).isoformat()
        spend = float(payload.get("spend", 100.0))
        requested_id = payload.get("id")

        enrollment = face_service.enroll(embeddings, user_id=requested_id)
        resolved_id = enrollment["id"]
        with connect(app.config["DB_PATH"]) as connection:
            record = get_user(connection, resolved_id)
            if record is None:
                create_user_record(
                    connection,
                    user_id=resolved_id,
                    created_at=timestamp,
                    purchase=purchase,
                    spend=spend,
                    source="enroll",
                )
                message = "新用戶已建檔"
            else:
                record_visit(
                    connection,
                    user_id=resolved_id,
                    visit_time=timestamp,
                    purchase=purchase,
                    spend=spend,
                    source="enroll",
                )
                message = "使用者已存在，已更新影像"
            recent_visits = fetch_recent_visits(connection, user_id=resolved_id, limit=5)

        duration_ms = (time.perf_counter() - start) * 1000
        response_payload = {
            "id": resolved_id,
            "message": message,
            "timestamp": timestamp,
            "embeddings_registered": len(enrollment["embeddings"]),
            "visit_count": len(recent_visits),
            "duration_ms": duration_ms,
            "visits": [visit.__dict__ for visit in recent_visits],
        }
        status_code = 201 if message == "新用戶已建檔" else 200
        return _log_and_response(response_payload, endpoint="/enroll", status_code=status_code)

    @app.post("/detect_face")
    def detect_face() -> Any:
        start = time.perf_counter()
        try:
            payload = request.get_json(force=True) or {}
            embedding = _resolve_single_embedding(payload)
        except (TypeError, ValueError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 400

        purchase = payload.get("purchase", "Milk")
        timestamp = payload.get("timestamp") or datetime.now(UTC).isoformat()
        spend = float(payload.get("spend", 100.0))

        person_id, confidence, is_new = face_service.identify_embedding(embedding)
        with connect(app.config["DB_PATH"]) as connection:
            record = get_user(connection, person_id)
            if record is None:
                create_user_record(
                    connection,
                    user_id=person_id,
                    created_at=timestamp,
                    purchase=purchase,
                    spend=spend,
                    source="detect_face",
                )
                message = "新用戶已建檔"
                promotion = f"首次來店，{purchase} 9折！"
                is_new = True
            else:
                record_visit(
                    connection,
                    user_id=person_id,
                    visit_time=timestamp,
                    purchase=purchase,
                    spend=spend,
                    source="detect_face",
                )
                message = "老朋友歡迎回來"
                promotion = f"上次買{record.last_purchase}，現9折！"
            recent_visits = fetch_recent_visits(connection, user_id=person_id, limit=5)

        duration_ms = (time.perf_counter() - start) * 1000
        response_payload = {
            "id": person_id,
            "confidence": confidence,
            "new_user": is_new,
            "message": message,
            "promotion": promotion,
            "timestamp": timestamp,
            "duration_ms": duration_ms,
            "visit_count": len(recent_visits),
            "visits": [visit.__dict__ for visit in recent_visits],
        }
        return _log_and_response(response_payload, endpoint="/detect_face")

    @app.get("/admin/users")
    def list_users() -> Any:
        with connect(app.config["DB_PATH"]) as connection:
            records = [record.__dict__ for record in fetch_all_users(connection)]
        return jsonify({"users": records})

    return app


app = create_app()

__all__ = ["create_app", "app"]