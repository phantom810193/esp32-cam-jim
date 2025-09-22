"""Flask application exposing the ``/detect_face`` endpoint."""
from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request

from database_utils import (
    connect,
    create_user_record,
    fetch_all_users,
    get_user,
    initialize_database,
    record_visit,
)
from vision import FaceRecognitionService


def _append_json_line(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def create_app(config: Dict[str, Any] | None = None) -> Flask:
    config = config or {}
    app = Flask(__name__)

    db_path = Path(config.get("DB_PATH", "users.db"))
    sql_path = Path(config.get("SQL_PATH", Path(__file__).resolve().parent / "data.sql"))
    api_log_path = Path(config.get("API_LOG_PATH", "api_test.log"))
    recognition_threshold = float(config.get("THRESHOLD", 0.8))

    app.config.update(
        DB_PATH=str(db_path),
        SQL_PATH=str(sql_path),
        API_LOG_PATH=str(api_log_path),
        THRESHOLD=recognition_threshold,
    )

    initialize_database(db_path, sql_path, reset=config.get("RESET_DB", False))
    face_service = FaceRecognitionService(threshold=recognition_threshold)

    @app.post("/detect_face")
    def detect_face() -> Any:
        start = time.perf_counter()
        payload = request.get_json(force=True) or {}
        embedding = payload.get("embedding")
        if embedding is None:
            return jsonify({"error": "missing embedding"}), 400
        purchase = payload.get("purchase", "Milk")
        timestamp = payload.get("timestamp") or datetime.now(UTC).isoformat()

        person_id, confidence, is_new = face_service.identify_embedding(embedding)
        with connect(app.config["DB_PATH"]) as connection:
            record = get_user(connection, person_id)
            if record is None:
                create_user_record(connection, user_id=person_id, created_at=timestamp, purchase=purchase)
                message = "新用戶已建檔"
                promotion = f"首次來店，{purchase} 9折！"
                is_new = True
            else:
                record_visit(connection, user_id=person_id, visit_time=timestamp, purchase=purchase)
                message = "老朋友歡迎回來"
                promotion = f"上次買{record.last_purchase}，現9折！"

        duration_ms = (time.perf_counter() - start) * 1000
        response_payload = {
            "id": person_id,
            "confidence": confidence,
            "new_user": is_new,
            "message": message,
            "promotion": promotion,
            "timestamp": timestamp,
            "duration_ms": duration_ms,
        }
        log_entry = {"endpoint": "/detect_face", "status": "ok", **response_payload}
        _append_json_line(Path(app.config["API_LOG_PATH"]), log_entry)
        return jsonify(response_payload)

    @app.get("/admin/users")
    def list_users() -> Any:
        with connect(app.config["DB_PATH"]) as connection:
            records = [record.__dict__ for record in fetch_all_users(connection)]
        return jsonify({"users": records})

    return app


app = create_app()


__all__ = ["create_app", "app"]
