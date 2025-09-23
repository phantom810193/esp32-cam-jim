"""Flask application exposing '/', '/healthz', and '/detect_face'."""
from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


def _extract_embedding_or_identify(
    face_service: FaceRecognitionService,
    *,
    json_payload: Dict[str, Any] | None,
    file_bytes: bytes | None,
) -> Tuple[Optional[str], Optional[float], Optional[bool], Optional[list]]:
    """
    嘗試從輸入取 embedding，或直接呼叫服務做影像辨識。
    回傳：(person_id, confidence, is_new, embedding)
      - 若可直接辨識（例如有 identify_image），前 3 個值有內容、embedding 為 None
      - 若只拿得到 embedding，前 3 個為 (None, None, None)，embedding 有內容
      - 兩者都失敗則都為 None
    """
    # 1) JSON 直接帶 embedding（原本流程）
    if json_payload:
        emb = json_payload.get("embedding")
        if emb is not None:
            return None, None, None, emb

    # 2) multipart 上傳圖片且 vision/face 服務支援
    if file_bytes:
        # 有些實作可能提供 "identify_image"（直接回識別）
        if hasattr(face_service, "identify_image"):
            try:
                person_id, confidence, is_new = face_service.identify_image(file_bytes)  # type: ignore[attr-defined]
                return str(person_id), float(confidence), bool(is_new), None
            except Exception:
                pass

        # 或者有 "embedding_from_image" / "embed_image" 等方法
        for method_name in ("embedding_from_image", "embed_image", "image_to_embedding"):
            if hasattr(face_service, method_name):
                try:
                    emb = getattr(face_service, method_name)(file_bytes)  # type: ignore[misc]
                    return None, None, None, emb
                except Exception:
                    pass

    return None, None, None, None


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

    # 初始化 SQLite（若已存在則不會覆蓋）
    initialize_database(db_path, sql_path, reset=config.get("RESET_DB", False))
    face_service = FaceRecognitionService(threshold=recognition_threshold)

    # --- 健康檢查與根路徑（Cloud Run/CI 用） ---
    @app.get("/")
    def root() -> Any:
        return "ok", 200

    @app.get("/healthz")
    def healthz() -> Any:
        return "ok", 200

    # --- 主要偵測端點 ---
    @app.post("/detect_face")
    def detect_face() -> Any:
        start = time.perf_counter()

        # 同時支援 JSON 與 multipart（file=影像）
        json_payload: Dict[str, Any] = {}
        if request.is_json:
            try:
                json_payload = request.get_json(force=True) or {}
            except Exception:
                json_payload = {}

        file_bytes = None
        if "file" in request.files:
            file_storage = request.files["file"]
            file_bytes = file_storage.read() if file_storage else None

        # 抽取 embedding 或直接辨識影像
        person_id, confidence, is_new, embedding = _extract_embedding_or_identify(
            face_service, json_payload=json_payload, file_bytes=file_bytes
        )

        # 購買項目/時間戳
        purchase = (json_payload.get("purchase") if json_payload else None) or request.form.get("purchase") or "Milk"
        timestamp = (
            (json_payload.get("timestamp") if json_payload else None)
            or request.form.get("timestamp")
            or datetime.now(UTC).isoformat()
        )

        # 若已可直接辨識（例如 identify_image），就用結果；否則需要 embedding
        if person_id is None:
            if embedding is None:
                return jsonify({"error": "missing embedding or unsupported image-to-embedding"}), 400
            # 原流程：用 embedding 做辨識
            person_id, confidence, is_new = face_service.identify_embedding(embedding)

        # 寫入/查詢 SQLite 並產出訊息
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
                # 若有歷史購買紀錄欄位則使用，否則改用本次
                last_purchase = getattr(record, "last_purchase", purchase)
                promotion = f"上次買{last_purchase}，現9折！"

        duration_ms = (time.perf_counter() - start) * 1000.0
        response_payload = {
            "id": person_id,
            "confidence": confidence,
            "new_user": bool(is_new),
            "message": message,
            "promotion": promotion,
            "timestamp": timestamp,
            "duration_ms": duration_ms,
        }
        log_entry = {"endpoint": "/detect_face", "status": "ok", **response_payload}
        _append_json_line(Path(app.config["API_LOG_PATH"]), log_entry)
        return jsonify(response_payload), 200

    # --- 管理查詢 ---
    @app.get("/admin/users")
    def list_users() -> Any:
        with connect(app.config["DB_PATH"]) as connection:
            records = [record.__dict__ for record in fetch_all_users(connection)]
        return jsonify({"users": records})

    return app


app = create_app()

# 讓本地/非 buildpack 環境也能直接執行（Cloud Run 仍會以 gunicorn 啟動）
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
