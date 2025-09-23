"""Flask application exposing /enroll (建檔) 與 /detect_face (辨識) 端點。"""
from __future__ import annotations

import base64
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
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


# ---------- helpers ----------
def _append_json_line(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _b64_to_bytes(s: str) -> bytes:
    """支援 dataURL 或純 base64。"""
    if s.startswith("data:image"):
        s = s.split(",", 1)[1]
    return base64.b64decode(s)


def _fetch_bytes(url: str, timeout: float = 10.0) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def _gather_embeddings_from_payload(
    payload: Dict[str, Any], face_service: FaceRecognitionService
) -> List[List[float]]:
    """
    從請求內容組出 embeddings：
      1) payload['embeddings']：直接使用
      2) payload['embedding']：單一向量
      3) payload['images'] (base64 array) 或 payload['urls']：嘗試轉圖片→向量
         * 僅當 FaceRecognitionService 有提供 bytes→embedding 能力時才會成功
    """
    embs: List[List[float]] = []

    # 1) 多筆向量
    if isinstance(payload.get("embeddings"), list) and payload["embeddings"]:
        for e in payload["embeddings"]:
            if isinstance(e, list) and e:
                embs.append([float(x) for x in e])
        if embs:
            return embs

    # 2) 單筆向量
    if isinstance(payload.get("embedding"), list) and payload["embedding"]:
        return [[float(x) for x in payload["embedding"]]]

    # 3) 圖片（base64 或 URL）
    images: List[bytes] = []
    if isinstance(payload.get("images"), list):
        for s in payload["images"]:
            try:
                images.append(_b64_to_bytes(str(s)))
            except Exception:
                pass
    if isinstance(payload.get("urls"), list):
        for u in payload["urls"]:
            try:
                images.append(_fetch_bytes(str(u)))
            except Exception:
                pass

    if images:
        # 嘗試從 face_service 找可用的方法把 bytes -> embedding
        method_names = ["embed_bytes", "embedding_from_bytes", "embed_image", "embed"]
        embed_fn = None
        for name in method_names:
            if hasattr(face_service, name):
                embed_fn = getattr(face_service, name)
                break
        if embed_fn is None:
            # 伺服器端沒有提供影像轉向量的方法；讓前端先算好 embeddings 再呼叫
            return []

        for img in images:
            try:
                e = embed_fn(img)  # type: ignore[misc]
                if isinstance(e, (list, tuple)) and e and isinstance(e[0], (float, int)):
                    embs.append([float(x) for x in e])
            except Exception:
                # 單張失敗不致命，繼續
                pass

    return embs


# ---------- app factory ----------
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

    # DB 初始化 + 臉辨服務
    initialize_database(db_path, sql_path, reset=config.get("RESET_DB", False))
    face_service = FaceRecognitionService(threshold=recognition_threshold)

    def _log_and_response(payload: Dict[str, Any], *, endpoint: str, status_code: int = 200):
        log_entry = {"endpoint": endpoint, "status": "ok", **payload}
        _append_json_line(Path(app.config["API_LOG_PATH"]), log_entry)
        return jsonify(payload), status_code

    # -------- health --------
    @app.get("/health")
    def health() -> Any:
        return jsonify({"ok": True, "time": _now_iso()})

    # -------- enroll：首次建檔（支援 embeddings / images / urls）--------
    @app.post("/enroll")
    def enroll() -> Any:
        start = time.perf_counter()
        payload = request.get_json(force=True) or {}

        try:
            embeddings = _gather_embeddings_from_payload(payload, face_service)
        except Exception as exc:
            return jsonify({"error": f"invalid payload: {exc}"}), 400

        if not embeddings:
            return jsonify({"error": "no embeddings or images/urls to enroll"}), 400

        purchase = str(payload.get("purchase") or "Milk")
        timestamp = payload.get("timestamp") or _now_iso()
        spend = float(payload.get("spend", 100.0))
        requested_id = payload.get("id")

        # 先計算最高信心（給回應用）
        best_conf = 0.0
        for e in embeddings:
            _, conf, _ = face_service.identify_embedding(e)
            best_conf = max(best_conf, float(conf))

        # 若服務支援 enroll，優先使用；否則 fallback 取最高信心者的 id
        resolved_id: Optional[str] = None
        n_registered = len(embeddings)
        if hasattr(face_service, "enroll"):
            try:
                enrollment = face_service.enroll(embeddings, user_id=requested_id)  # type: ignore[misc]
                resolved_id = str(enrollment.get("id"))
                n_registered = int(enrollment.get("embeddings_count", len(enrollment.get("embeddings", embeddings))))
            except Exception:
                resolved_id = None

        if not resolved_id:
            best_id = None
            best = -1.0
            for e in embeddings:
                pid, conf, _ = face_service.identify_embedding(e)
                if conf > best:
                    best = float(conf)
                    best_id = pid
            resolved_id = str(best_id) if best_id else None

        if not resolved_id:
            return jsonify({"error": "failed to resolve user id"}), 500

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
                is_new = True
                base_purchase = purchase
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
                is_new = False
                base_purchase = record.last_purchase or purchase

            recent_visits = fetch_recent_visits(connection, user_id=resolved_id, limit=5)

        promotion = f"上次買{base_purchase}，現9折！" if not is_new else f"首次來店，{purchase} 9折！"
        duration_ms = (time.perf_counter() - start) * 1000.0

        resp = {
            "id": resolved_id,
            "confidence": best_conf,
            "new_user": is_new,
            "message": message,
            "promotion": promotion,
            "timestamp": timestamp,
            "duration_ms": duration_ms,
            "embeddings_registered": n_registered,
            "visit_count": len(recent_visits),
            "visits": [v.__dict__ for v in recent_visits],
        }
        return _log_and_response(resp, endpoint="/enroll", status_code=201 if is_new else 200)

    # -------- detect_face：返店辨識（單一 embedding）--------
    @app.post("/detect_face")
    def detect_face() -> Any:
        start = time.perf_counter()
        payload = request.get_json(force=True) or {}

        embedding = payload.get("embedding")
        if embedding is None:
            return jsonify({"error": "missing embedding"}), 400

        purchase = payload.get("purchase", "Milk")
        timestamp = payload.get("timestamp") or _now_iso()
        spend = float(payload.get("spend", 100.0))

        person_id, confidence, _ = face_service.identify_embedding(embedding)
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
                is_new = False

            recent_visits = fetch_recent_visits(connection, user_id=person_id, limit=5)

        duration_ms = (time.perf_counter() - start) * 1000.0
        resp = {
            "id": person_id,
            "confidence": float(confidence),
            "new_user": is_new,
            "message": message,
            "promotion": promotion,
            "timestamp": timestamp,
            "duration_ms": duration_ms,
            "visit_count": len(recent_visits),
            "visits": [v.__dict__ for v in recent_visits],
        }
        return _log_and_response(resp, endpoint="/detect_face")

    # -------- 管理小工具 --------
    @app.get("/admin/users")
    def list_users() -> Any:
        with connect(app.config["DB_PATH"]) as connection:
            records = [record.__dict__ for record in fetch_all_users(connection)]
        return jsonify({"users": records})

    return app


app = create_app()

__all__ = ["create_app", "app"]