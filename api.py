"""Flask application exposing /enroll (建檔) 與 /detect_face (辨識) 端點。"""
from __future__ import annotations

import base64
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
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
      1) payload['embeddings']：直接使用（最佳）
      2) payload['embedding']：單一向量
      3) payload['images'] (base64 array) 或 payload['urls']：嘗試轉圖片→向量
         * 僅當 FaceRecognitionService 有提供「bytes -> embedding」能力時才會成功
          （會嘗試呼叫 embed_bytes / embedding_from_bytes / embed_image / embed）
    """
    embs: List[List[float]] = []

    # 1) 多筆向量
    if isinstance(payload.get("embeddings"), list) and payload["embeddings"]:
        for e in payload["embeddings"]:
            if isinstance(e, list) and e:
                embs.append(e)
        if embs:
            return embs

    # 2) 單筆向量
    if isinstance(payload.get("embedding"), list) and payload["embedding"]:
        return [payload["embedding"]]

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
                e = embed_fn(img)  # type: ignore[call-arg]
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

    # DB 初始化
    initialize_database(db_path, sql_path, reset=config.get("RESET_DB", False))
    # 臉辨服務
    face_service = FaceRecognitionService(threshold=recognition_threshold)

    # -------- health --------
    @app.get("/health")
    def health() -> Any:
        return jsonify({"ok": True, "time": _now_iso()})

    # -------- enroll：首次建檔 --------
    @app.post("/enroll")
    def enroll() -> Any:
        """
        請求格式（任一種即可）：
          - JSON: {"embeddings":[[...],[...]], "purchase":"Milk", "timestamp":"..."}
          - JSON: {"images":[<base64>, ...], "purchase":"Milk"}
          - JSON: {"urls":[<http(s)://...>, ...], "purchase":"Milk"}

        流程：
          1) 取得1~N筆 embedding
          2) 使用 face_service.identify_embedding(e) 取得 person_id / 信心值
          3) 若 DB 無此人 → 建檔、寫一筆消費；若已存在 → 僅寫一次消費（視作完成建檔）
        """
        start = time.perf_counter()
        payload = request.get_json(force=True) or {}

        # 蒐集 embeddings
        embeddings = _gather_embeddings_from_payload(payload, face_service)
        if not embeddings:
            return jsonify({"error": "no embeddings or images/urls to enroll"}), 400

        purchase = str(payload.get("purchase") or "Milk")
        timestamp = payload.get("timestamp") or _now_iso()

        person_id: Optional[str] = None
        confidence_max = 0.0
        # 取多張中的最好一張（或你也可改成平均/多筆存檔）
        for e in embeddings:
            pid, conf, _ = face_service.identify_embedding(e)
            if conf > confidence_max:
                confidence_max = conf
                person_id = pid

        if person_id is None:
            return jsonify({"error": "failed to generate person id"}), 500

        with connect(app.config["DB_PATH"]) as connection:
            record = get_user(connection, person_id)
            if record is None:
                create_user_record(connection, user_id=person_id, created_at=timestamp, purchase=purchase)
                record_visit(connection, user_id=person_id, visit_time=timestamp, purchase=purchase)
                message = "新用戶已建檔"
                promotion = f"首次來店，{purchase} 9折！"
                is_new = True
            else:
                # 視為已建檔，寫一筆消費
                record_visit(connection, user_id=person_id, visit_time=timestamp, purchase=purchase)
                message = "已存在用戶，新增消費紀錄"
                promotion = f"上次買{record.last_purchase}，現9折！"
                is_new = False

        duration_ms = (time.perf_counter() - start) * 1000
        resp = {
            "id": person_id,
            "confidence": confidence_max,
            "new_user": is_new,
            "message": message,
            "promotion": promotion,
            "timestamp": timestamp,
            "duration_ms": duration_ms,
            "n_embeddings": len(embeddings),
        }
        _append_json_line(Path(app.config["API_LOG_PATH"]), {"endpoint": "/enroll", "status": "ok", **resp})
        return jsonify(resp)

    # -------- detect_face：返店辨識 --------
    @app.post("/detect_face")
    def detect_face() -> Any:
        start = time.perf_counter()
        payload = request.get_json(force=True) or {}

        # 這個端點假設前端已計算好單一 embedding
        embedding = payload.get("embedding")
        if embedding is None:
            return jsonify({"error": "missing embedding"}), 400

        purchase = payload.get("purchase", "Milk")
        timestamp = payload.get("timestamp") or _now_iso()

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
        resp = {
            "id": person_id,
            "confidence": confidence,
            "new_user": is_new,
            "message": message,
            "promotion": promotion,
            "timestamp": timestamp,
            "duration_ms": duration_ms,
        }
        _append_json_line(Path(app.config["API_LOG_PATH"]), {"endpoint": "/detect_face", "status": "ok", **resp})
        return jsonify(resp)

    # -------- 管理小工具 --------
    @app.get("/admin/users")
    def list_users() -> Any:
        with connect(app.config["DB_PATH"]) as connection:
            records = [record.__dict__ for record in fetch_all_users(connection)]
        return jsonify({"users": records})

    return app


app = create_app()
__all__ = ["create_app", "app"]
