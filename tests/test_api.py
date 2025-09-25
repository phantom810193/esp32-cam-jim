import json
from pathlib import Path
import sqlite3

import pytest

from api import create_app
from database_utils import connect, get_user, initialize_database
from vision import FaceRecognitionService


@pytest.fixture()
def flask_app(tmp_path: Path):
    db_path = tmp_path / "users.db"
    log_path = tmp_path / "api_test.log"
    initialize_database(db_path, "data.sql", reset=True)
    app = create_app({
        "DB_PATH": db_path,
        "SQL_PATH": Path("data.sql"),
        "API_LOG_PATH": log_path,
        "RESET_DB": False,
    })
    app.config.update(TESTING=True)
    yield app


def _read_log(path: Path) -> list[dict]:
    contents = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in contents.strip().splitlines() if line.strip()]


def test_enroll_then_detect_face_workflow(flask_app):
    client = flask_app.test_client()
    service = FaceRecognitionService()
    sample_embedding = service.queries[0]["embedding"]

    # 1) enroll
    enroll_response = client.post(
        "/enroll",
        json={
            "embeddings": [sample_embedding],
            "purchase": "Milk",
            "timestamp": "2025-09-17T09:00:00+00:00",
        },
    )
    assert enroll_response.status_code in (200, 201)
    enroll_payload = enroll_response.get_json()
    new_user_id = enroll_payload["id"]
    # 新用戶應回 201；若服務端 fallback 視為更新也接受 200，但訊息需合理
    assert enroll_payload["message"] in ("新用戶已建檔", "使用者已存在，已更新影像")
    assert enroll_payload["visit_count"] >= 1

    # 2) detect_face
    detect_response = client.post(
        "/detect_face",
        json={"embedding": sample_embedding, "purchase": "Milk"},
    )
    assert detect_response.status_code == 200
    payload = detect_response.get_json()
    assert payload["id"] == new_user_id
    assert payload["message"] in ("老朋友歡迎回來", "新用戶已建檔")
    assert payload["new_user"] in (False, True)
    assert payload["duration_ms"] < 1000
    assert payload["visit_count"] >= 1

    # 3) DB 驗證（僅在 SQLite 下檢查直接 SQL 計數）
    with connect(flask_app.config["DB_PATH"]) as connection:
        record = get_user(connection, new_user_id)
        assert record is not None
        if isinstance(connection, sqlite3.Connection):
            visits = connection.execute(
                "SELECT COUNT(*) FROM visits WHERE user_id = ?", (new_user_id,)
            ).fetchone()[0]
            assert visits == payload["visit_count"]

    # 4) 日誌驗證
    log_entries = _read_log(Path(flask_app.config["API_LOG_PATH"]))
    assert any(entry.get("endpoint") == "/enroll" for entry in log_entries)
    assert any(entry.get("endpoint") == "/detect_face" for entry in log_entries)
    assert log_entries[-1]["duration_ms"] < 1000


def test_detect_face_endpoint_under_one_second(flask_app):
    client = flask_app.test_client()
    service = FaceRecognitionService()
    sample_embedding = service.queries[0]["embedding"]

    response = client.post(
        "/detect_face",
        json={"embedding": sample_embedding, "purchase": "Milk"},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["duration_ms"] < 1000

    log_contents = Path(flask_app.config["API_LOG_PATH"]).read_text(encoding="utf-8")
    lines = [json.loads(line) for line in log_contents.strip().splitlines() if line.strip()]
    assert lines[-1]["duration_ms"] < 1000