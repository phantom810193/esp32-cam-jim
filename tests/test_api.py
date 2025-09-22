import json
from pathlib import Path

import pytest

from api import create_app
from database_utils import initialize_database
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
    assert payload["id"].startswith("ID-")
    assert payload["duration_ms"] < 1000

    log_contents = Path(flask_app.config["API_LOG_PATH"]).read_text(encoding="utf-8")
    lines = [json.loads(line) for line in log_contents.strip().splitlines() if line.strip()]
    assert lines[-1]["duration_ms"] < 1000
