"""End-to-end orchestration script for the demo pipeline."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable

from admin_dashboard import build_dashboard
from api import create_app
from camera import simulate_camera_capture
from database_utils import connect, fetch_all_users, initialize_database
from display import render_messages
from promo_ui import render_promotions
from stability import run_stability_assessment
from vision import FaceRecognitionService


def run_pipeline(
    *,
    output_dir: Path | str = Path("."),
    purchase: str = "Milk",
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_sql_path = Path(__file__).resolve().parent / "data.sql"
    db_path = output_dir / "users.db"
    initialize_database(db_path, data_sql_path, reset=True)

    start = time.perf_counter()
    camera_metrics = simulate_camera_capture(
        duration_seconds=5.0,
        target_fps=12.0,
        log_path=output_dir / "cam.log",
    )

    stability_report = run_stability_assessment(log_path=output_dir / "cam_stable.log")

    service = FaceRecognitionService()
    recognition_metrics = service.evaluate_queries(log_path=output_dir / "id_test.log")

    query = service.queries[0]
    detected_id, confidence, _ = service.identify_embedding(query["embedding"])

    app = create_app(
        {
            "DB_PATH": db_path,
            "SQL_PATH": data_sql_path,
            "API_LOG_PATH": output_dir / "api_test.log",
            "RESET_DB": False,
        }
    )
    app.config.update(TESTING=True)

    with app.test_client() as client:
        api_response = client.post(
            "/detect_face",
            json={"embedding": query["embedding"], "purchase": purchase},
        )
        api_payload = api_response.get_json()

    with connect(db_path) as connection:
        users = fetch_all_users(connection)

    messages = _build_messages(
        detected_id,
        confidence,
        purchase,
        users,
        api_message=api_payload["message"],
        promotion=api_payload["promotion"],
    )
    render_messages(messages, log_path=output_dir / "text_test.log", enable_print=False)

    promo_path = render_promotions(
        [api_payload["promotion"]], output_path=output_dir / "promo_display.log"
    )
    promo_metadata = json.loads(
        promo_path.with_suffix(promo_path.suffix + ".json").read_text(encoding="utf-8")
    )

    dashboard_path = build_dashboard(
        db_path=db_path, output_path=output_dir / "admin_dashboard.html"
    )

    duration_seconds = time.perf_counter() - start
    summary = {
        "camera": camera_metrics.to_dict(),
        "recognition": recognition_metrics,
        "stability": stability_report.to_dict(),
        "api": api_payload,
        "detected_id": detected_id,
        "confidence": confidence,
        "users": [user.__dict__ for user in users],
        "messages": messages,
        "promotion": promo_metadata,
        "dashboard_path": str(dashboard_path),
        "logs": {
            "camera": str(output_dir / "cam.log"),
            "recognition": str(output_dir / "id_test.log"),
            "text": str(output_dir / "text_test.log"),
            "stability": str(output_dir / "cam_stable.log"),
            "api": str(output_dir / "api_test.log"),
            "promotion": str(promo_path),
            "promotion_meta": str(promo_path.with_suffix(promo_path.suffix + ".json")),
            "dashboard": str(dashboard_path),
            "summary": str(output_dir / "e2e.log"),
        },
        "duration_seconds": duration_seconds,
        "status": "ok" if duration_seconds < 30 else "timeout",
    }

    e2e_path = output_dir / "e2e.log"
    e2e_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _build_messages(
    detected_id: str,
    confidence: float,
    purchase: str,
    users: Iterable[object],
    *,
    api_message: str,
    promotion: str,
) -> list[str]:
    history_messages = [
        f"歷史消費 - {user.id}: {user.last_purchase}" for user in users
    ]
    messages = [
        f"辨識ID: {detected_id}",
        f"相似度: {confidence:.2f}",
        f"系統訊息: {api_message}",
        f"推播: {promotion}",
    ]
    if history_messages:
        messages.append(history_messages[0])
    else:
        messages.append(f"建議商品: {purchase}")
    if len(messages) < 5 and len(history_messages) > 1:
        messages.append(history_messages[1])
    elif len(messages) < 5:
        messages.append(f"建議商品: {purchase}")
    return messages[:5]


if __name__ == "__main__":
    run_pipeline()
