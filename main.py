"""End-to-end orchestration script for the demo pipeline."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable

from admin_dashboard import build_dashboard
from api import create_app
from camera import simulate_camera_capture
from database_utils import connect, fetch_all_users, fetch_recent_visits, initialize_database
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

    # 初始化資料庫
    data_sql_path = Path(__file__).resolve().parent / "data.sql"
    db_path = output_dir / "users.db"
    initialize_database(db_path, data_sql_path, reset=True)

    start = time.perf_counter()

    # 相機模擬與效能
    camera_metrics = simulate_camera_capture(
        duration_seconds=5.0,
        target_fps=12.0,
        log_path=output_dir / "cam.log",
    )
    stability_report = run_stability_assessment(log_path=output_dir / "cam_stable.log")

    # 人臉辨識測試
    service = FaceRecognitionService()
    recognition_metrics = service.evaluate_queries(log_path=output_dir / "id_test.log")

    # 啟動 Flask 測試 client，測試 /enroll 與 /detect_face
    app = create_app(
        {
            "DB_PATH": db_path,
            "SQL_PATH": data_sql_path,
            "API_LOG_PATH": output_dir / "api_test.log",
            "RESET_DB": False,
        }
    )
    app.config.update(TESTING=True)

    enrollment_payload: Dict[str, object] = {}
    api_payload: Dict[str, object] = {}

    with app.test_client() as client:
        # 取前兩筆 query 的 embedding 做 enroll，若不足則取第一筆
        enrollment_embeddings = [q["embedding"] for q in service.queries[:2]] or [
            service.queries[0]["embedding"]
        ]

        enroll_response = client.post(
            "/enroll",
            json={
                "embeddings": enrollment_embeddings,
                "purchase": purchase,
                "timestamp": "2025-09-17T09:00:00+00:00",
            },
        )
        enrollment_payload = enroll_response.get_json() or {}

        detect_response = client.post(
            "/detect_face",
            json={"embedding": service.queries[0]["embedding"], "purchase": purchase},
        )
        api_payload = detect_response.get_json() or {}

    # 讀取資料庫資料
    with connect(db_path) as connection:
        users = fetch_all_users(connection)
        recent_visits = fetch_recent_visits(connection, user_id=api_payload.get("id"), limit=5)

    # UI 訊息、促銷與儀表板
    detected_id = api_payload.get("id", "未知")
    confidence = float(api_payload.get("confidence", 0.0))
    messages = _build_messages(
        enrollment_message=enrollment_payload.get("message", ""),
        detection_message=api_payload.get("message", ""),
        detected_id=detected_id,
        confidence=confidence,
        promotion=api_payload.get("promotion", ""),
        purchase=purchase,
        users=users,
    )
    render_messages(messages, log_path=output_dir / "text_test.log", enable_print=False)

    promo_path = render_promotions(
        [api_payload.get("promotion", "")], output_path=output_dir / "promo_display.log"
    )
    promo_metadata = json.loads(
        promo_path.with_suffix(promo_path.suffix + ".json").read_text(encoding="utf-8")
    )

    dashboard_path = build_dashboard(
        db_path=db_path, output_path=output_dir / "admin_dashboard.html"
    )

    # 總結
    duration_seconds = time.perf_counter() - start
    summary = {
        "camera": camera_metrics.to_dict(),
        "recognition": recognition_metrics,
        "stability": stability_report.to_dict(),
        "enrollment": enrollment_payload,
        "api": api_payload,
        "detected_id": detected_id,
        "confidence": confidence,
        "users": [user.__dict__ for user in users],
        "recent_visits": [visit.__dict__ for visit in recent_visits],
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
    *,
    enrollment_message: str,
    detection_message: str,
    detected_id: str,
    confidence: float,
    promotion: str,
    purchase: str,
    users: Iterable[object],
) -> list[str]:
    history_messages = [
        f"歷史消費 - {user.id}: {user.last_purchase}" for user in users
    ]
    baseline_history = history_messages[0] if history_messages else f"建議商品: {purchase}"
    promotion_line = f"{baseline_history}｜推播: {promotion}" if promotion else baseline_history

    messages = [
        enrollment_message or "新用戶已建檔",
        f"辨識ID: {detected_id}",
        f"相似度: {confidence:.2f}",
        f"系統訊息: {detection_message}",
        promotion_line,
    ]

    if len(messages) < 5 and len(history_messages) > 1:
        messages.extend(history_messages[1 : 1 + (5 - len(messages))])
    while len(messages) < 5:
        messages.append(f"建議商品: {purchase}")

    return messages[:5]


if __name__ == "__main__":
    run_pipeline()