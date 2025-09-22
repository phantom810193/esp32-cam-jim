"""End-to-end orchestration script for the demo pipeline."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable

from camera import simulate_camera_capture
from database_utils import connect, fetch_all_users, initialize_database
from display import render_messages
from vision import FaceRecognitionService


def run_pipeline(
    *,
    output_dir: Path | str = Path("."),
    purchase: str = "Milk",
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    initialize_database(output_dir / "users.db", Path("data.sql"))

    start = time.perf_counter()
    camera_metrics = simulate_camera_capture(
        duration_seconds=5.0,
        target_fps=12.0,
        log_path=output_dir / "cam.log",
    )

    service = FaceRecognitionService()
    recognition_metrics = service.evaluate_queries(log_path=output_dir / "id_test.log")

    query = service.queries[0]
    detected_id, confidence, _ = service.identify_embedding(query["embedding"])

    with connect(output_dir / "users.db") as connection:
        users = fetch_all_users(connection)

    messages = _build_messages(detected_id, confidence, purchase, users)
    render_messages(messages, log_path=output_dir / "text_test.log", enable_print=False)

    duration_seconds = time.perf_counter() - start
    summary = {
        "camera": camera_metrics.to_dict(),
        "recognition": recognition_metrics,
        "detected_id": detected_id,
        "confidence": confidence,
        "messages": messages,
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
) -> list[str]:
    formatted = [
        f"辨識ID: {detected_id}",
        f"相似度: {confidence:.2f}",
        f"建議商品: {purchase}",
    ]
    formatted.extend([f"歷史消費 - {user.id}: {user.last_purchase}" for user in users][:2])
    formatted.append("上次買牛奶，現9折！")
    return formatted[:5]


if __name__ == "__main__":
    run_pipeline()
