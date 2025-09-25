from pathlib import Path

from camera import simulate_camera_capture


def test_camera_log_contains_expected_metrics(tmp_path: Path) -> None:
    log_path = tmp_path / "cam.log"
    metrics = simulate_camera_capture(duration_seconds=5.0, target_fps=12.0, log_path=log_path)

    assert metrics.fps > 10
    assert metrics.frames_captured == int(5.0 * 12.0)

    data = log_path.read_text(encoding="utf-8")
    assert "\"fps\": 12.0" in data
    assert "\"frames_captured\": 60" in data
