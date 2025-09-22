from pathlib import Path

from stability import run_stability_assessment


def test_stability_report_meets_threshold(tmp_path: Path) -> None:
    log_path = tmp_path / "cam_stable.log"
    report = run_stability_assessment(log_path=log_path)

    assert report.duration_minutes >= 30
    assert report.fps > 10
    assert report.accuracy >= 0.8
    assert report.crashes == 0

    contents = log_path.read_text(encoding="utf-8")
    assert "\"status\": \"stable\"" in contents
