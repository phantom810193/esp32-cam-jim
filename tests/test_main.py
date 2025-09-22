from pathlib import Path

from main import run_pipeline


def test_run_pipeline_produces_expected_logs(tmp_path: Path) -> None:
    summary = run_pipeline(output_dir=tmp_path)

    assert summary["camera"]["fps"] > 10
    assert summary["recognition"]["accuracy"] >= 0.8
    assert summary["duration_seconds"] < 30

    for filename in ["cam.log", "id_test.log", "text_test.log", "e2e.log"]:
        assert (tmp_path / filename).exists()
