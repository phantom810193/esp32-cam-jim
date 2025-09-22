from pathlib import Path

from vision import FaceRecognitionService


def test_evaluate_queries_reaches_required_accuracy(tmp_path: Path) -> None:
    service = FaceRecognitionService()
    log_path = tmp_path / "id_test.log"
    report = service.evaluate_queries(log_path=log_path)

    assert report["accuracy"] >= 0.8
    assert report["total_images"] == 5
    assert "ID-abc123" in log_path.read_text(encoding="utf-8")


def test_identify_embedding_flags_new_user() -> None:
    service = FaceRecognitionService(threshold=0.95)
    new_embedding = [0.9, 0.9, 0.9, 0.9]
    person_id, confidence, is_new = service.identify_embedding(new_embedding)

    assert is_new
    assert person_id.startswith("ID-")
    assert confidence < 0.95
