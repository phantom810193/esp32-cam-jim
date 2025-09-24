"""Face recognition helpers leveraging deterministic sample embeddings.

The real deployment integrates Google Cloud Vision and an ArcFace
embedding model stored in GCS. For CI we ship a small JSON dataset that
behaves like the production pipeline so automated tests can validate
accuracy thresholds without accessing binary assets.
"""
from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

DATASET_PATH = Path(__file__).resolve().parent / "sample_data" / "embeddings.json"


@dataclass
class RecognitionResult:
    image: str
    predicted_id: str
    expected_id: str
    confidence: float
    match: bool


def _cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
    a = list(vec_a)
    b = list(vec_b)
    if len(a) != len(b):
        raise ValueError("Embedding vectors must be the same length")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class FaceRecognitionService:
    """Utility object that mimics the behaviour of the production service."""

    def __init__(self, dataset_path: Path | str = DATASET_PATH, threshold: float = 0.8):
        self.dataset_path = Path(dataset_path)
        self.threshold = threshold
        self._dataset_cache: Dict[str, List[Dict[str, object]]] | None = None
        self._enrolled_embeddings: Dict[str, List[List[float]]] = {}

    def _load_dataset(self) -> Dict[str, List[Dict[str, object]]]:
        if self._dataset_cache is None:
            with self.dataset_path.open("r", encoding="utf-8") as handle:
                self._dataset_cache = json.load(handle)
        return self._dataset_cache

    @property
    def customers(self) -> List[Dict[str, object]]:
        base = [dict(item) for item in self._load_dataset()["customers"]]
        dynamic = [
            {"id": user_id, "embedding": embedding}
            for user_id, embeddings in self._enrolled_embeddings.items()
            for embedding in embeddings
        ]
        return base + dynamic

    @property
    def queries(self) -> List[Dict[str, object]]:
        return [dict(item) for item in self._load_dataset()["queries"]]

    def enroll(
        self,
        embeddings: Iterable[Iterable[float]],
        *,
        user_id: str | None = None,
    ) -> Dict[str, object]:
        """Register one or more embeddings for a visitor and return metadata."""

        vectors: List[List[float]] = []
        for embedding in embeddings:
            vector = [float(value) for value in embedding]
            if not vector:
                raise ValueError("embedding vectors must not be empty")
            vectors.append(vector)
        if not vectors:
            raise ValueError("at least one embedding is required")
        if user_id is None:
            user_id = f"ID-{uuid.uuid4().hex[:6]}"
        self._enrolled_embeddings.setdefault(user_id, []).extend(vectors)
        return {"id": user_id, "embeddings": vectors}

    def identify_embedding(self, embedding: Iterable[float]) -> Tuple[str, float, bool]:
        """Return ``(person_id, confidence, is_new_user)`` for an embedding."""

        best_score = -1.0
        best_id = None
        for customer in self.customers:
            score = _cosine_similarity(embedding, customer["embedding"])
            if score > best_score:
                best_score = score
                best_id = str(customer["id"])

        if best_score >= self.threshold and best_id is not None:
            return best_id, best_score, False

        generated_id = f"ID-{uuid.uuid4().hex[:6]}"
        return generated_id, max(best_score, 0.0), True

    def evaluate_queries(self, log_path: Path | str = "id_test.log") -> Dict[str, object]:
        """Run the bundled evaluation set and persist a JSON log."""

        results: List[RecognitionResult] = []
        correct = 0
        for item in self.queries:
            predicted_id, score, is_new = self.identify_embedding(item["embedding"])
            match = predicted_id == item["expected_id"] and not is_new and score >= self.threshold
            if match:
                correct += 1
            results.append(
                RecognitionResult(
                    image=str(item["image"]),
                    predicted_id=predicted_id,
                    expected_id=str(item["expected_id"]),
                    confidence=score,
                    match=match,
                )
            )

        accuracy = correct / len(results) if results else 0.0
        payload = {
            "threshold": self.threshold,
            "total_images": len(results),
            "correct_matches": correct,
            "accuracy": accuracy,
            "results": [result.__dict__ for result in results],
        }

        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload


__all__ = ["FaceRecognitionService", "RecognitionResult", "_cosine_similarity"]
