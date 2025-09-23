# vision.py
"""Face recognition helpers (CI-friendly) with safe dimension handling.

- Keep the bundled small JSON dataset (typically 4-dim) for CI.
- NEVER raise on dimension mismatch. Cosine returns None if dims differ.
- identify_embedding() only compares against customers whose embedding
  has the SAME dimension as the query embedding.

Production now uses ArcFace (512-d) computed server-side in api.py.
This module stays useful for tests / sample evaluation without crashing.
"""
from __future__ import annotations

import json
import math
import uuid
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

DATASET_PATH = Path(__file__).resolve().parent / "sample_data" / "embeddings.json"


@dataclass
class RecognitionResult:
    image: str
    predicted_id: str
    expected_id: str
    confidence: float
    match: bool


def _cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> Optional[float]:
    """Cosine similarity. Return None when dimensions differ (no crash)."""
    a = list(float(x) for x in vec_a)
    b = list(float(x) for x in vec_b)
    if len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class FaceRecognitionService:
    """Utility object that mimics the behaviour of the production service."""

    def __init__(self, dataset_path: Path | str = DATASET_PATH, threshold: float = 0.8):
        self.dataset_path = Path(dataset_path)
        self.threshold = threshold
        self._dataset_cache: Dict[str, List[Dict[str, object]]] | None = None
        # dynamically enrolled vectors (for tests)
        self._enrolled_embeddings: Dict[str, List[List[float]]] = {}

    # ---------- dataset ----------
    def _load_dataset(self) -> Dict[str, List[Dict[str, object]]]:
        if self._dataset_cache is None:
            with self.dataset_path.open("r", encoding="utf-8") as handle:
                self._dataset_cache = json.load(handle)
        return self._dataset_cache

    @property
    def customers(self) -> List[Dict[str, object]]:
        base = [dict(item) for item in self._load_dataset().get("customers", [])]
        dynamic = [
            {"id": user_id, "embedding": embedding}
            for user_id, embeddings in self._enrolled_embeddings.items()
            for embedding in embeddings
        ]
        return base + dynamic

    @property
    def queries(self) -> List[Dict[str, object]]:
        return [dict(item) for item in self._load_dataset().get("queries", [])]

    # ---------- optional embedding helpers (for images/urls) ----------
    def embed_bytes(self, data: bytes) -> List[float]:
        """Derive a tiny deterministic embedding from bytes (CI-friendly 4-d)."""
        d = hashlib.sha256(data).digest()  # 32 bytes
        vec: List[float] = []
        for i in range(0, 16, 4):  # 4 dims
            ui = int.from_bytes(d[i:i + 4], "big", signed=False)
            vec.append(ui / 0xFFFFFFFF)
        return vec

    # aliases some clients may look for
    def embedding_from_bytes(self, data: bytes) -> List[float]:
        return self.embed_bytes(data)

    def embed_image(self, data: bytes) -> List[float]:
        return self.embed_bytes(data)

    def embed(self, data: bytes) -> List[float]:
        return self.embed_bytes(data)

    # ---------- enrollment ----------
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
        return {
            "id": user_id,
            "embeddings": vectors,
            "embeddings_count": len(vectors),
            "dim": len(vectors[0]),
        }

    # ---------- identify & evaluate ----------
    def identify_embedding(self, embedding: Iterable[float]) -> Tuple[str, float, bool]:
        """Return (person_id, confidence, is_new_user) for an embedding.

        Only compares against customers whose embedding dimension equals
        the query embedding dimension. Dimension mismatches are skipped.
        """
        query = list(float(x) for x in embedding)
        qdim = len(query)

        best_score = -1.0
        best_id: Optional[str] = None

        for customer in self.customers:
            base = customer.get("embedding")
            if not isinstance(base, (list, tuple)):
                continue
            if len(base) != qdim:
                # skip mismatched dims
                continue
            score = _cosine_similarity(query, base)
            if score is None:
                continue
            if score > best_score:
                best_score = score
                best_id = str(customer.get("id"))

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
                    image=str(item.get("image")),
                    predicted_id=predicted_id,
                    expected_id=str(item.get("expected_id")),
                    confidence=float(score),
                    match=bool(match),
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