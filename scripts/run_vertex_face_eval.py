#!/usr/bin/env python3
"""Comprehensive multi-person face evaluation against Vertex Matching Engine."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
from google.api_core import retry as ga_retry
from google.api_core import exceptions as gax_exceptions
from google.cloud import aiplatform_v1
from insightface.app import FaceAnalysis


@dataclass
class EvaluationResult:
    image_path: Path
    actual_label: Optional[str]
    predicted_id: Optional[str]
    predicted_label: Optional[str]
    distance: Optional[float]
    passed: bool
    notes: Optional[str]


@dataclass
class NeighborRecord:
    predicted_id: Optional[str]
    predicted_label: Optional[str]
    distance: Optional[float]


THRESHOLD_TABLE = [0.15, 0.20, 0.25, 0.30]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Vertex AI Matching Engine faces.")
    parser.add_argument("--endpoint", required=True, help="Full resource name of the index endpoint.")
    parser.add_argument("--deploy-id", required=True, help="Deployed index ID to query.")
    parser.add_argument("--images-dir", required=True, help="Directory containing face images organised by label.")
    parser.add_argument("--out-dir", required=True, help="Directory to write evaluation outputs.")
    parser.add_argument("--threshold", type=float, default=0.25, help="Cosine distance threshold for match pass/fail.")
    parser.add_argument("--neighbors", type=int, default=3, help="Number of neighbors to request from the index.")
    parser.add_argument("--region", help="Override the Vertex AI region (defaults to the region from endpoint).")
    return parser.parse_args()


def extract_region(endpoint: str) -> str:
    if "/locations/" not in endpoint:
        raise ValueError(f"Unable to determine region from endpoint: {endpoint}")
    return endpoint.split("/locations/")[1].split("/")[0]


def init_logger(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("vertex_face_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    log_path = out_dir / "debug.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def format_optional(value: Optional[float]) -> str:
    return f"{value:.4f}" if value is not None else "N/A"


def collect_images(images_dir: Path) -> List[Path]:
    valid_suffixes = {".jpg", ".jpeg", ".png"}
    paths = [
        p for p in sorted(images_dir.rglob("*"))
        if p.suffix.lower() in valid_suffixes
    ]
    return paths


def label_from_path(path: Path, root: Path) -> Optional[str]:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    parts = [part for part in relative.parts if part not in ("", ".")]
    if len(parts) >= 2:
        return parts[-2]
    return None


def label_from_datapoint_id(datapoint_id: Optional[str]) -> Optional[str]:
    if not datapoint_id:
        return None
    cleaned = datapoint_id.replace("\\", "/")
    parts = [part for part in cleaned.split("/") if part and not part.startswith("gs:")]
    if len(parts) >= 2:
        return parts[-2]
    return None


def init_face_analysis(logger: logging.Logger) -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l")
    try:
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as exc:
        logger.warning("FaceAnalysis ctx_id=0 failed (%s); falling back to CPU ctx_id=-1", exc)
        app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def build_match_client(region: str) -> aiplatform_v1.MatchServiceClient:
    return aiplatform_v1.MatchServiceClient(
        client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
    )


def query_neighbors(
    client: aiplatform_v1.MatchServiceClient,
    endpoint: str,
    deploy_id: str,
    embedding: Sequence[float],
    neighbors: int,
    logger: logging.Logger,
) -> NeighborRecord:
    request = aiplatform_v1.FindNeighborsRequest(
        index_endpoint=endpoint,
        deployed_index_id=deploy_id,
        queries=[
            aiplatform_v1.FindNeighborsRequest.Query(
                datapoint=aiplatform_v1.IndexDatapoint(feature_vector=list(map(float, embedding))),
                neighbor_count=neighbors,
            )
        ],
    )
    try:
        response = ga_retry.Retry(initial=1.0, maximum=10.0, multiplier=2.0, deadline=60.0)(
            client.find_neighbors
        )(request=request)
    except (gax_exceptions.GoogleAPICallError, gax_exceptions.RetryError, gax_exceptions.ServiceUnavailable) as exc:
        logger.error("MatchService find_neighbors failed: %s", exc)
        return NeighborRecord(None, None, None)

    if not response.nearest_neighbors or not response.nearest_neighbors[0].neighbors:
        return NeighborRecord(None, None, None)

    top_neighbor = response.nearest_neighbors[0].neighbors[0]
    predicted_id = top_neighbor.datapoint.datapoint_id or None
    predicted_label = label_from_datapoint_id(predicted_id)
    return NeighborRecord(predicted_id, predicted_label, top_neighbor.distance)


def evaluate_images(
    images: Iterable[Path],
    images_dir: Path,
    app: FaceAnalysis,
    client: aiplatform_v1.MatchServiceClient,
    endpoint: str,
    deploy_id: str,
    neighbors: int,
    threshold: float,
    logger: logging.Logger,
) -> Tuple[List[EvaluationResult], List[NeighborRecord]]:
    results: List[EvaluationResult] = []
    neighbor_details: List[NeighborRecord] = []

    for path in images:
        label = label_from_path(path, images_dir)
        img = cv2.imread(str(path))
        if img is None:
            note = "image_read_failed"
            logger.warning("Failed to read image %s", path)
            results.append(EvaluationResult(path, label, None, None, None, False, note))
            neighbor_details.append(NeighborRecord(None, None, None))
            continue

        faces = app.get(img)
        if not faces:
            note = "no_face_detected"
            logger.warning("No face detected in %s", path)
            results.append(EvaluationResult(path, label, None, None, None, False, note))
            neighbor_details.append(NeighborRecord(None, None, None))
            continue

        embedding = faces[0].normed_embedding.astype(float)
        neighbor = query_neighbors(client, endpoint, deploy_id, embedding, neighbors, logger)
        neighbor_details.append(neighbor)

        passed = False
        distance = neighbor.distance
        predicted_label = neighbor.predicted_label
        predicted_id = neighbor.predicted_id
        note = None
        if distance is not None and distance <= threshold and predicted_label:
            passed = label is not None and predicted_label == label
        if predicted_id is None and distance is None:
            note = "no_neighbors"
        results.append(
            EvaluationResult(path, label, predicted_id, predicted_label, distance, passed, note)
        )

    return results, neighbor_details


def compute_metrics(
    results: Sequence[EvaluationResult],
    neighbor_details: Sequence[NeighborRecord],
    threshold: float,
) -> Tuple[float, Dict[str, Dict[str, Optional[float]]], Counter, int]:
    label_metrics: Dict[str, Dict[str, Optional[float]]] = {}
    matrix_counter: Counter = Counter()
    labeled_count = 0
    correct = 0

    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for res, neighbor in zip(results, neighbor_details):
        actual = res.actual_label
        predicted_label = neighbor.predicted_label if neighbor.predicted_label else None
        distance = neighbor.distance
        passed = distance is not None and distance <= threshold and predicted_label is not None

        if actual is not None:
            labeled_count += 1
            predicted_for_matrix = "UNKNOWN"
            if passed:
                if predicted_label == actual:
                    correct += 1
                    stats[actual]["tp"] += 1
                    predicted_for_matrix = actual
                else:
                    stats[actual]["fn"] += 1
                    if predicted_label:
                        stats[predicted_label]["fp"] += 1
                        predicted_for_matrix = predicted_label
                matrix_counter[(actual, predicted_for_matrix)] += 1
            else:
                stats[actual]["fn"] += 1
                matrix_counter[(actual, "UNKNOWN")] += 1
        else:
            if passed and predicted_label:
                stats[predicted_label]["fp"] += 1

    accuracy = correct / labeled_count if labeled_count else 0.0

    for label, values in stats.items():
        tp = values["tp"]
        fp = values["fp"]
        fn = values["fn"]
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        label_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return accuracy, label_metrics, matrix_counter, labeled_count


def accuracy_for_thresholds(
    results: Sequence[EvaluationResult],
    neighbor_details: Sequence[NeighborRecord],
    thresholds: Sequence[float],
) -> List[Dict[str, float]]:
    records = []
    for thresh in thresholds:
        correct = 0
        labeled = 0
        for res, neighbor in zip(results, neighbor_details):
            actual = res.actual_label
            predicted_label = neighbor.predicted_label
            distance = neighbor.distance
            passed = distance is not None and distance <= thresh and predicted_label is not None
            if actual is not None:
                labeled += 1
                if passed and predicted_label == actual:
                    correct += 1
        accuracy = correct / labeled if labeled else 0.0
        records.append({"threshold": thresh, "top1_accuracy": accuracy})
    return records


def write_matches_csv(results: Sequence[EvaluationResult], out_path: Path) -> None:
    fieldnames = [
        "image_path",
        "actual_label",
        "predicted_id",
        "predicted_label",
        "distance",
        "passed",
        "notes",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(
                {
                    "image_path": res.image_path.as_posix(),
                    "actual_label": res.actual_label or "",
                    "predicted_id": res.predicted_id or "",
                    "predicted_label": res.predicted_label or "",
                    "distance": "" if res.distance is None else f"{res.distance:.6f}",
                    "passed": str(res.passed),
                    "notes": res.notes or "",
                }
            )


def write_confusion_matrix(counter: Counter, out_path: Path) -> None:
    if not counter:
        return
    with out_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["actual_label", "predicted_label", "count"])
        for (actual, predicted), count in sorted(counter.items()):
            writer.writerow([actual, predicted, count])


def write_summary(
    out_dir: Path,
    args: argparse.Namespace,
    accuracy: float,
    label_metrics: Dict[str, Dict[str, Optional[float]]],
    thresholds: List[Dict[str, float]],
    results: Sequence[EvaluationResult],
    labeled_count: int,
) -> None:
    total_images = len(results)
    images_with_faces = sum(1 for res in results if res.notes not in {"image_read_failed", "no_face_detected"})
    no_face = sum(1 for res in results if res.notes == "no_face_detected")
    read_failures = sum(1 for res in results if res.notes == "image_read_failed")
    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "endpoint": args.endpoint,
        "deployed_index_id": args.deploy_id,
        "images_dir": str(Path(args.images_dir).resolve()),
        "threshold": args.threshold,
        "neighbor_count": args.neighbors,
        "top1_accuracy": accuracy,
        "total_images": total_images,
        "images_with_faces": images_with_faces,
        "images_without_faces": no_face,
        "image_read_failures": read_failures,
        "labeled_samples": labeled_count,
        "threshold_accuracies": thresholds,
        "per_label_metrics": label_metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def render_report(
    out_dir: Path,
    accuracy: float,
    thresholds: Sequence[Dict[str, float]],
    label_metrics: Dict[str, Dict[str, Optional[float]]],
    primary_threshold: float,
) -> None:
    rows = "".join(
        f"<tr><td>{row['threshold']:.2f}</td><td>{row['top1_accuracy']:.4f}</td></tr>" for row in thresholds
    )
    label_rows = "".join(
        f"<tr><td>{label}</td><td>{format_optional(metrics.get('precision'))}</td>"
        f"<td>{format_optional(metrics.get('recall'))}</td><td>{metrics['tp']}</td>"
        f"<td>{metrics['fp']}</td><td>{metrics['fn']}</td></tr>"
        for label, metrics in sorted(label_metrics.items())
    )
    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Vertex Face Evaluation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; }}
    table {{ border-collapse: collapse; width: 60%; margin-bottom: 2rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }}
    th {{ background: #f0f0f0; }}
  </style>
</head>
<body>
  <h1>Vertex Matching Engine Face Evaluation</h1>
  <p><strong>Top-1 Accuracy @ threshold:</strong> {accuracy:.4f}</p>
  <h2>Accuracy by Threshold</h2>
  <table>
    <thead><tr><th>Threshold</th><th>Top-1 Accuracy</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h2>Per-label Metrics (threshold {primary_threshold:.2f})</h2>
  <table>
    <thead><tr><th>Label</th><th>Precision</th><th>Recall</th><th>TP</th><th>FP</th><th>FN</th></tr></thead>
    <tbody>{label_rows if label_rows else '<tr><td colspan="6">No labeled samples.</td></tr>'}</tbody>
  </table>
</body>
</html>
"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = init_logger(out_dir)
    region = args.region or extract_region(args.endpoint)
    logger.info("Using endpoint %s (deploy_id=%s) in region %s", args.endpoint, args.deploy_id, region)

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        logger.error("Images directory not found: %s", images_dir)
        raise SystemExit(1)

    images = collect_images(images_dir)
    if not images:
        logger.warning("No images found under %s. Exiting without evaluation.", images_dir)
        return

    logger.info("Found %d images for evaluation", len(images))

    app = init_face_analysis(logger)
    client = build_match_client(region)

    results, neighbors = evaluate_images(
        images,
        images_dir,
        app,
        client,
        args.endpoint,
        args.deploy_id,
        args.neighbors,
        args.threshold,
        logger,
    )

    accuracy, label_metrics, matrix_counter, labeled_count = compute_metrics(results, neighbors, args.threshold)
    threshold_records = accuracy_for_thresholds(results, neighbors, sorted(set(THRESHOLD_TABLE + [args.threshold])))

    write_matches_csv(results, out_dir / "matches.csv")
    write_confusion_matrix(matrix_counter, out_dir / "confusion_matrix.csv")
    write_summary(out_dir, args, accuracy, label_metrics, threshold_records, results, labeled_count)
    render_report(out_dir, accuracy, threshold_records, label_metrics, args.threshold)

    logger.info("Evaluation completed. Top-1 accuracy: %.4f", accuracy)


if __name__ == "__main__":
    main()
