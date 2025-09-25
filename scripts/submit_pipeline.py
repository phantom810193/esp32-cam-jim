"""Submit the faces embed + upsert pipeline and wait for completion."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from google.api_core import exceptions
from google.cloud import aiplatform


def configure_logger(log_file: Optional[Path]) -> logging.Logger:
    logger = logging.getLogger("submit_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_template(template_path: Path, logger: logging.Logger) -> Path:
    if template_path.exists():
        logger.info("Using existing pipeline template at %s", template_path)
        return template_path

    compile_script = Path(__file__).resolve().parent.parent / "pipeline" / "compile.py"
    if not compile_script.exists():
        raise FileNotFoundError(
            "Pipeline template not found and compile.py is missing."
        )

    logger.info("Pipeline template %s missing – compiling via %s", template_path, compile_script)
    try:
        subprocess.run([sys.executable, str(compile_script)], check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to compile pipeline specification: {exc}") from exc

    if not template_path.exists():
        raise FileNotFoundError(
            f"Pipeline template expected at {template_path} after compilation, but not found."
        )

    return template_path


def submit_pipeline(args: argparse.Namespace, logger: logging.Logger) -> Dict[str, Any]:
    template_path = ensure_template(Path(args.template), logger)

    if not args.index_name:
        raise ValueError("--index-name is required for pipeline submission")

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    display_name = args.display_name or f"faces-embed-upsert-{timestamp}"
    pipeline_root = args.pipeline_root

    logger.info("Initialising Vertex AI (project=%s, region=%s)", args.project, args.region)
    aiplatform.init(project=args.project, location=args.region)

    parameter_values = {
        "project_id": args.project,
        "region": args.region,
        "index_name": args.index_name,
        "gcs_faces_root": args.gcs_faces_root,
    }
    logger.info("Pipeline parameters: %s", parameter_values)

    pipeline_job = aiplatform.PipelineJob(
        display_name=display_name,
        template_path=str(template_path),
        parameter_values=parameter_values,
        pipeline_root=pipeline_root,
        enable_caching=False,
    )

    logger.info("Submitting pipeline job %s", display_name)
    pipeline_job.submit(
        service_account=args.service_account,
        project=args.project,
        location=args.region,
    )

    logger.info("Submitted job name: %s", pipeline_job.resource_name)
    logger.info("Pipeline console URI: %s", pipeline_job.dashboard_uri)

    try:
        pipeline_job.wait()
    except exceptions.GoogleAPICallError as exc:
        logger.error("Pipeline wait failed: %s", exc)
        raise

    state = pipeline_job.state
    logger.info("Pipeline job %s completed with state: %s", pipeline_job.resource_name, state)

    task_summaries = []
    try:
        for detail in pipeline_job.list_pipeline_task_details():
            task_summaries.append(
                {
                    "task_id": detail.task_id,
                    "task_name": detail.display_name,
                    "state": detail.state.name if detail.state else None,
                    "create_time": detail.create_time.isoformat() if detail.create_time else None,
                    "start_time": detail.start_time.isoformat() if detail.start_time else None,
                    "end_time": detail.end_time.isoformat() if detail.end_time else None,
                    "error": detail.error.message if detail.error else None,
                }
            )
    except exceptions.GoogleAPICallError as exc:
        logger.warning("Unable to list pipeline task details: %s", exc)

    summary: Dict[str, Any] = {
        "job_name": pipeline_job.resource_name,
        "display_name": display_name,
        "state": state.name if state else str(state),
        "start_time": pipeline_job._gca_resource.start_time.isoformat()
        if pipeline_job._gca_resource.start_time
        else None,
        "end_time": pipeline_job._gca_resource.end_time.isoformat()
        if pipeline_job._gca_resource.end_time
        else None,
        "pipeline_root": pipeline_root,
        "dashboard_uri": pipeline_job.dashboard_uri,
        "task_summaries": task_summaries,
        "parameters": parameter_values,
    }

    if state and state.name != "SUCCEEDED":
        raise RuntimeError(f"Pipeline job finished with state {state}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit Vertex AI pipeline job")
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--template", required=True, help="Path to compiled pipeline JSON/YAML")
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--index-name", required=True)
    parser.add_argument("--gcs-faces-root", required=True)
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--display-name", default="")
    parser.add_argument("--log-file", default="")
    parser.add_argument("--summary-file", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_file = Path(args.log_file) if args.log_file else None
    logger = configure_logger(log_file)

    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    logger.info("GOOGLE_APPLICATION_CREDENTIALS=%s", credentials_path or "<not set>")
    try:
        active_account = subprocess.run(
            ["gcloud", "config", "get-value", "account"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:  # noqa: PERF203
        active_account = "<unknown>"
        logger.warning("Unable to determine active gcloud account: %s", exc)
    logger.info("Active gcloud account=%s", active_account)

    try:
        summary = submit_pipeline(args, logger)
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline submission failed: %s", exc)
        logger.error("Troubleshooting suggestions:")
        logger.error("- Verify service account has roles/aiplatform.user and storage object viewer/creator as needed.")
        logger.error("- Confirm Matching Engine index and endpoint exist in region %s and are ready.", args.region)
        logger.error("- Ensure service account has access to Vertex AI Vector Search indexes/endpoints.")
        if log_file is not None:
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(f"ERROR: {exc}\n")
                handle.write(
                    "SUGGESTIONS: Check service account IAM roles, index/endpoint readiness, and regional settings.\n"
                )
        raise

    if args.summary_file:
        summary_path = Path(args.summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Pipeline summary written to %s", summary_path)

    logger.info(
        "Pipeline job %s finished successfully with state %s",
        summary.get("job_name", "<unknown>"),
        summary.get("state", "<unknown>"),
    )


if __name__ == "__main__":
    main()
