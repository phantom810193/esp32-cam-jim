"""Compile the faces-embed-upsert pipeline to JSON."""

from __future__ import annotations

import importlib
import pathlib

from kfp import compiler

PIPELINE_MODULE = "pipeline.pipeline"
PIPELINE_FUNC = "faces_embed_upsert_pipeline"


def main() -> None:
    package_path = pathlib.Path(__file__).with_name("pipeline.json")
    module = importlib.import_module(PIPELINE_MODULE)
    pipeline_func = getattr(module, PIPELINE_FUNC)
    compiler.Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=str(package_path),
    )
    print(f"[INFO] Pipeline specification written to {package_path}")


if __name__ == "__main__":
    main()
