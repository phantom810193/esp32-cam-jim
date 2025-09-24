# scripts/build_embeddings_jsonl.py
# 用專案的 embedding_arcface 產生 Vertex Vector Search 可用的 JSONL 檔。
# 用法（範例）：
#   python scripts/build_embeddings_jsonl.py --input ./faces_raw --out ./datapoints.jsonl
# 也支援環境變數：INPUT_DIR, OUT_JSONL

from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import uuid
from typing import Iterable

import numpy as np
from PIL import Image

# 優先重用專案內的 embedding 模組；失敗時才啟用內建後備
try:
    from embedding_arcface import get_app, embed_image
    _USE_LOCAL_EMB = True
except Exception as e:
    _USE_LOCAL_EMB = False
    _IMPORT_ERR = e

def iter_images(root: str) -> Iterable[str]:
    exts = ("*.jpg", "*.jpeg", "*.png")
    for ext in exts:
        yield from glob.glob(os.path.join(root, "**", ext), recursive=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default=os.environ.get("INPUT_DIR", "./faces_raw"),
                        help="輸入根目錄。結構建議為 <root>/<person_id>/*.jpg")
    parser.add_argument("--out", "-o", default=os.environ.get("OUT_JSONL", "./datapoints.jsonl"),
                        help="輸出 JSONL 檔路徑")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    # 準備 embedding 引擎
    if _USE_LOCAL_EMB:
        app = get_app()
        def _embed(np_img: np.ndarray):
            return embed_image(app, np_img)
    else:
        # 後備：若專案內模組無法匯入，直接在此初始化 insightface，避免卡住流程
        try:
            import insightface
            app = insightface.app.FaceAnalysis(
                name=os.getenv("INSIGHTFACE_MODEL", "buffalo_l"),
                allowed_modules=["detection", "recognition"]
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as e:
            print("ERROR: cannot init insightface:", e, file=sys.stderr)
            print("Original import error:", _IMPORT_ERR, file=sys.stderr)
            sys.exit(2)

        def _embed(np_img: np.ndarray):
            faces = app.get(np_img)
            if not faces:
                raise ValueError("No face detected")
            return faces[0].normed_embedding.astype(float).tolist()

    count_total, count_ok, count_skip = 0, 0, 0

    with open(args.out, "w", encoding="utf-8") as fw:
        for f in iter_images(args.input):
            count_total += 1
            person_id = os.path.basename(os.path.dirname(f))  # 父資料夾名視為身分
            try:
                img = Image.open(f).convert("RGB")
                vec = _embed(np.array(img))
                dp = {
                    "id": str(uuid.uuid4()),
                    "embedding": vec,
                    "restricts": [{"namespace": "person", "allow": [person_id]}],
                    "crowdingTag": f
                }
                fw.write(json.dumps(dp, ensure_ascii=False) + "\n")
                count_ok += 1
            except Exception as e:
                count_skip += 1
                print(f"skip {f}: {e}", file=sys.stderr)

    print(f"done -> {args.out} (ok={count_ok}, skip={count_skip}, total={count_total})")

if __name__ == "__main__":
    main()
