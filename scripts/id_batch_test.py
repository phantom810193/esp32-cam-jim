# scripts/id_batch_test.py  (ArcFace 版)
from __future__ import annotations
import os, json, time
from pathlib import Path
from typing import List, Tuple
import numpy as np

# 用我們自己寫的 ArcFace + Vision 取向量（不依賴 dlib/face_recognition）
from embedding_arcface import embed as arcface_embed

SAMPLES_DIR = Path(os.getenv("SAMPLES_DIR", "samples/faces/personA"))
N_IMAGES = int(os.getenv("N_IMAGES", "0"))            # 0=全部
THRESHOLD = float(os.getenv("THRESHOLD", "0.8"))      # 命中門檻（cosine）
BASELINE_INDEX = int(os.getenv("BASELINE_INDEX", "-1"))  # -1=自動挑代表向量
LOG_PATH = Path("id_test.log")

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    # 我們的向量本身已 L2 normalize，cos = dot 即可；加個保險
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def _auto_baseline(vecs: List[np.ndarray]) -> int:
    # 挑「和其他圖片平均最相似」的那一張當 baseline
    if len(vecs) == 1:
        return 0
    sims = []
    for i, vi in enumerate(vecs):
        s = 0.0
        for j, vj in enumerate(vecs):
            if i == j: 
                continue
            s += _cos(vi, vj)
        sims.append(s / (len(vecs) - 1))
    return int(np.argmax(sims))

def main() -> int:
    LOG_PATH.unlink(missing_ok=True)

    imgs = sorted([p for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    if N_IMAGES > 0:
        imgs = imgs[:N_IMAGES]
    if not imgs:
        print(f"no images under {SAMPLES_DIR}")
        return 2

    print(f"using {len(imgs)} images from {SAMPLES_DIR}")

    # 取向量
    t0 = time.perf_counter()
    vecs: List[np.ndarray] = []
    for p in imgs:
        raw = p.read_bytes()  # bytes 直接丟給 arcface_embed
        v = arcface_embed(raw)
        vecs.append(v.astype("float32"))
    t1 = time.perf_counter()

    # 決定 baseline
    if BASELINE_INDEX >= 0 and BASELINE_INDEX < len(vecs):
        bidx = BASELINE_INDEX
    else:
        bidx = _auto_baseline(vecs)
    bvec = vecs[bidx]

    # 計算每張與 baseline 的相似度
    ok = 0
    details = []
    for i, (p, v) in enumerate(zip(imgs, vecs)):
        sim = _cos(bvec, v)
        hit = sim >= THRESHOLD
        ok += int(hit)
        details.append({
            "file": p.name,
            "similarity": round(sim, 4),
            "hit": hit
        })

    acc = ok / len(imgs)
    summary = {
        "samples": len(imgs),
        "baseline_index": bidx,
        "threshold": THRESHOLD,
        "accuracy": round(acc, 4),
        "extract_ms": round((t1 - t0) * 1000.0, 2),
    }

    # 輸出到 id_test.log（JSONL：每張 + 最後 summary）
    with LOG_PATH.open("a", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps({"type":"detail", **d}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"type":"summary", **summary}, ensure_ascii=False) + "\n")

    # 同步印出摘要
    print(f"baseline={bidx}, threshold={THRESHOLD}, accuracy={acc:.3f}")
    print(f"wrote {LOG_PATH}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())