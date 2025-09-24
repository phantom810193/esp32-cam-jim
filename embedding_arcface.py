# embedding_arcface.py
# 統一的人臉嵌入（ArcFace/InsightFace）介面，避免在專案各處重複載入模型。
# 需要套件：insightface、onnxruntime、onnx、numpy、Pillow

from __future__ import annotations
import os
import threading
from typing import List, Tuple

import numpy as np
from PIL import Image

try:
    import insightface
    _INSIGHTFACE_AVAILABLE = True
except Exception as e:
    _INSIGHTFACE_AVAILABLE = False
    _IMPORT_ERR = e

_APP = None           # 全域單例
_LOCK = threading.Lock()


def _pick_largest_face(faces):
    """多臉時挑選最大的人臉框，降低誤選機率。"""
    if not faces:
        return None
    def area(face):
        x1, y1, x2, y2 = face.bbox.astype(float)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return max(faces, key=area)


def get_app(det_size: Tuple[int, int] = (640, 640), ctx_id: int = 0):
    """
    取得 InsightFace 的單例 App。
    det_size: 偵測輸入尺寸
    ctx_id: 0 表 CPU（或第一張 GPU），Cloud Run/Cloud Shell 通常用 0。
    """
    global _APP
    if _APP is not None:
        return _APP
    if not _INSIGHTFACE_AVAILABLE:
        raise RuntimeError(
            "insightface not installed. pip install insightface onnxruntime onnx\n"
            f"original import error: {_IMPORT_ERR}"
        )
    with _LOCK:
        if _APP is None:
            model_name = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")
            app = insightface.app.FaceAnalysis(
                name=model_name,
                allowed_modules=["detection", "recognition"]
            )
            # ctx_id=0 在多數雲端環境會走 CPU（若有 GPU 會用第一張）
            app.prepare(ctx_id=ctx_id, det_size=det_size)
            _APP = app
    return _APP


def embed_image(app, image_rgb: np.ndarray) -> List[float]:
    """
    將 RGB np.uint8 影像轉為 512 維臉部向量（list[float]）。
    若無臉會丟 ValueError。
    """
    if image_rgb is None or not isinstance(image_rgb, np.ndarray):
        raise TypeError("image_rgb must be a numpy array (H,W,3) in RGB")
    faces = app.get(image_rgb)
    face = _pick_largest_face(faces)
    if face is None:
        raise ValueError("No face detected")
    # InsightFace already gives L2-normalized embedding
    return face.normed_embedding.astype(float).tolist()


def embed_pil(image: Image.Image) -> List[float]:
    """
    便利版本：直接吃 PIL.Image，內部呼叫 get_app()。
    """
    app = get_app()
    return embed_image(app, np.array(image.convert("RGB")))
