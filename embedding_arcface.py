# embedding_arcface.py
from __future__ import annotations
import base64, io, os
import numpy as np
from PIL import Image
from typing import Optional, Tuple
from google.cloud import vision
import onnxruntime as ort

# 下載或放一個 ArcFace ONNX（512 維），可先用 garavv/arcface-onnx
# 你可以把檔案放在 repo 的 models/arcface.onnx
# 參考: https://huggingface.co/garavv/arcface-onnx
MODEL_PATH = os.getenv("ARCFACE_ONNX", "models/arcface.onnx")

_ort_sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
_vision  = vision.ImageAnnotatorClient()

def _read_image(data_uri_or_bytes: bytes | str) -> Image.Image:
    if isinstance(data_uri_or_bytes, str) and data_uri_or_bytes.startswith("data:image"):
        b64 = data_uri_or_bytes.split(",", 1)[1]
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(data_uri_or_bytes, (bytes, bytearray)):
        return Image.open(io.BytesIO(data_uri_or_bytes)).convert("RGB")
    raise ValueError("unsupported image input")

def _detect_face_bbox(img_bytes: bytes) -> Optional[Tuple[int,int,int,int]]:
    image = vision.Image(content=img_bytes)
    res = _vision.face_detection(image=image)
    if not res.face_annotations:
        return None
    # 取第一張臉 bounding box
    fb = res.face_annotations[0].bounding_poly
    xs = [v.x for v in fb.vertices]; ys = [v.y for v in fb.vertices]
    x1, y1, x2, y2 = max(min(xs), 0), max(min(ys), 0), max(xs), max(ys)
    return int(x1), int(y1), int(x2), int(y2)

def _preprocess_rgb(img: Image.Image) -> np.ndarray:
    # ArcFace 預設 112x112, RGB, normalize 到 [-1,1]
    img = img.resize((112,112))
    arr = np.asarray(img).astype("float32")
    arr = (arr - 127.5) / 128.0
    # NCHW
    arr = np.transpose(arr, (2,0,1))[None, ...]
    return arr

def embed(data_uri_or_bytes: bytes | str) -> np.ndarray:
    # 先用 Vision 抓臉框再裁切
    if isinstance(data_uri_or_bytes, str) and data_uri_or_bytes.startswith("data:image"):
        raw = base64.b64decode(data_uri_or_bytes.split(",",1)[1])
    elif isinstance(data_uri_or_bytes, (bytes, bytearray)):
        raw = bytes(data_uri_or_bytes)
    else:
        raise ValueError("unsupported image input")

    im = Image.open(io.BytesIO(raw)).convert("RGB")
    bbox = _detect_face_bbox(raw)
    if bbox:
        x1,y1,x2,y2 = bbox
        im = im.crop((x1,y1,x2,y2))

    inp = _preprocess_rgb(im)
    out = _ort_sess.run(None, {"input": inp})[0]  # 形狀 [1,512]
    vec = out[0].astype("float32")
    # L2 normalize（跟 Vector Search 設的 UNIT_L2_NORM 一致）
    norm = np.linalg.norm(vec) + 1e-9
    return (vec / norm).astype("float32")