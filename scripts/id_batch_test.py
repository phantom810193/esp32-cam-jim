# scripts/id_batch_test.py
import os, io, logging, tempfile, numpy as np
from pathlib import Path
from typing import List, Optional

logging.basicConfig(filename='integrated_test.log',
                    level=logging.INFO,
                    format='%(asctime)s %(message)s')

SAMPLES_DIR    = Path(os.environ.get('SAMPLES_DIR', 'samples/faces/personA'))
N_IMAGES       = int(os.environ.get('N_IMAGES', '0'))           # 0=全部
THRESHOLD      = float(os.environ.get('THRESHOLD', '0.8'))
TARGET_ACC     = float(os.environ.get('TARGET_ACC', '0.8'))
BASELINE_INDEX = int(os.environ.get('BASELINE_INDEX', '-1'))    # -1 自動
USE_VISION     = os.environ.get('USE_VISION', '1') == '1'
DEBUG_FACE     = os.environ.get('DEBUG_FACE', '1') == '1'       # 預設開

DBG_DIR = Path('debug'); DBG_DIR.mkdir(exist_ok=True)

def log(msg: str):
    print(msg); logging.info(msg)

def to_rgb_jpeg_bytes(img_bytes: bytes) -> bytes:
    from PIL import Image
    im = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    bio = io.BytesIO(); im.save(bio, format='JPEG', quality=92)
    return bio.getvalue()

def cv2_rgb_from_bytes(jpg_bytes: bytes) -> Optional[np.ndarray]:
    """OpenCV 解碼 -> BGR -> 轉 RGB -> 連續 uint8 陣列"""
    import cv2
    buf = np.frombuffer(jpg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)   # HxWx3 (BGR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    return rgb

def crop_face_vision(img_bytes: bytes) -> bytes:
    if not USE_VISION:
        return img_bytes
    try:
        from google.cloud import vision
        from PIL import Image
        client = vision.ImageAnnotatorClient()
        res = client.face_detection(image=vision.Image(content=img_bytes))
        if not res.face_annotations:
            return img_bytes
        v = res.face_annotations[0].bounding_poly.vertices
        im = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        xs, ys = [p.x for p in v], [p.y for p in v]
        x1, y1 = max(min(xs), 0), max(min(ys), 0)
        x2, y2 = min(max(xs), im.width), min(max(ys), im.height)
        if x2 <= x1 or y2 <= y1:
            return img_bytes
        bio = io.BytesIO(); im.crop((x1, y1, x2, y2)).save(bio, format='JPEG', quality=92)
        return bio.getvalue()
    except Exception as e:
        log(f"vision_crop_error: {e}")
        return img_bytes

def embed(img_bytes: bytes, tag: str) -> Optional[np.ndarray]:
    """OpenCV 強制解碼路線 + 詳細記錄，最後丟給 face_recognition。"""
    import face_recognition, cv2
    try:
        jpg = to_rgb_jpeg_bytes(img_bytes)       # 先統一到 JPEG
        arr = cv2_rgb_from_bytes(jpg)            # 再用 OpenCV 確保 HxWx3 uint8 連續
        if arr is None:
            log(f"{tag} decode_error: cv2.imdecode failed")
            return None

        info = f"{tag} shape={arr.shape} dtype={arr.dtype} contiguous={arr.flags['C_CONTIGUOUS']}"
        log("image_info " + info)
        if DEBUG_FACE:
            cv2.imwrite(str(DBG_DIR / f"{tag}_rgb.jpg"), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

        encs = face_recognition.face_encodings(arr)
        if not encs:
            log(f"{tag} no_face_after_convert")
            return None
        return encs[0]
    except Exception as e:
        log(f"{tag} embed_error: {e}")
        return None

def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def main() -> int:
    imgs: List[Path] = sorted(p for p in SAMPLES_DIR.glob('*')
                              if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'))
    if not imgs:
        log(f"no images found under {SAMPLES_DIR}"); return 1
    use_imgs = imgs if N_IMAGES <= 0 else imgs[:N_IMAGES]
    log(f"using {len(use_imgs)} images from {SAMPLES_DIR}")

    # baseline：指定或自動找第一張能取到 embedding 的
    baseline = None; baseline_name = None
    candidates = [use_imgs[BASELINE_INDEX]] if 0 <= BASELINE_INDEX < len(use_imgs) else use_imgs
    for p in candidates:
        e0 = embed(crop_face_vision(p.read_bytes()), f"baseline_{p.name}")
        if e0 is not None:
            baseline = e0; baseline_name = p.name; log(f"baseline={baseline_name}"); break
    if baseline is None:
        log("no usable baseline (no face found in selected images)")
        try:
            Path('id_test.log').write_text(Path('integrated_test.log').read_text(encoding='utf-8'), encoding='utf-8')
        except Exception:
            pass
        return 2

    ok = 0
    for p in use_imgs:
        e = embed(crop_face_vision(p.read_bytes()), p.name)
        if e is None:
            log(f"{p.name} sim=0.000 pass=False (no_face)"); continue
        s = cos(baseline, e); passed = s >= THRESHOLD; ok += int(passed)
        log(f"{p.name} sim={s:.3f} pass={passed}")

    n = len(use_imgs); acc = ok / n
    log(f"summary n={n} ok={ok} acc={acc:.3f} threshold={THRESHOLD} target_acc={TARGET_ACC}")

    try:
        Path('id_test.log').write_text(Path('integrated_test.log').read_text(encoding='utf-8'), encoding='utf-8')
    except Exception:
        pass
    return 0 if acc >= TARGET_ACC else 3

if __name__ == '__main__':
    raise SystemExit(main())
