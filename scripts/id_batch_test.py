# scripts/id_batch_test.py
import os, io, time, logging, numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

logging.basicConfig(filename='integrated_test.log',
                    level=logging.INFO,
                    format='%(asctime)s %(message)s')

SAMPLES_DIR = Path('samples/faces/personA')     # 放 >=5 張測試照
THRESHOLD = float(os.environ.get('THRESHOLD', '0.8'))

def log(msg: str):
    print(msg); logging.info(msg)

def crop_face_vision(img_bytes: bytes) -> bytes:
    """用 Cloud Vision 找第一張臉並裁切；找不到就回原圖。"""
    from google.cloud import vision
    from PIL import Image
    client = vision.ImageAnnotatorClient()
    res = client.face_detection(image=vision.Image(content=img_bytes))
    if not res.face_annotations:
        return img_bytes
    v = res.face_annotations[0].bounding_poly.vertices
    im = Image.open(io.BytesIO(img_bytes))
    im = im.convert('RGB')  # 先轉成 8-bit RGB
    xs, ys = [p.x for p in v], [p.y for p in v]
    x1, y1 = max(min(xs), 0), max(min(ys), 0)
    x2, y2 = min(max(xs), im.width), min(max(ys), im.height)
    if x2 <= x1 or y2 <= y1:
        return img_bytes
    bio = io.BytesIO()
    im.crop((x1, y1, x2, y2)).save(bio, format='JPEG', quality=92)
    return bio.getvalue()

def embed(img_bytes: bytes) -> Optional[np.ndarray]:
    """保證轉成 8-bit RGB 後再丟給 face_recognition。"""
    import face_recognition
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(img_bytes))
        mode_before = im.mode
        if im.mode not in ('RGB', 'L'):
            im = im.convert('RGB')      # RGBA/CMYK/16-bit 等一律轉 RGB
        elif im.mode == 'L':            # 8-bit Gray 也可，但統一轉 RGB 比較穩
            im = im.convert('RGB')
        arr = np.asarray(im)
        encs = face_recognition.face_encodings(arr)  # model='small' 也可更快
        if not encs:
            log(f"no_face_after_convert mode_before={mode_before} size={im.size}")
            return None
        return encs[0]
    except Exception as e:
        log(f"embed_error: {e}")
        return None

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def main() -> int:
    imgs: List[Path] = sorted(p for p in SAMPLES_DIR.glob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png','.webp'))
    if len(imgs) < 5:
        log(f"need >=5 images under {SAMPLES_DIR}, found {len(imgs)}"); return 1

    # 找一張可用的 baseline
    baseline = None
    for p in imgs[:5]:
        e0 = embed(crop_face_vision(p.read_bytes()))
        if e0 is not None:
            baseline = e0
            log(f"baseline={p.name}")
            break
    if baseline is None:
        log("no usable baseline (no face in first 5 images)"); return 2

    ok, sims = 0, []
    for i, p in enumerate(imgs[:5], 1):
        e = embed(crop_face_vision(p.read_bytes()))
        if e is None:
            sims.append(0.0)
            log(f"{p.name} sim=0.000 pass=False (no_face)")
            continue
        s = cosine(baseline, e)
        sims.append(s)
        passed = s >= THRESHOLD
        ok += int(passed)
        log(f"{p.name} sim={s:.3f} pass={passed}")

    acc = ok / 5
    log(f"summary n=5 ok={ok} acc={acc:.3f} threshold={THRESHOLD}")

    # 題目指定檔名：同步輸出
    try:
        Path('id_test.log').write_text(Path('integrated_test.log').read_text(encoding='utf-8'), encoding='utf-8')
    except Exception:
        pass

    return 0 if acc >= 0.8 else 3

if __name__ == '__main__':
    raise SystemExit(main())
