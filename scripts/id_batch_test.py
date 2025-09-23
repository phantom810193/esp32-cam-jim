# scripts/id_batch_test.py
import os, io, logging, numpy as np
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    filename='integrated_test.log',
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)

# ===== 可用環境變數 =====
SAMPLES_DIR = Path(os.environ.get('SAMPLES_DIR', 'samples/faces/personA'))
N_IMAGES = int(os.environ.get('N_IMAGES', '0'))          # 0=用全部
THRESHOLD = float(os.environ.get('THRESHOLD', '0.8'))    # 單張相似度門檻
TARGET_ACC = float(os.environ.get('TARGET_ACC', '0.8'))  # 整體正確率門檻
# =====================

def log(msg: str):
    print(msg); logging.info(msg)

def to_rgb_uint8(img_bytes: bytes) -> np.ndarray:
    """把任何格式統一轉 8-bit RGB 並確保 C-contiguous。"""
    from PIL import Image
    im = Image.open(io.BytesIO(img_bytes))
    mode_before = im.mode
    im = im.convert('RGB')  # RGBA/CMYK/16-bit 等一律轉 RGB
    arr = np.array(im, dtype=np.uint8, copy=True)
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    logging.info(f"image_info mode_before={mode_before} size={im.size} shape={arr.shape} dtype={arr.dtype} contig={arr.flags['C_CONTIGUOUS']}")
    return arr

def crop_face_vision(img_bytes: bytes) -> bytes:
    """用 Cloud Vision 取第一張臉並裁切；沒有臉就回原圖。"""
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
    bio = io.BytesIO()
    im.crop((x1, y1, x2, y2)).save(bio, format='JPEG', quality=92)
    return bio.getvalue()

def embed(img_bytes: bytes) -> Optional[np.ndarray]:
    """取得 128D 臉部向量；失敗回 None。"""
    import face_recognition
    try:
        arr = to_rgb_uint8(img_bytes)
        encs = face_recognition.face_encodings(arr)
        if not encs:
            log("no_face_after_convert")
            return None
        return encs[0]
    except Exception as e:
        log(f"embed_error: {e}")
        return None

def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def main() -> int:
    imgs: List[Path] = sorted(p for p in SAMPLES_DIR.glob('*')
                              if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'))
    if not imgs:
        log(f"no images found under {SAMPLES_DIR}")
        return 1

    # 要測的清單：N_IMAGES<=0 用全部；>0 則取前 N 張
    use_imgs = imgs if N_IMAGES <= 0 else imgs[:N_IMAGES]
    log(f"using {len(use_imgs)} images from {SAMPLES_DIR}")

    # 採用第一張能取到 embedding 的圖片當 baseline
    baseline = None; baseline_name = None
    for p in use_imgs:
        e0 = embed(crop_face_vision(p.read_bytes()))
        if e0 is not None:
            baseline = e0; baseline_name = p.name
            log(f"baseline={baseline_name}")
            break
    if baseline is None:
        log("no usable baseline (no face found in selected images)")
        # 也輸出 id_test.log 讓你下載查看
        try:
            Path('id_test.log').write_text(Path('integrated_test.log').read_text(encoding='utf-8'), encoding='utf-8')
        except Exception:
            pass
        return 2

    ok, sims = 0, []
    for p in use_imgs:
        e = embed(crop_face_vision(p.read_bytes()))
        if e is None:
            sims.append(0.0)
            log(f"{p.name} sim=0.000 pass=False (no_face)")
            continue
        s = cos(baseline, e)
        sims.append(s)
        passed = s >= THRESHOLD
        ok += int(passed)
        log(f"{p.name} sim={s:.3f} pass={passed}")

    n = len(use_imgs)
    acc = ok / n
    log(f"summary n={n} ok={ok} acc={acc:.3f} threshold={THRESHOLD} target_acc={TARGET_ACC}")

    # 題目指定檔名：同步輸出
    try:
        Path('id_test.log').write_text(Path('integrated_test.log').read_text(encoding='utf-8'), encoding='utf-8')
    except Exception:
        pass

    return 0 if acc >= TARGET_ACC else 3

if __name__ == '__main__':
    raise SystemExit(main())
