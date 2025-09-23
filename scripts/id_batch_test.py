import os, io, time, json, logging, numpy as np
from pathlib import Path
from typing import List
logging.basicConfig(filename='integrated_test.log', level=logging.INFO, format='%(asctime)s %(message)s')

SAMPLES_DIR = Path('samples/faces/personA')  # 放 5 張照
THRESHOLD = float(os.environ.get('THRESHOLD', '0.8'))

def log(msg): print(msg); logging.info(msg)

def crop_face_vision(img_bytes: bytes) -> bytes:
    from google.cloud import vision
    from PIL import Image
    client = vision.ImageAnnotatorClient()
    res = client.face_detection(image=vision.Image(content=img_bytes))
    if not res.face_annotations: return img_bytes
    v = res.face_annotations[0].bounding_poly.vertices
    im = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    xs, ys = [p.x for p in v], [p.y for p in v]
    x1, y1, x2, y2 = max(min(xs),0), max(min(ys),0), min(max(xs),im.width), min(max(ys),im.height)
    if x2<=x1 or y2<=y1: return img_bytes
    bio = io.BytesIO(); im.crop((x1,y1,x2,y2)).save(bio, format='JPEG', quality=92); return bio.getvalue()

def embed(img_bytes: bytes) -> np.ndarray:
    import face_recognition
    arr = face_recognition.load_image_file(io.BytesIO(img_bytes))
    encs = face_recognition.face_encodings(arr)
    return encs[0] if encs else None

def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

def main():
    imgs: List[Path] = sorted(p for p in SAMPLES_DIR.glob('*.jpg'))
    if len(imgs) < 5: 
        log(f"need >=5 images under {SAMPLES_DIR}"); return 1
    # baseline from first image
    e0 = embed(crop_face_vision(imgs[0].read_bytes()))
    if e0 is None: log("no face in baseline"); return 1
    ok=0; sims=[]
    for i,p in enumerate(imgs[:5],1):
        e = embed(crop_face_vision(p.read_bytes()))
        if e is None: 
            log(f"{p.name} no_face"); sims.append(0.0); continue
        s = cos(e0,e); sims.append(s)
        passed = s>=THRESHOLD; ok += int(passed)
        log(f"{p.name} sim={s:.3f} pass={passed}")
    acc = ok/5
    log(f"summary n=5 ok={ok} acc={acc:.3f} threshold={THRESHOLD}")
    # 供題目檔名
    Path('id_test.log').write_text(Path('integrated_test.log').read_text(encoding='utf-8'), encoding='utf-8')
    return 0 if acc>=0.8 else 2

if __name__ == '__main__': raise SystemExit(main())
