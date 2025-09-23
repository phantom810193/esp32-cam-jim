import os, io, time, logging, numpy as np
from pathlib import Path

logging.basicConfig(filename='integrated_test.log', level=logging.INFO, format='%(asctime)s %(message)s')

GCP_PROJECT = os.environ.get('GCP_PROJECT')
COL = os.environ.get('FIRESTORE_COLLECTION', 'visitors')
EVENTS = os.environ.get('FIRESTORE_EVENTS_COLLECTION', 'events')
ESP32_IMAGE_URL = os.environ.get('ESP32_IMAGE_URL', '')
CLOUD_RUN_URL = (os.environ.get('CLOUD_RUN_URL') or '').rstrip('/')
DETECT_PATH = os.environ.get('CLOUD_RUN_DETECT_PATH')  # e.g. /detect_face

def log(msg): 
    print(msg); logging.info(msg)

def fetch_frame_bytes():
    import requests
    if ESP32_IMAGE_URL:
        r = requests.get(ESP32_IMAGE_URL, timeout=10); r.raise_for_status()
        return r.content
    for p in [Path('samples/faces/personA/1.jpg'), Path('samples/face.jpg')]:
        if p.exists(): return p.read_bytes()
    raise FileNotFoundError("No ESP32_IMAGE_URL and no local sample image found")

def crop_face_by_vision(img_bytes: bytes):
    from google.cloud import vision
    from PIL import Image
    client = vision.ImageAnnotatorClient()
    res = client.face_detection(image=vision.Image(content=img_bytes))
    if not res.face_annotations: 
        return img_bytes
    vs = res.face_annotations[0].bounding_poly.vertices
    im = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    xs=[v.x for v in vs]; ys=[v.y for v in vs]
    x1,y1,x2,y2 = max(min(xs),0), max(min(ys),0), min(max(xs), im.width), min(max(ys), im.height)
    if x2<=x1 or y2<=y1: 
        return img_bytes
    bio = io.BytesIO(); im.crop((x1,y1,x2,y2)).save(bio, format='JPEG', quality=92)
    return bio.getvalue()

def embed_face(img_bytes: bytes):
    import face_recognition
    arr = face_recognition.load_image_file(io.BytesIO(img_bytes))
    encs = face_recognition.face_encodings(arr)
    return encs[0] if encs else None

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

def write_event(db, kind, payload):
    db.collection(EVENTS).document(f"{kind}-{int(time.time())}").set({
        'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
        'kind': kind,
        **payload
    })

def post_cloud_run_detect(frame: bytes):
    import requests
    url = CLOUD_RUN_URL; path = DETECT_PATH or ''
    if not url or not path: 
        return
    files={'file': ('frame.jpg', io.BytesIO(frame), 'image/jpeg')}
    r = requests.post(url + path, files=files, timeout=15)
    if r.status_code in (401,403):
        from google.oauth2 import service_account
        from google.auth.transport.requests import AuthorizedSession
        creds = service_account.IDTokenCredentials.from_service_account_file(
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'], target_audience=url
        )
        authed = AuthorizedSession(creds)
        r = authed.post(url + path, files=files, timeout=15)
    log(f"cloudrun detect status={r.status_code} body={r.text[:200]}")

def main():
    assert GCP_PROJECT, "GCP_PROJECT is required"
    frame = fetch_frame_bytes()
    face_crop = crop_face_by_vision(frame)
    enc = embed_face(face_crop)

    if enc is None:
        # 這裡要帶上 database id
        from google.cloud import firestore
        db = firestore.Client(
            project=GCP_PROJECT,
            database=os.environ.get('FIRESTORE_DATABASE_ID','(default)')
        )
        write_event(db, 'no_face', {'note': 'no embedding'})
        log("no-embedding-from-image")
        return 0

    from firestore_embedding_utils import get_db, ensure_enrolled, load_embed
    db = get_db(GCP_PROJECT)

    enrolled = ensure_enrolled(db, COL, 'personA', enc)
    if enrolled: 
        log("enrolled baseline personA")

    ref = load_embed(db, COL, 'personA')
    sim = cos_sim(ref, enc) if ref is not None else 0.0
    ident = 'personA' if sim >= 0.6 else 'unknown'
    write_event(db, 'identify', {'similarity': sim, 'id': ident})
    log(f"identify result id={ident} sim={sim:.3f}")

    post_cloud_run_detect(frame)

if __name__ == "__main__":
    main()
