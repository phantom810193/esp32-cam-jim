#
import os, time, json, io
from pathlib import Path

CLOUD_RUN_URL=(os.environ.get('CLOUD_RUN_URL') or '').rstrip('/')
DETECT_PATH=os.environ.get('CLOUD_RUN_DETECT_PATH') or '/detect_face'
IMG = Path('samples/faces/personA/1.jpg')

def write(line): 
    with open('api_test.log','a',encoding='utf-8') as f: f.write(line+'\n')
    print(line)

def embed(img_bytes: bytes):
    import face_recognition
    arr = face_recognition.load_image_file(io.BytesIO(img_bytes))
    encs = face_recognition.face_encodings(arr)
    return encs[0].tolist() if encs else None

def main():
    import requests
    if not CLOUD_RUN_URL: 
        write("skip: CLOUD_RUN_URL not set"); return 0
    emb = embed(IMG.read_bytes())
    payload={'embedding': emb, 'purchase':'Milk'}
    t0=time.time()
    r = requests.post(CLOUD_RUN_URL+DETECT_PATH, json=payload, timeout=10)
    ms=round((time.time()-t0)*1000)
    try:
        server_ms = r.json().get('duration_ms')
    except Exception:
        server_ms = None
    write(f"status={r.status_code} latency_ms={ms} server_ms={server_ms}")
    # 成功條件：<1s 且 2xx
    return 0 if (ms<1000 and 200<=r.status_code<300) else 3

if __name__ == '__main__': raise SystemExit(main())
