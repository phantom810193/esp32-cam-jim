import os, time, base64, logging, sys
from pathlib import Path

logging.basicConfig(filename='gcp_test.log', level=logging.INFO, format='%(asctime)s %(message)s')

def log_to(path, msg):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

def test_vision():
    from google.cloud import vision
    # 優先用你的樣本；沒有就用 1x1 PNG 測連通性
    img_bytes = None
    for p in [Path('samples/faces/personA/1.jpg'), Path('samples/face.jpg')]:
        if p.exists():
            img_bytes = p.read_bytes(); break
    if img_bytes is None:
        img_bytes = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2x0nQAAAAASUVORK5CYII=')
    t0 = time.time()
    client = vision.ImageAnnotatorClient()
    _ = client.face_detection(image=vision.Image(content=img_bytes))
    ms = round((time.time()-t0)*1000)
    log_to('vision_test.log', f"ok latency_ms={ms}")
    return True

def test_firestore():
    from google.cloud import firestore
    proj = os.environ.get('GCP_PROJECT')
    col = os.environ.get('FIRESTORE_COLLECTION', 'visitors')
    dbid = os.environ.get('FIRESTORE_DATABASE_ID', '(default)')
    if not proj:
        log_to('firestore_test.log', "missing GCP_PROJECT")
        return False
    db = firestore.Client(project=proj, database=dbid)
    doc_id = f"ci-{int(time.time())}"
    t0 = time.time()
    db.collection(col).document(doc_id).set({
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'source': 'github-actions',
        'note': 'connectivity-test'
    })
    ms = round((time.time()-t0)*1000)
    log_to('firestore_test.log', f"ok project={proj} database={dbid} collection={col} doc={doc_id} latency_ms={ms}")
    return True

def test_cloud_run():
    import requests
    url = os.environ.get('CLOUD_RUN_URL', '').rstrip('/')
    if not url:
        log_to('cloudrun_test.log', "skipped (set CLOUD_RUN_URL to enable)")
        return True
    path = os.environ.get('CLOUD_RUN_HEALTH_PATH', '/')
    # 先試未驗證；401/403 則帶 ID Token 再試
    t0 = time.time()
    r = requests.get(url + path, timeout=10)
    ms = round((time.time()-t0)*1000)
    if r.status_code in (401, 403):
        try:
            from google.oauth2 import service_account
            from google.auth.transport.requests import AuthorizedSession
            creds = service_account.IDTokenCredentials.from_service_account_file(
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'],
                target_audience=url
            )
            authed = AuthorizedSession(creds)
            t0 = time.time()
            r = authed.get(url + path, timeout=10)
            ms = round((time.time()-t0)*1000)
        except Exception as e:
            log_to('cloudrun_test.log', f"auth_error: {e}")
            return False
    log_to('cloudrun_test.log', f"status={r.status_code} latency_ms={ms}")
    return 200 <= r.status_code < 400

if __name__ == "__main__":
    ok_v = ok_f = ok_r = False
    try: ok_v = test_vision()
    except Exception as e: log_to('vision_test.log', f"error: {e}")
    try: ok_f = test_firestore()
    except Exception as e: log_to('firestore_test.log', f"error: {e}")
    try: ok_r = test_cloud_run()
    except Exception as e: log_to('cloudrun_test.log', f"error: {e}")

    summary = f"vision={ok_v} firestore={ok_f} cloudrun={ok_r}"
    logging.info(summary); print(summary)
    if not ok_v or not ok_f or (os.environ.get('CLOUD_RUN_URL') and not ok_r):
        sys.exit(1)
