import base64, os, numpy as np
from google.cloud import firestore

def _np_to_b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype('float32').tobytes()).decode('ascii')

def _b64_to_np(s: str) -> np.ndarray:
    b = base64.b64decode(s.encode('ascii'))
    return np.frombuffer(b, dtype='float32')

def get_db(project: str):
    dbid = os.environ.get('FIRESTORE_DATABASE_ID', '(default)')  # 讀 DB ID
    return firestore.Client(project=project, database=dbid)

def ensure_enrolled(db, collection: str, visitor_id: str, enc: np.ndarray):
    doc = db.collection(collection).document(visitor_id).get()
    if doc.exists: 
        return False
    db.collection(collection).document(visitor_id).set({
        'enc_b64': _np_to_b64(enc),
        'dim': int(enc.shape[0]),
        'source': 'ci-enroll'
    })
    return True

def load_embed(db, collection: str, visitor_id: str):
    doc = db.collection(collection).document(visitor_id).get()
    if not doc.exists: 
        return None
    data = doc.to_dict()
    return _b64_to_np(data['enc_b64'])
