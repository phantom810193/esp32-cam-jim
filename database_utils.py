# database_utils.py
from __future__ import annotations
import os, sqlite3
from dataclasses import dataclass
from typing import List, Optional, Iterable, Any

# ==== 資料模型（兩個後端共用） ====
@dataclass
class UserRecord:
  user_id: str
  created_at: str
  last_purchase: str
  last_visit_at: Optional[str] = None
  visits_count: int = 0

# ==== 後端切換邏輯 ====
def _use_firestore() -> bool:
  # 優先 DB_BACKEND=firestore；或有設定 FIRESTORE_DATABASE_ID / FIRESTORE_COLLECTION 也視為 Firestore
  return os.getenv("DB_BACKEND", "").lower() == "firestore" or \
         os.getenv("FIRESTORE_DATABASE_ID") or os.getenv("FIRESTORE_COLLECTION")

# ---- Firestore 連線包裝 ----
class FirestoreConn:
  def __init__(self):
    from google.cloud import firestore
    project = os.getenv("GCP_PROJECT")
    database_id = os.getenv("FIRESTORE_DATABASE_ID", "(default)")
    self.client = firestore.Client(project=project, database=database_id)
    self.users_name = os.getenv("FIRESTORE_COLLECTION", "users")
    self.events_name = os.getenv("FIRESTORE_EVENTS_COLLECTION", "visits")
  def __enter__(self): return self
  def __exit__(self, exc_type, exc, tb): return False
  # helpers
  def users(self):
    return self.client.collection(self.users_name)
  def events(self):
    return self.client.collection(self.events_name)

# ---------------- 對外 API（相容 api.py 既有呼叫） ----------------
def connect(db_path: str | os.PathLike[str]) -> Any:
  """api.py 會 `with connect(...) as conn:`；這裡回傳 FirestoreConn 或 sqlite3.Connection"""
  if _use_firestore():
    return FirestoreConn()
  return sqlite3.connect(str(db_path))

def initialize_database(db_path: str | os.PathLike[str], sql_seed_path: str | os.PathLike[str], reset: bool=False) -> None:
  """Firestore：不需要初始化；SQLite：建立表並可選擇清庫/灌樣本。"""
  if _use_firestore():
    # Firestore 無需 schema；reset 這裡不幫你刪庫，以免誤刪
    return
  # SQLite 初始化
  db = sqlite3.connect(str(db_path))
  cur = db.cursor()
  cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
      user_id TEXT PRIMARY KEY,
      created_at TEXT,
      last_purchase TEXT,
      last_visit_at TEXT,
      visits_count INTEGER DEFAULT 0
    )
  """)
  cur.execute("""
    CREATE TABLE IF NOT EXISTS visits(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id TEXT,
      visit_time TEXT,
      purchase TEXT
    )
  """)
  if reset:
    cur.execute("DELETE FROM visits")
    cur.execute("DELETE FROM users")
  db.commit(); db.close()

# ---- 查詢單一使用者 ----
def get_user(conn: Any, user_id: str) -> Optional[UserRecord]:
  if isinstance(conn, FirestoreConn):
    doc = conn.users().document(user_id).get()
    if not doc.exists: return None
    d = doc.to_dict() or {}
    return UserRecord(
      user_id=user_id,
      created_at=d.get("created_at",""),
      last_purchase=d.get("last_purchase",""),
      last_visit_at=d.get("last_visit_at"),
      visits_count=int(d.get("visits_count",0))
    )
  # SQLite
  cur = conn.cursor()
  cur.execute("SELECT user_id, created_at, last_purchase, last_visit_at, visits_count FROM users WHERE user_id=?",(user_id,))
  row = cur.fetchone()
  if not row: return None
  return UserRecord(*row)

# ---- 建立新用戶 ----
def create_user_record(conn: Any, user_id: str, created_at: str, purchase: str) -> None:
  if isinstance(conn, FirestoreConn):
    ref = conn.users().document(user_id)
    ref.set({
      "created_at": created_at,
      "last_purchase": purchase,
      "last_visit_at": created_at,
      "visits_count": 1
    }, merge=True)
    # 同步寫一筆事件
    conn.events().add({"user_id": user_id, "visit_time": created_at, "purchase": purchase})
    return
  # SQLite
  cur = conn.cursor()
  cur.execute("INSERT OR REPLACE INTO users(user_id, created_at, last_purchase, last_visit_at, visits_count) VALUES(?,?,?,?,?)",
              (user_id, created_at, purchase, created_at, 1))
  cur.execute("INSERT INTO visits(user_id, visit_time, purchase) VALUES(?,?,?)", (user_id, created_at, purchase))
  conn.commit()

# ---- 記錄返店/消費 ----
def record_visit(conn: Any, user_id: str, visit_time: str, purchase: str) -> None:
  if isinstance(conn, FirestoreConn):
    # 事件表新增一筆
    conn.events().add({"user_id": user_id, "visit_time": visit_time, "purchase": purchase})
    # 使用者表更新
    user_ref = conn.users().document(user_id)
    user_ref.set({
      "last_purchase": purchase,
      "last_visit_at": visit_time,
      "visits_count": firestore_increment(1)
    }, merge=True)
    return
  # SQLite
  cur = conn.cursor()
  cur.execute("INSERT INTO visits(user_id, visit_time, purchase) VALUES(?,?,?)", (user_id, visit_time, purchase))
  cur.execute("UPDATE users SET last_purchase=?, last_visit_at=?, visits_count=COALESCE(visits_count,0)+1 WHERE user_id=?",
              (purchase, visit_time, user_id))
  conn.commit()

# Firestore 的遞增 helper
def firestore_increment(n: int):
  try:
    from google.cloud.firestore_v1 import Increment
    return Increment(n)
  except Exception:
    # 舊版客戶端沒有 Increment 時，退而求其次：先讀後寫（但在上面我們用了 set(merge=True)；此處僅作介面相容）
    return n

# ---- 列出全部使用者（/admin/users 用） ----
def fetch_all_users(conn: Any) -> List[UserRecord]:
  if isinstance(conn, FirestoreConn):
    docs = conn.users().stream()
    out: List[UserRecord] = []
    for doc in docs:
      d = doc.to_dict() or {}
      out.append(UserRecord(
        user_id=doc.id,
        created_at=d.get("created_at",""),
        last_purchase=d.get("last_purchase",""),
        last_visit_at=d.get("last_visit_at"),
        visits_count=int(d.get("visits_count",0))
      ))
    return sorted(out, key=lambda r: (r.last_visit_at or "", r.user_id), reverse=True)
  # SQLite
  cur = conn.cursor()
  cur.execute("SELECT user_id, created_at, last_purchase, last_visit_at, visits_count FROM users ORDER BY COALESCE(last_visit_at, created_at) DESC")
  rows = cur.fetchall()
  return [UserRecord(*row) for row in rows]
