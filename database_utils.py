"""Helper functions for the visitor database (SQLite or Firestore)."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Any

# ==== 路徑預設 ====
DEFAULT_DB_PATH = Path("users.db")
DEFAULT_SQL_PATH = Path(__file__).resolve().parent / "data.sql"

# ==== 資料模型（沿用 codex 版欄位，對 Firestore 做相容轉換） ====
@dataclass
class UserRecord:
    id: str
    created_at: str
    last_visit: str
    last_purchase: str
    total_spend: float


@dataclass
class VisitRecord:
    id: int
    user_id: str
    visit_time: str
    purchase: str
    spend: float
    source: str


# ==== 後端判斷 ====
def _use_firestore() -> bool:
    return (
        os.getenv("DB_BACKEND", "").lower() == "firestore"
        or bool(os.getenv("FIRESTORE_DATABASE_ID"))
        or bool(os.getenv("FIRESTORE_COLLECTION"))
    )


# ---- Firestore 連線包裝 ----
class FirestoreConn:
    def __init__(self):
        from google.cloud import firestore  # 延後載入，避免未用時出錯
        project = os.getenv("GCP_PROJECT")
        database_id = os.getenv("FIRESTORE_DATABASE_ID", "(default)")
        self.client = firestore.Client(project=project, database=database_id)
        self.users_name = os.getenv("FIRESTORE_COLLECTION", "users")
        self.events_name = os.getenv("FIRESTORE_EVENTS_COLLECTION", "visits")

    def __enter__(self) -> "FirestoreConn":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False  # 不吃掉例外

    # helpers
    def users(self):
        return self.client.collection(self.users_name)

    def events(self):
        return self.client.collection(self.events_name)


# Firestore 的遞增 helper
def firestore_increment(n: int):
    try:
        from google.cloud.firestore_v1 import Increment
        return Increment(n)
    except Exception:
        # 沒有 Increment 類別時，呼叫端仍會 set(merge=True)，此回傳僅為介面相容
        return n


# ---------------- 對外 API ----------------
def initialize_database(
    db_path: Path | str = DEFAULT_DB_PATH,
    sql_path: Path | str = DEFAULT_SQL_PATH,
    *,
    reset: bool = False,
) -> None:
    """
    SQLite：從 data.sql 初始化資料庫（若不存在）；支援 reset=True 先刪檔重建。
    Firestore：無需初始化（no-op）。
    """
    if _use_firestore():
        return

    db_path = Path(db_path)
    sql_path = Path(sql_path)

    if reset and db_path.exists():
        db_path.unlink(missing_ok=True)

    if db_path.exists():
        return

    script = ""
    if sql_path.exists():
        script = sql_path.read_text(encoding="utf-8")

    with sqlite3.connect(db_path) as connection:
        if script.strip():
            connection.executescript(script)
        else:
            # 後備：建立與 codex 版相容的最小 schema
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users(
                    id TEXT PRIMARY KEY,
                    created_at TEXT,
                    last_visit TEXT,
                    last_purchase TEXT,
                    total_spend REAL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS visits(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    visit_time TEXT,
                    purchase TEXT,
                    spend REAL DEFAULT 0,
                    source TEXT
                );
                """
            )
        connection.commit()


@contextmanager
def connect(db_path: Path | str = DEFAULT_DB_PATH):
    """
    提供 with connect(...) as conn: 使用
    - Firestore: 回傳 FirestoreConn
    - SQLite: 回傳 sqlite3.Connection（row_factory=sqlite3.Row）
    """
    if _use_firestore():
        conn = FirestoreConn()
        yield conn
    else:
        connection = sqlite3.connect(Path(db_path))
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.commit()
            connection.close()


# ---- 內部工具（SQLite）----
def _record_visit_row(
    connection: sqlite3.Connection,
    *,
    user_id: str,
    visit_time: str,
    purchase: str,
    spend: float,
    source: str,
) -> None:
    connection.execute(
        "INSERT INTO visits(user_id, visit_time, purchase, spend, source) VALUES (?, ?, ?, ?, ?)",
        (user_id, visit_time, purchase, float(spend), source),
    )


# ---- 建立新用戶 ----
def create_user_record(
    connection: Any,
    *,
    user_id: str,
    created_at: str,
    purchase: str,
    spend: float = 100.0,
    source: str = "enroll",
) -> None:
    if isinstance(connection, FirestoreConn):
        # users
        connection.users().document(user_id).set(
            {
                "created_at": created_at,
                "last_purchase": purchase,
                "last_visit_at": created_at,
                "visits_count": firestore_increment(1),
                "total_spend": firestore_increment(float(spend)),
            },
            merge=True,
        )
        # events
        connection.events().add(
            {
                "user_id": user_id,
                "visit_time": created_at,
                "purchase": purchase,
                "spend": float(spend),
                "source": source,
            }
        )
        return

    # SQLite
    connection.execute(
        "INSERT INTO users(id, created_at, last_visit, last_purchase, total_spend)"
        " VALUES (?, ?, ?, ?, ?)",
        (user_id, created_at, created_at, purchase, float(spend)),
    )
    _record_visit_row(
        connection,
        user_id=user_id,
        visit_time=created_at,
        purchase=purchase,
        spend=spend,
        source=source,
    )


# ---- 記錄返店/消費 ----
def record_visit(
    connection: Any,
    *,
    user_id: str,
    visit_time: str,
    purchase: str,
    spend: float = 100.0,
    source: str = "detect_face",
) -> None:
    if isinstance(connection, FirestoreConn):
        # 事件
        connection.events().add(
            {
                "user_id": user_id,
                "visit_time": visit_time,
                "purchase": purchase,
                "spend": float(spend),
                "source": source,
            }
        )
        # 使用者彙總
        connection.users().document(user_id).set(
            {
                "last_purchase": purchase,
                "last_visit_at": visit_time,
                "visits_count": firestore_increment(1),
                "total_spend": firestore_increment(float(spend)),
            },
            merge=True,
        )
        return

    # SQLite
    connection.execute(
        "UPDATE users SET last_visit = ?, last_purchase = ?, total_spend = total_spend + ? WHERE id = ?",
        (visit_time, purchase, float(spend), user_id),
    )
    _record_visit_row(
        connection,
        user_id=user_id,
        visit_time=visit_time,
        purchase=purchase,
        spend=spend,
        source=source,
    )


# ---- 查單一使用者 ----
def get_user(connection: Any, user_id: str) -> Optional[UserRecord]:
    if isinstance(connection, FirestoreConn):
        doc = connection.users().document(user_id).get()
        if not doc.exists:
            return None
        d = doc.to_dict() or {}
        return UserRecord(
            id=user_id,
            created_at=str(d.get("created_at", "")),
            last_visit=str(d.get("last_visit_at", "")),
            last_purchase=str(d.get("last_purchase", "")),
            total_spend=float(d.get("total_spend", d.get("visits_count", 0.0)) or 0.0),
        )

    cursor = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    if row is None:
        return None
    return UserRecord(
        id=row["id"],
        created_at=row["created_at"],
        last_visit=row["last_visit"],
        last_purchase=row["last_purchase"],
        total_spend=float(row["total_spend"]),
    )


# ---- 取全部使用者（供 UI 顯示） ----
def fetch_all_users(connection: Any) -> List[UserRecord]:
    if isinstance(connection, FirestoreConn):
        docs = list(connection.users().stream())
        out: List[UserRecord] = []
        for doc in docs:
            d = doc.to_dict() or {}
            out.append(
                UserRecord(
                    id=str(doc.id),
                    created_at=str(d.get("created_at", "")),
                    last_visit=str(d.get("last_visit_at", "")),
                    last_purchase=str(d.get("last_purchase", "")),
                    total_spend=float(d.get("total_spend", d.get("visits_count", 0.0)) or 0.0),
                )
            )
        # 近訪優先，其次依 id
        return sorted(out, key=lambda r: (r.last_visit or "", r.id), reverse=True)

    cursor = connection.execute(
        "SELECT id, created_at, last_visit, last_purchase, total_spend FROM users ORDER BY created_at"
    )
    return [
        UserRecord(
            id=row["id"],
            created_at=row["created_at"],
            last_visit=row["last_visit"],
            last_purchase=row["last_purchase"],
            total_spend=float(row["total_spend"]),
        )
        for row in cursor.fetchall()
    ]


# ---- 取近期事件 ----
def fetch_recent_visits(
    connection: Any,
    *,
    user_id: str | None = None,
    limit: int | None = None,
) -> List[VisitRecord]:
    if isinstance(connection, FirestoreConn):
        q = connection.events().order_by("visit_time", direction="DESCENDING")
        if user_id:
            q = q.where("user_id", "==", user_id)
        docs = list(q.stream())
        items = [
            VisitRecord(
                id=i,  # Firestore 無 auto int id，改用索引
                user_id=str(d.get("user_id", "")),
                visit_time=str(d.get("visit_time", "")),
                purchase=str(d.get("purchase", "")),
                spend=float(d.get("spend", 0.0) or 0.0),
                source=str(d.get("source", "unknown")),
            )
            for i, d in enumerate((doc.to_dict() or {}) for doc in docs)
        ]
        return items[:limit] if limit else items

    query = "SELECT id, user_id, visit_time, purchase, spend, source FROM visits"
    params: Sequence[object] = ()
    if user_id:
        query += " WHERE user_id = ?"
        params = (user_id,)
    query += " ORDER BY visit_time DESC"
    if limit:
        query += " LIMIT ?"
        params = params + (limit,) if params else (limit,)
    cursor = connection.execute(query, params)
    return [
        VisitRecord(
            id=int(row["id"]),
            user_id=row["user_id"],
            visit_time=row["visit_time"],
            purchase=row["purchase"],
            spend=float(row["spend"]),
            source=row["source"],
        )
        for row in cursor.fetchall()
    ]


# ---- 輔助 ----
def to_rows(records: Iterable[UserRecord]) -> List[tuple]:
    return [
        (r.id, r.created_at, r.last_visit, r.last_purchase, r.total_spend)
        for r in records
    ]


__all__ = [
    "UserRecord",
    "VisitRecord",
    "initialize_database",
    "connect",
    "create_user_record",
    "record_visit",
    "get_user",
    "fetch_all_users",
    "fetch_recent_visits",
    "to_rows",
]