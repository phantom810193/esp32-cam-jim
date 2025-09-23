"""Helper functions for the SQLite visitor database."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

DEFAULT_DB_PATH = Path("users.db")
DEFAULT_SQL_PATH = Path(__file__).resolve().parent / "data.sql"


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


def initialize_database(
    db_path: Path | str = DEFAULT_DB_PATH,
    sql_path: Path | str = DEFAULT_SQL_PATH,
    *,
    reset: bool = False,
) -> None:
    """Create the database from ``data.sql`` if it does not exist."""

    db_path = Path(db_path)
    sql_path = Path(sql_path)

    if reset and db_path.exists():
        db_path.unlink()

    if db_path.exists():
        return

    script = sql_path.read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as connection:
        connection.executescript(script)
        connection.commit()


@contextmanager
def connect(db_path: Path | str = DEFAULT_DB_PATH):
    connection = sqlite3.connect(Path(db_path))
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.commit()
        connection.close()


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


def create_user_record(
    connection: sqlite3.Connection,
    *,
    user_id: str,
    created_at: str,
    purchase: str,
    spend: float = 100.0,
    source: str = "enroll",
) -> None:
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


def record_visit(
    connection: sqlite3.Connection,
    *,
    user_id: str,
    visit_time: str,
    purchase: str,
    spend: float = 100.0,
    source: str = "detect_face",
) -> None:
    connection.execute(
        "UPDATE users SET last_visit = ?, last_purchase = ?, total_spend = total_spend + ?"
        " WHERE id = ?",
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


def get_user(connection: sqlite3.Connection, user_id: str) -> Optional[UserRecord]:
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


def fetch_all_users(connection: sqlite3.Connection) -> List[UserRecord]:
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


def fetch_recent_visits(
    connection: sqlite3.Connection,
    *,
    user_id: str | None = None,
    limit: int | None = None,
) -> List[VisitRecord]:
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


def to_rows(records: Iterable[UserRecord]) -> List[tuple]:
    return [
        (record.id, record.created_at, record.last_visit, record.last_purchase, record.total_spend)
        for record in records
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
