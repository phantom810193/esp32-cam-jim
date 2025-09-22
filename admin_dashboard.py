"""Render a lightweight HTML dashboard for store managers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from database_utils import UserRecord, connect, fetch_all_users

TEMPLATE = """
<!DOCTYPE html>
<html lang=\"zh-Hant\">
<head>
    <meta charset=\"utf-8\" />
    <title>ESP32 賣場管理員後台</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 2rem; background: #f4f4f4; }}
        h1 {{ color: #333333; }}
        table {{ border-collapse: collapse; width: 100%; background: #ffffff; }}
        th, td {{ border: 1px solid #dddddd; padding: 0.5rem; text-align: left; }}
        th {{ background: #0d47a1; color: #ffffff; }}
        tr:nth-child(even) {{ background: #f1f1f1; }}
    </style>
</head>
<body>
    <h1>賣場管理員後台</h1>
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>建檔時間</th>
                <th>最後訪問</th>
                <th>最近購買</th>
                <th>累積消費</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
</body>
</html>
"""


def _render_rows(records: Iterable[UserRecord]) -> str:
    cells = []
    for record in records:
        cells.append(
            """
            <tr>
                <td>{id}</td>
                <td>{created_at}</td>
                <td>{last_visit}</td>
                <td>{last_purchase}</td>
                <td>{total_spend:.2f}</td>
            </tr>
            """.format(
                id=record.id,
                created_at=record.created_at,
                last_visit=record.last_visit,
                last_purchase=record.last_purchase,
                total_spend=record.total_spend,
            )
        )
    return "\n".join(cells)


def build_dashboard(
    *,
    db_path: Path | str = "users.db",
    output_path: Path | str = "admin_dashboard.html",
) -> Path:
    output_path = Path(output_path)
    with connect(db_path) as connection:
        records = fetch_all_users(connection)
    html = TEMPLATE.format(rows=_render_rows(records))
    output_path.write_text(html, encoding="utf-8")
    return output_path


__all__ = ["build_dashboard"]
