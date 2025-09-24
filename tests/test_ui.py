from pathlib import Path

from admin_dashboard import build_dashboard
from database_utils import initialize_database
from promo_ui import render_promotions


def test_build_dashboard_creates_html(tmp_path: Path) -> None:
    db_path = tmp_path / "users.db"
    initialize_database(db_path, "data.sql", reset=True)
    output_path = tmp_path / "admin_dashboard.html"
    build_dashboard(db_path=db_path, output_path=output_path)

    html = output_path.read_text(encoding="utf-8")
    assert "賣場管理員後台" in html
    assert "ID-abc123" in html


def test_render_promotions_outputs_ascii(tmp_path: Path) -> None:
    output_path = tmp_path / "promo_display.log"
    render_promotions(["ID-abc123: 牛奶9折"], output_path=output_path)

    text = output_path.read_text(encoding="utf-8")
    assert "牛奶9折" in text
    json_payload = output_path.with_suffix(output_path.suffix + ".json").read_text(encoding="utf-8")
    assert "\"count\": 1" in json_payload
