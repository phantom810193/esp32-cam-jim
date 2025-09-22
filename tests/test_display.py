from pathlib import Path

from display import render_messages


def test_render_messages_generates_log(tmp_path: Path) -> None:
    log_path = tmp_path / "text_test.log"
    messages = [f"Message {i}" for i in range(5)]
    rendered = render_messages(messages, log_path=log_path, enable_print=False)

    assert len(rendered) == 5
    payload = log_path.read_text(encoding="utf-8")
    assert "Message 4" in payload
