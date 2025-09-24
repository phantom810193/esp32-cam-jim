"""Generate terminal friendly promotion layouts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def render_promotions(
    promotions: Iterable[str],
    *,
    output_path: Path | str = "promo_display.log",
) -> Path:
    lines = ["╔════════════════════════════════╗"]
    for promotion in promotions:
        text = promotion.strip()
        lines.append(f"║ {text:<30} ║")
    lines.append("╚════════════════════════════════╝")

    payload = {"lines": lines, "count": len(lines) - 2}
    output_path = Path(output_path)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    (output_path.with_suffix(output_path.suffix + ".json")).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path


__all__ = ["render_promotions"]
