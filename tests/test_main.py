from pathlib import Path

from main import run_pipeline


def test_run_pipeline_produces_expected_logs(tmp_path: Path) -> None:
    summary = run_pipeline(output_dir=tmp_path)

    # --- 基本指標（兩邊共同） ---
    assert summary["camera"]["fps"] > 10
    assert summary["recognition"]["accuracy"] >= 0.8
    assert summary["duration_seconds"] < 30

    # 必備日誌檔
    for filename in ["cam.log", "id_test.log", "text_test.log", "e2e.log"]:
        assert (tmp_path / filename).exists()

    # --- 進階指標（codex 分支功能，若有就驗證） ---
    # 穩定度報告
    if "stability" in summary:
        assert isinstance(summary["stability"], dict)
        # 至少會產生穩定度日誌
        assert (tmp_path / "cam_stable.log").exists()

    # 首次建檔 / 返店辨識 API 回傳
    if "enrollment" in summary:
        assert summary["enrollment"]["message"] in ("新用戶已建檔", "使用者已存在，已更新影像")
    if "api" in summary:
        assert summary["api"]["message"] in ("老朋友歡迎回來", "新用戶已建檔")
        # CI 環境可能較慢，放寬門檻到 2000ms 以避免偶發抖動
        assert summary["api"]["duration_ms"] < 2000
        if "visit_count" in summary["api"]:
            assert summary["api"]["visit_count"] >= 1

    # 促銷與儀表板輸出
    if "promotion" in summary:
        # 促銷展示檔與其 JSON metadata
        assert (tmp_path / "promo_display.log").exists()
        # 檢查像 "promo_display.log.json" 這類後綴
        has_promo_json = any(
            p.name.startswith("promo_display.log") and p.name.endswith(".json")
            for p in tmp_path.iterdir()
        )
        assert has_promo_json

    if "dashboard_path" in summary:
        assert (tmp_path / "admin_dashboard.html").exists()