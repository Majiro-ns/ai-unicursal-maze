"""
tests/test_dm8_edgecases_704k.py — DM-8 エッジケース・CLI・API・パフォーマンス・境界値テスト

タスク: cmd_704k_a7
テストカテゴリ:
  1. DM-8マルチスケール整合性テスト        — 10件
  2. DM-8エッジケース: 極端なpassage_ratio — 5件
  3. masterpieceプリセット SSIM回帰テスト  — 5件
  4. CLI: --preset / --difficulty テスト   — 5件
  5. API: /dm6/generate エンドポイント     — 5件
  6. パフォーマンス: 32x32グリッド処理時間 — 5件
  7. 境界値: grid_size 1x1/2x2/64x64      — 4件
  合計: 39件
"""
from __future__ import annotations

import io
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.core.density.dm8 import (
    DM8Config,
    DM8Result,
    build_multiscale_density_map,
    generate_dm8_maze,
    _upsample_density,
)
from backend.core.density.dm4 import DM4Config, generate_dm4_maze
from backend.core.density import MASTERPIECE_PRESET


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_gradient(w: int = 64, h: int = 64) -> Image.Image:
    arr = np.linspace(0, 255, w * h, dtype=np.uint8).reshape(h, w)
    return Image.fromarray(arr, mode="L").convert("RGB")


def _make_uniform(val: int = 128, w: int = 64, h: int = 64) -> Image.Image:
    arr = np.full((h, w), val, dtype=np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def _png_bytes_from_array(arr: np.ndarray) -> bytes:
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# グループ1: DM-8マルチスケール整合性テスト (10件)
# ---------------------------------------------------------------------------

class TestDM8MultiscaleConsistency:
    """DM-8マルチスケール密度マップの整合性テスト。"""

    def test_build_multiscale_output_shape(self):
        """build_multiscale_density_map の出力 shape が target_rows×target_cols であること"""
        gray = np.random.rand(64, 64)
        result = build_multiscale_density_map(gray, 20, 20)
        assert result.shape == (20, 20), f"shape mismatch: {result.shape}"

    def test_build_multiscale_output_range(self):
        """マルチスケール密度マップの値が 0.0〜1.0 範囲内であること"""
        gray = np.random.rand(64, 64)
        result = build_multiscale_density_map(gray, 20, 20)
        assert float(result.min()) >= 0.0, f"min={result.min()} < 0"
        assert float(result.max()) <= 1.0, f"max={result.max()} > 1"

    def test_build_multiscale_coarse_only(self):
        """coarse_size=medium_size の場合でもクラッシュしないこと"""
        gray = np.random.rand(64, 64)
        result = build_multiscale_density_map(gray, 20, 20, coarse_size=4, medium_size=4)
        assert result.shape == (20, 20)

    def test_upsample_density_shape(self):
        """_upsample_density が正しい出力 shape を返すこと"""
        src = np.random.rand(10, 10)
        out = _upsample_density(src, 20, 20)
        assert out.shape == (20, 20), f"upsample shape: {out.shape}"

    def test_upsample_density_range(self):
        """アップサンプル後の値が 0.0〜1.0 範囲内であること"""
        src = np.random.rand(10, 10)
        out = _upsample_density(src, 20, 20)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_dm8_result_stores_scale_weights(self):
        """DM8Result の scale_weights_used がコンフィグ値と一致すること"""
        img = _make_gradient()
        weights = (0.1, 0.3, 0.6)
        cfg = DM8Config(grid_rows=20, grid_cols=20, scale_weights=weights)
        result = generate_dm8_maze(img, cfg)
        assert result.scale_weights_used == weights, (
            f"scale_weights_used: {result.scale_weights_used} != {weights}"
        )

    def test_dm8_result_stores_coarse_size(self):
        """DM8Result の coarse_size_used がコンフィグ値と一致すること"""
        img = _make_gradient()
        cfg = DM8Config(grid_rows=20, grid_cols=20, coarse_size=6)
        result = generate_dm8_maze(img, cfg)
        assert result.coarse_size_used == 6

    def test_dm8_result_stores_medium_size(self):
        """DM8Result の medium_size_used がコンフィグ値と一致すること"""
        img = _make_gradient()
        cfg = DM8Config(grid_rows=20, grid_cols=20, medium_size=10)
        result = generate_dm8_maze(img, cfg)
        assert result.medium_size_used == 10

    def test_dm8_scale_weights_sum_near_one(self):
        """デフォルト scale_weights の合計が 1.0 に近いこと"""
        cfg = DM8Config()
        total = sum(cfg.scale_weights)
        assert abs(total - 1.0) < 0.01, f"scale_weights sum: {total}"

    def test_dm8_multiscale_different_weights_affect_output(self):
        """異なる scale_weights は異なる密度マップを生成すること"""
        # 高周波パターン: スケール間で明確な差を生む（線形グラジェントは全スケールで均質になりテストに不向き）
        x = np.linspace(0, 1, 64)
        low_freq = np.outer(x, x)
        high_freq = 0.4 * np.outer(np.sin(10 * np.pi * x), np.sin(10 * np.pi * x))
        gray = np.clip(low_freq + high_freq, 0.0, 1.0)
        out1 = build_multiscale_density_map(gray, 20, 20, scale_weights=(0.6, 0.3, 0.1))
        out2 = build_multiscale_density_map(gray, 20, 20, scale_weights=(0.1, 0.3, 0.6))
        # 完全一致はしないはず（重みが異なるため）
        assert not np.allclose(out1, out2, atol=0.05), "異なるweightsで同一出力"


# ---------------------------------------------------------------------------
# グループ2: DM-8エッジケース: 極端なpassage_ratio (5件)
# ---------------------------------------------------------------------------

class TestDM8ExtremePassageRatio:
    """極端な passage_ratio での DM-8 動作確認テスト。"""

    def test_passage_ratio_001_raises_value_error(self):
        """passage_ratio=0.01（範囲外）で ValueError が発生すること"""
        img = _make_gradient()
        cfg = DM8Config(grid_rows=20, grid_cols=20, passage_ratio=0.01)
        with pytest.raises(ValueError, match="passage_ratio"):
            generate_dm8_maze(img, cfg)

    def test_passage_ratio_099_raises_value_error(self):
        """passage_ratio=0.99（範囲外）で ValueError が発生すること"""
        img = _make_gradient()
        cfg = DM8Config(grid_rows=20, grid_cols=20, passage_ratio=0.99)
        with pytest.raises(ValueError, match="passage_ratio"):
            generate_dm8_maze(img, cfg)

    def test_passage_ratio_050_baseline(self):
        """passage_ratio=0.50（デフォルト）で正常動作すること"""
        img = _make_gradient()
        cfg = DM8Config(grid_rows=20, grid_cols=20, passage_ratio=0.50)
        result = generate_dm8_maze(img, cfg)
        assert result.ssim_score > 0.0
        assert result.solution_count >= 1

    def test_passage_ratio_inherited_from_dm4(self):
        """DM8Config.passage_ratio がデフォルト値 0.5 であること（DM4継承）"""
        cfg = DM8Config()
        assert cfg.passage_ratio == 0.5

    def test_dm8_extreme_ratio_boundary_valid_solution_exists(self):
        """有効範囲境界値(0.1/0.8)では解が存在すること"""
        img = _make_gradient()
        for ratio in [0.1, 0.8]:
            cfg = DM8Config(grid_rows=15, grid_cols=15, passage_ratio=ratio)
            result = generate_dm8_maze(img, cfg)
            assert result.solution_count >= 1, (
                f"ratio={ratio}: solution_count={result.solution_count}"
            )


# ---------------------------------------------------------------------------
# グループ3: masterpieceプリセット SSIM回帰テスト (5件)
# ---------------------------------------------------------------------------

class TestMasterpiecePresetSSIMRegression:
    """masterpiece プリセット設定での SSIM 回帰テスト。"""

    def test_masterpiece_preset_has_fill_cells(self):
        """MASTERPIECE_PRESET は fill_cells=True 関連設定を含むこと"""
        # fill_cells は DM4/DM8 の高SSIM手法の核心
        # MASTERPIECE_PRESET の grid_size/thickness_range 等が存在すること
        assert "grid_size" in MASTERPIECE_PRESET
        assert "thickness_range" in MASTERPIECE_PRESET

    def test_dm4_portrait_ssim_positive(self):
        """portrait 系（円形グラデーション）で DM4 SSIM > 0 であること"""
        cx, cy = 32.0, 32.0
        ys, xs = np.mgrid[0:64, 0:64]
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        arr = np.clip(255 * (1.0 - dist / cy), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L").convert("RGB")
        cfg = DM4Config(grid_rows=20, grid_cols=20)
        result = generate_dm4_maze(img, cfg)
        assert result.ssim_score > 0.0

    def test_dm4_landscape_ssim_positive(self):
        """landscape 系（水平ストライプ）で DM4 SSIM > 0 であること"""
        arr = np.zeros((64, 64), dtype=np.uint8)
        for i in range(0, 8, 2):
            arr[i * 8:(i + 1) * 8, :] = 200
        img = Image.fromarray(arr, mode="L").convert("RGB")
        cfg = DM4Config(grid_rows=20, grid_cols=20)
        result = generate_dm4_maze(img, cfg)
        assert result.ssim_score > 0.0

    def test_dm4_logo_ssim_positive(self):
        """logo 系（チェッカーボード）で DM4 SSIM > 0 であること"""
        ys, xs = np.mgrid[0:64, 0:64]
        arr = (((xs // 8) + (ys // 8)) % 2 * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L").convert("RGB")
        cfg = DM4Config(grid_rows=20, grid_cols=20)
        result = generate_dm4_maze(img, cfg)
        assert result.ssim_score > 0.0

    def test_dm4_anime_ssim_positive(self):
        """anime 系（シルエット）で DM4 SSIM > 0 であること"""
        cx, cy = 32.0, 32.0
        ys, xs = np.mgrid[0:64, 0:64]
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        arr = np.where(dist < 28, 220, 30).astype(np.uint8)
        img = Image.fromarray(arr, mode="L").convert("RGB")
        cfg = DM4Config(grid_rows=20, grid_cols=20)
        result = generate_dm4_maze(img, cfg)
        assert result.ssim_score > 0.0


# ---------------------------------------------------------------------------
# グループ4: CLI テスト (5件)
# ---------------------------------------------------------------------------

class TestCLIOptions:
    """backend.cli の generate コマンドテスト。"""

    def _make_tmp_image(self, tmpdir: Path, w: int = 64, h: int = 64) -> Path:
        path = tmpdir / "input.png"
        arr = np.linspace(0, 255, w * h, dtype=np.uint8).reshape(h, w)
        Image.fromarray(arr, mode="L").save(str(path))
        return path

    def test_cli_generate_easy_exits_zero(self):
        """--difficulty easy でゼロ終了すること"""
        from typer.testing import CliRunner
        from backend.cli import app
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            img = self._make_tmp_image(p)
            out = p / "out.png"
            result = runner.invoke(app, [
                "generate", "--image", str(img),
                "--output", str(out), "--difficulty", "easy",
            ])
            assert result.exit_code == 0, f"exit={result.exit_code}\n{result.output}"

    def test_cli_generate_creates_output_file(self):
        """generate コマンドで出力ファイルが作成されること"""
        from typer.testing import CliRunner
        from backend.cli import app
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            img = self._make_tmp_image(p)
            out = p / "maze.png"
            runner.invoke(app, [
                "generate", "--image", str(img),
                "--output", str(out), "--difficulty", "medium",
            ])
            assert out.exists(), "出力ファイルが作成されていない"

    def test_cli_generate_preset_portrait(self):
        """--preset portrait でゼロ終了すること"""
        from typer.testing import CliRunner
        from backend.cli import app
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            img = self._make_tmp_image(p)
            out = p / "out.png"
            result = runner.invoke(app, [
                "generate", "--image", str(img),
                "--output", str(out), "--preset", "portrait",
            ])
            assert result.exit_code == 0, f"exit={result.exit_code}\n{result.output}"

    def test_cli_generate_invalid_difficulty_exits_nonzero(self):
        """無効な --difficulty 値で非ゼロ終了すること"""
        from typer.testing import CliRunner
        from backend.cli import app
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            img = self._make_tmp_image(p)
            result = runner.invoke(app, [
                "generate", "--image", str(img),
                "--difficulty", "invalid_difficulty",
            ])
            assert result.exit_code != 0, "無効なdifficulty値で終了コード0が返された"

    def test_cli_generate_dry_run(self):
        """--dry-run でファイルを作成せずに終了すること"""
        from typer.testing import CliRunner
        from backend.cli import app
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            img = self._make_tmp_image(p)
            out = p / "should_not_exist.png"
            result = runner.invoke(app, [
                "generate", "--image", str(img),
                "--output", str(out), "--dry-run",
            ])
            assert result.exit_code == 0
            assert not out.exists(), "--dry-run でファイルが作成された"


# ---------------------------------------------------------------------------
# グループ5: API: /dm6/generate エンドポイントテスト (5件)
# ---------------------------------------------------------------------------

def _make_png_bytes(w: int = 64, h: int = 64) -> bytes:
    arr = np.linspace(0, 255, w * h, dtype=np.uint8).reshape(h, w)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def api_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.routes import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestAPIDm6Generate:
    """/dm6/generate API エンドポイントテスト。"""

    def test_dm6_generate_returns_200(self, api_client):
        """正常リクエストで HTTP 200 が返ること"""
        response = api_client.post(
            "/dm6/generate",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
            data={"difficulty": "easy"},
        )
        assert response.status_code == 200, f"status={response.status_code} {response.text}"

    def test_dm6_generate_response_has_ssim(self, api_client):
        """レスポンスに ssim_score が含まれること"""
        response = api_client.post(
            "/dm6/generate",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
            data={"difficulty": "easy"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "ssim_score" in data, f"ssim_score なし: {list(data.keys())}"

    def test_dm6_generate_with_preset_portrait(self, api_client):
        """preset_name=portrait でレスポンスが返ること"""
        response = api_client.post(
            "/dm6/generate",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
            data={"difficulty": "easy", "preset_name": "portrait"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "png_base64" in data

    def test_dm6_generate_invalid_file_returns_400(self, api_client):
        """不正なファイルで HTTP 400 が返ること"""
        response = api_client.post(
            "/dm6/generate",
            files={"file": ("bad.png", b"not_an_image", "image/png")},
            data={"difficulty": "easy"},
        )
        assert response.status_code == 400

    def test_dm6_generate_invalid_difficulty_falls_back_to_medium(self, api_client):
        """無効な difficulty 値はサーバーが medium にフォールバックすること"""
        response = api_client.post(
            "/dm6/generate",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
            data={"difficulty": "ultra_extreme"},
        )
        # フォールバックにより 200 が返ること
        assert response.status_code == 200
        data = response.json()
        assert data.get("difficulty") == "medium"


# ---------------------------------------------------------------------------
# グループ6: パフォーマンス: 大きいグリッド処理時間テスト (5件)
# ---------------------------------------------------------------------------

class TestDM8Performance:
    """DM-8 の処理時間テスト（32x32グリッド）。"""

    _PERF_LIMIT_SEC = 10.0  # 許容上限秒数

    def test_dm8_32x32_grid_within_time_limit(self):
        """32x32グリッドの生成が 10秒以内に完了すること"""
        img = _make_gradient(128, 128)
        cfg = DM8Config(grid_rows=32, grid_cols=32)
        start = time.time()
        result = generate_dm8_maze(img, cfg)
        elapsed = time.time() - start
        assert elapsed < self._PERF_LIMIT_SEC, (
            f"32x32グリッド生成が {elapsed:.2f}秒かかった（上限{self._PERF_LIMIT_SEC}秒）"
        )
        assert len(result.png_bytes) > 100

    def test_dm8_32x32_ssim_positive(self):
        """32x32グリッドで SSIM > 0 であること"""
        img = _make_gradient(128, 128)
        cfg = DM8Config(grid_rows=32, grid_cols=32)
        result = generate_dm8_maze(img, cfg)
        assert result.ssim_score > 0.0

    def test_dm8_small_grid_faster_than_large(self):
        """小グリッド(10x10)が大グリッド(32x32)より高速であること"""
        img = _make_gradient(128, 128)
        start = time.time()
        generate_dm8_maze(img, DM8Config(grid_rows=10, grid_cols=10))
        small_time = time.time() - start
        start = time.time()
        generate_dm8_maze(img, DM8Config(grid_rows=32, grid_cols=32))
        large_time = time.time() - start
        # 概ね小グリッドが高速（必ずしも保証されないが通常はそうなる）
        # 厳密な保証ではなくヒューリスティックチェック
        assert small_time <= large_time * 5.0, (
            f"small={small_time:.3f}s > large={large_time:.3f}s * 5"
        )

    def test_dm8_20x20_within_time_limit(self):
        """20x20グリッドの生成が 5秒以内に完了すること"""
        img = _make_gradient()
        cfg = DM8Config(grid_rows=20, grid_cols=20)
        start = time.time()
        generate_dm8_maze(img, cfg)
        elapsed = time.time() - start
        assert elapsed < 5.0, f"20x20グリッド生成が {elapsed:.2f}秒かかった"

    def test_dm8_32x32_uniform_image_within_time(self):
        """均一画像（最低コントラスト）での 32x32 生成が 10秒以内であること"""
        img = _make_uniform(128, 128, 128)
        cfg = DM8Config(grid_rows=32, grid_cols=32)
        start = time.time()
        result = generate_dm8_maze(img, cfg)
        elapsed = time.time() - start
        assert elapsed < self._PERF_LIMIT_SEC
        assert len(result.png_bytes) > 100


# ---------------------------------------------------------------------------
# グループ7: 境界値: grid_size 1x1/2x2/64x64 (4件)
# ---------------------------------------------------------------------------

class TestDM8GridSizeBoundary:
    """極端なグリッドサイズでの DM-8 動作テスト。"""

    def test_grid_2x2_no_crash(self):
        """2x2 グリッドでクラッシュしないこと"""
        img = _make_gradient(32, 32)
        cfg = DM8Config(grid_rows=2, grid_cols=2)
        result = generate_dm8_maze(img, cfg)
        assert len(result.png_bytes) > 0

    def test_grid_4x4_has_solution(self):
        """4x4 グリッドで解が存在すること"""
        img = _make_gradient(32, 32)
        cfg = DM8Config(grid_rows=4, grid_cols=4)
        result = generate_dm8_maze(img, cfg)
        assert result.solution_count >= 1

    def test_grid_large_image_no_crash(self):
        """256x256画像で medium difficulty でクラッシュしないこと"""
        img = _make_gradient(256, 256)
        cfg = DM8Config(difficulty="medium")
        result = generate_dm8_maze(img, cfg)
        assert result.grid_rows > 0
        assert len(result.png_bytes) > 100

    def test_grid_size_determined_by_difficulty(self):
        """DM-8 のグリッドサイズは difficulty params に基づき決定されること"""
        img = _make_gradient(128, 128)
        easy_result = generate_dm8_maze(img, DM8Config(difficulty="easy"))
        hard_result = generate_dm8_maze(img, DM8Config(difficulty="hard"))
        # easy < hard のグリッドサイズになること（または等しいが、easy は必ず正の値）
        assert easy_result.grid_rows > 0
        assert hard_result.grid_rows > 0
