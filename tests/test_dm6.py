"""
tests/test_dm6.py — DM-6 Bayesian最適化 + 難易度制御テスト（30件以上）

テストカテゴリ:
  1. 定数・ユーティリティ (6件)
  2. DM6Config デフォルト値 (5件)
  3. DM6Result デフォルト値 (4件)
  4. 難易度制御 generate_dm6_maze (8件)
  5. difficulty_score 変換 (5件)
  6. optuna最適化 optimize_for_image (4件)
  7. プリセット generate_preset (4件)
  8. CLI dry-run (4件)
  合計: 40件
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.core.density.dm6 import (
    DIFFICULTY_PARAMS,
    VALID_DIFFICULTIES,
    VALID_PRESETS,
    DM6Config,
    DM6Result,
    generate_dm6_maze,
    _score_to_difficulty,
    _difficulty_to_score,
)
from backend.core.density.dm6_optimizer import (
    CATEGORY_CONSTRAINTS,
    VALID_CATEGORIES,
    _levels_to_grades,
    optimize_for_image,
    generate_preset,
    load_preset,
    save_preset,
)
from backend.core.density.dm4 import DM4Config, DM4Result


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_gradient_image(width: int = 32, height: int = 32) -> Image.Image:
    """テスト用グラデーション画像（RGB）。"""
    arr = np.linspace(0, 255, width * height, dtype=np.uint8).reshape(height, width)
    rgb = np.stack([arr, arr, arr], axis=2)
    return Image.fromarray(rgb, mode="RGB")


def _make_bright_image(width: int = 32, height: int = 32) -> Image.Image:
    """全ピクセルが明るい画像（fill_cells=True で SSIM が高くなる）。"""
    arr = np.full((height, width), 240, dtype=np.uint8)
    rgb = np.stack([arr, arr, arr], axis=2)
    return Image.fromarray(rgb, mode="RGB")


# ===========================================================================
# 1. 定数・ユーティリティ (6件)
# ===========================================================================

class TestConstants:
    def test_difficulty_params_keys(self):
        """DIFFICULTY_PARAMS に easy/medium/hard/extreme が存在する。"""
        assert set(DIFFICULTY_PARAMS.keys()) == {"easy", "medium", "hard", "extreme"}

    def test_difficulty_params_easy_grid_size(self):
        """easy の grid_size は 6。"""
        assert DIFFICULTY_PARAMS["easy"]["grid_size"] == 6

    def test_difficulty_params_extreme_grid_size(self):
        """extreme の grid_size は 16。"""
        assert DIFFICULTY_PARAMS["extreme"]["grid_size"] == 16

    def test_difficulty_params_medium_extra_removal(self):
        """medium の extra_removal_rate は 0.15。"""
        assert DIFFICULTY_PARAMS["medium"]["extra_removal_rate"] == pytest.approx(0.15)

    def test_valid_presets(self):
        """VALID_PRESETS に portrait/landscape/logo/anime が含まれる。"""
        assert {"portrait", "landscape", "logo", "anime"} <= VALID_PRESETS

    def test_levels_to_grades_n2(self):
        """_levels_to_grades(2) → [0, 255]。"""
        assert _levels_to_grades(2) == [0, 255]

    def test_levels_to_grades_n4(self):
        """_levels_to_grades(4) → [0, 85, 170, 255]。"""
        grades = _levels_to_grades(4)
        assert grades[0] == 0
        assert grades[-1] == 255
        assert len(grades) == 4

    def test_levels_to_grades_n8_length(self):
        """_levels_to_grades(8) は 8 要素。"""
        assert len(_levels_to_grades(8)) == 8


# ===========================================================================
# 2. DM6Config デフォルト値 (5件)
# ===========================================================================

class TestDM6ConfigDefaults:
    def test_default_difficulty(self):
        """デフォルト difficulty は \"medium\"。"""
        cfg = DM6Config()
        assert cfg.difficulty == "medium"

    def test_default_difficulty_score_is_none(self):
        """デフォルト difficulty_score は None。"""
        cfg = DM6Config()
        assert cfg.difficulty_score is None

    def test_default_extra_removal_rate(self):
        """デフォルト extra_removal_rate は 0.15（medium 相当）。"""
        cfg = DM6Config()
        assert cfg.extra_removal_rate == pytest.approx(0.15)

    def test_default_preset_name_is_none(self):
        """デフォルト preset_name は None。"""
        cfg = DM6Config()
        assert cfg.preset_name is None

    def test_inherits_dm4config(self):
        """DM6Config は DM4Config のサブクラス。"""
        assert issubclass(DM6Config, DM4Config)


# ===========================================================================
# 3. DM6Result デフォルト値 (4件)
# ===========================================================================

class TestDM6ResultDefaults:
    def test_inherits_dm4result(self):
        """DM6Result は DM4Result のサブクラス。"""
        assert issubclass(DM6Result, DM4Result)

    def test_default_difficulty_in_result(self):
        """DM6Result デフォルト difficulty は \"medium\"。"""
        r = DM6Result(svg="", png_bytes=b"", entrance=0, exit_cell=1,
                      solution_path=[], grid_rows=10, grid_cols=10,
                      density_map=np.zeros((10, 10)), adj={}, edge_map={},
                      solution_count=1, clahe_clip_limit_used=2.0,
                      clahe_n_tiles_used=(4, 4), ssim_score=0.5, dark_coverage=0.5)
        assert r.difficulty == "medium"

    def test_default_difficulty_score_in_result(self):
        """DM6Result デフォルト difficulty_score は 0.375。"""
        r = DM6Result(svg="", png_bytes=b"", entrance=0, exit_cell=1,
                      solution_path=[], grid_rows=10, grid_cols=10,
                      density_map=np.zeros((10, 10)), adj={}, edge_map={},
                      solution_count=1, clahe_clip_limit_used=2.0,
                      clahe_n_tiles_used=(4, 4), ssim_score=0.5, dark_coverage=0.5)
        assert r.difficulty_score == pytest.approx(0.375)

    def test_result_has_ssim_score(self):
        """DM6Result に ssim_score フィールドが存在する。"""
        img = _make_gradient_image()
        result = generate_dm6_maze(img, DM6Config())
        assert hasattr(result, "ssim_score")
        assert isinstance(result.ssim_score, float)


# ===========================================================================
# 4. 難易度制御 generate_dm6_maze (8件)
# ===========================================================================

class TestGenerateDM6Maze:
    # dm4.py: grid_rows = min(config.grid_rows, max(gray.shape[0] // 4, 1))
    # 32×32 → max grid 8。64×64 → max grid 16。極端なサイズ依存を避けるため
    # 十分大きい画像（64×64）を使用する。

    def test_easy_grid_size_is_6(self):
        """difficulty=\"easy\" で grid_rows=6, grid_cols=6（32×32: max=8 ≥ 6）。"""
        img = _make_gradient_image(32, 32)
        result = generate_dm6_maze(img, DM6Config(difficulty="easy"))
        assert result.grid_rows == 6
        assert result.grid_cols == 6

    def test_medium_grid_size_is_10(self):
        """difficulty=\"medium\" で grid_rows=10, grid_cols=10（64×64: max=16 ≥ 10）。"""
        img = _make_gradient_image(64, 64)
        result = generate_dm6_maze(img, DM6Config(difficulty="medium"))
        assert result.grid_rows == 10
        assert result.grid_cols == 10

    def test_hard_grid_size_is_14(self):
        """difficulty=\"hard\" で grid_rows=14, grid_cols=14（64×64: max=16 ≥ 14）。"""
        img = _make_gradient_image(64, 64)
        result = generate_dm6_maze(img, DM6Config(difficulty="hard"))
        assert result.grid_rows == 14
        assert result.grid_cols == 14

    def test_extreme_grid_size_is_16(self):
        """difficulty=\"extreme\" で grid_rows=16, grid_cols=16（64×64: max=16 = 16）。"""
        img = _make_gradient_image(64, 64)
        result = generate_dm6_maze(img, DM6Config(difficulty="extreme"))
        assert result.grid_rows == 16
        assert result.grid_cols == 16

    def test_result_difficulty_label_preserved(self):
        """結果の difficulty フィールドが入力と一致する。"""
        img = _make_gradient_image()
        result = generate_dm6_maze(img, DM6Config(difficulty="hard"))
        assert result.difficulty == "hard"

    def test_result_extra_removal_rate_easy(self):
        """easy の extra_removal_rate は 0.40。"""
        img = _make_gradient_image()
        result = generate_dm6_maze(img, DM6Config(difficulty="easy"))
        assert result.extra_removal_rate == pytest.approx(0.40)

    def test_result_extra_removal_rate_extreme(self):
        """extreme の extra_removal_rate は 0.00。"""
        img = _make_gradient_image()
        result = generate_dm6_maze(img, DM6Config(difficulty="extreme"))
        assert result.extra_removal_rate == pytest.approx(0.00)

    def test_invalid_difficulty_raises_value_error(self):
        """無効な difficulty は ValueError を発生させる。"""
        img = _make_gradient_image()
        with pytest.raises(ValueError, match="Invalid difficulty"):
            generate_dm6_maze(img, DM6Config(difficulty="ultra"))

    def test_none_config_uses_defaults(self):
        """config=None でデフォルト設定が適用される（difficulty=medium）。"""
        img = _make_gradient_image(64, 64)
        result = generate_dm6_maze(img, None)
        assert result.difficulty == "medium"
        assert result.grid_rows == 10

    def test_solution_path_exists(self):
        """生成された迷路に解経路が存在する。"""
        img = _make_gradient_image()
        result = generate_dm6_maze(img, DM6Config(difficulty="easy"))
        assert len(result.solution_path) > 0

    def test_preset_name_preserved(self):
        """preset_name が結果に保持される。"""
        img = _make_gradient_image()
        result = generate_dm6_maze(img, DM6Config(difficulty="medium", preset_name="portrait"))
        assert result.preset_name == "portrait"

    def test_svg_output_is_string(self):
        """SVG 出力が文字列である。"""
        img = _make_gradient_image()
        result = generate_dm6_maze(img, DM6Config(difficulty="easy"))
        assert isinstance(result.svg, str)
        assert len(result.svg) > 0

    def test_png_bytes_output(self):
        """PNG 出力が bytes である。"""
        img = _make_gradient_image()
        result = generate_dm6_maze(img, DM6Config(difficulty="easy"))
        assert isinstance(result.png_bytes, bytes)
        assert len(result.png_bytes) > 0


# ===========================================================================
# 5. difficulty_score 変換 (5件)
# ===========================================================================

class TestDifficultyScoreConversion:
    def test_score_0_0_is_easy(self):
        """difficulty_score=0.0 → \"easy\"。"""
        assert _score_to_difficulty(0.0) == "easy"

    def test_score_1_0_is_extreme(self):
        """difficulty_score=1.0 → \"extreme\"。"""
        assert _score_to_difficulty(1.0) == "extreme"

    def test_score_0_5_is_hard(self):
        """difficulty_score=0.5 → \"hard\"（[0.50, 0.75) 区間）。"""
        assert _score_to_difficulty(0.5) == "hard"

    def test_score_0_3_is_medium(self):
        """difficulty_score=0.3 → \"medium\"（[0.25, 0.50) 区間）。"""
        assert _score_to_difficulty(0.3) == "medium"

    def test_difficulty_score_overrides_difficulty(self):
        """difficulty_score 指定時は difficulty フィールドより優先される。"""
        img = _make_gradient_image(64, 64)
        # difficulty_score=0.9 → "extreme"（grid_size=16）
        config = DM6Config(difficulty="easy", difficulty_score=0.9)
        result = generate_dm6_maze(img, config)
        assert result.difficulty == "extreme"
        assert result.grid_rows == 16

    def test_score_to_difficulty_clamps_below_0(self):
        """score < 0 は 0 にクランプされ \"easy\"。"""
        assert _score_to_difficulty(-1.0) == "easy"

    def test_score_to_difficulty_clamps_above_1(self):
        """score > 1 は 1 にクランプされ \"extreme\"。"""
        assert _score_to_difficulty(2.0) == "extreme"

    def test_difficulty_to_score_medium(self):
        """_difficulty_to_score(\"medium\") は 0.375。"""
        assert _difficulty_to_score("medium") == pytest.approx(0.375)


# ===========================================================================
# 6. optuna最適化 optimize_for_image (4件)
# ===========================================================================

class TestOptimizeForImage:
    def test_returns_four_keys(self):
        """optimize_for_image は category/n_trials/best_value/best_params の 4 キーを返す。"""
        img = _make_gradient_image()
        result = optimize_for_image(img, n_trials=10, category="portrait", seed=0)
        assert set(result.keys()) == {"category", "n_trials", "best_value", "best_params"}

    def test_best_value_is_non_negative(self):
        """best_value は 0 以上（正常終了）。"""
        img = _make_gradient_image()
        result = optimize_for_image(img, n_trials=10, category="portrait", seed=0)
        assert result["best_value"] >= 0.0

    def test_n_trials_matches(self):
        """n_trials が指定通りの試行数と一致する。"""
        img = _make_gradient_image()
        result = optimize_for_image(img, n_trials=10, category="portrait", seed=0)
        assert result["n_trials"] == 10

    def test_invalid_category_raises_value_error(self):
        """無効な category は ValueError を発生させる。"""
        img = _make_gradient_image()
        with pytest.raises(ValueError, match="Invalid category"):
            optimize_for_image(img, n_trials=5, category="unknown")

    def test_best_params_has_grid_size(self):
        """best_params に grid_size が含まれる。"""
        img = _make_gradient_image()
        result = optimize_for_image(img, n_trials=10, category="logo", seed=42)
        assert "grid_size" in result["best_params"]

    def test_logo_category_runs(self):
        """logo カテゴリで最適化が正常に完了する。"""
        img = _make_gradient_image()
        result = optimize_for_image(img, n_trials=10, category="logo", seed=1)
        assert result["category"] == "logo"


# ===========================================================================
# 7. プリセット generate_preset (4件)
# ===========================================================================

class TestGeneratePreset:
    def test_portrait_preset_has_four_keys(self):
        """\"portrait\" プリセットは 4 キーを含む。"""
        img = _make_gradient_image()
        preset = generate_preset("portrait", img, n_trials=5, seed=0)
        assert set(preset.keys()) == {"category", "best_params", "best_value", "n_trials"}

    def test_logo_preset_has_four_keys(self):
        """\"logo\" プリセットは 4 キーを含む。"""
        img = _make_gradient_image()
        preset = generate_preset("logo", img, n_trials=5, seed=0)
        assert set(preset.keys()) == {"category", "best_params", "best_value", "n_trials"}

    def test_preset_category_field_matches(self):
        """プリセットの category フィールドが入力と一致する。"""
        img = _make_gradient_image()
        preset = generate_preset("anime", img, n_trials=5, seed=0)
        assert preset["category"] == "anime"

    def test_preset_save_and_load(self):
        """save_preset / load_preset のラウンドトリップが成功する。"""
        img = _make_gradient_image()
        preset = generate_preset("landscape", img, n_trials=5, seed=0)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_preset(preset, path)
            loaded = load_preset(path)
            assert loaded["category"] == preset["category"]
            assert loaded["best_value"] == pytest.approx(preset["best_value"])
        finally:
            os.unlink(path)

    def test_preset_best_params_json_serializable(self):
        """プリセットの best_params が JSON シリアライズ可能。"""
        img = _make_gradient_image()
        preset = generate_preset("portrait", img, n_trials=5, seed=0)
        serialized = json.dumps(preset)
        assert isinstance(serialized, str)


# ===========================================================================
# 8. CLI dry-run (4件)
# ===========================================================================

class TestCLIDryRun:
    """typer CLI の --dry-run モード確認（typer.testing.CliRunner 使用）。"""

    def _make_temp_image(self) -> str:
        """一時画像ファイルを作成して path を返す。"""
        img = _make_gradient_image(32, 32)
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(f.name)
        f.close()
        return f.name

    def test_generate_dry_run_exits_zero(self):
        """generate --dry-run がエラーなく終了する。"""
        from typer.testing import CliRunner
        from backend.cli import app

        runner = CliRunner()
        img_path = self._make_temp_image()
        try:
            result = runner.invoke(app, ["generate", "--image", img_path, "--dry-run"])
            assert result.exit_code == 0, result.output
        finally:
            os.unlink(img_path)

    def test_generate_dry_run_output_contains_dry_run(self):
        """generate --dry-run の出力に [dry-run] が含まれる。"""
        from typer.testing import CliRunner
        from backend.cli import app

        runner = CliRunner()
        img_path = self._make_temp_image()
        try:
            result = runner.invoke(app, ["generate", "--image", img_path, "--dry-run"])
            assert "dry-run" in result.output.lower()
        finally:
            os.unlink(img_path)

    def test_optimize_dry_run_exits_zero(self):
        """optimize --dry-run がエラーなく終了する。"""
        from typer.testing import CliRunner
        from backend.cli import app

        runner = CliRunner()
        img_path = self._make_temp_image()
        try:
            result = runner.invoke(app, [
                "optimize", "--image", img_path,
                "--category", "portrait", "--dry-run"
            ])
            assert result.exit_code == 0, result.output
        finally:
            os.unlink(img_path)

    def test_generate_invalid_difficulty_exits_nonzero(self):
        """generate に無効な difficulty を渡すと exit_code != 0。"""
        from typer.testing import CliRunner
        from backend.cli import app

        runner = CliRunner()
        img_path = self._make_temp_image()
        try:
            result = runner.invoke(app, [
                "generate", "--image", img_path,
                "--difficulty", "godmode", "--dry-run"
            ])
            assert result.exit_code != 0
        finally:
            os.unlink(img_path)

    def test_generate_preset_masterpiece_dry_run(self):
        """--preset masterpiece --dry-run がエラーなく終了する。"""
        from typer.testing import CliRunner
        from backend.cli import app

        runner = CliRunner()
        img_path = self._make_temp_image()
        try:
            result = runner.invoke(app, [
                "generate", "--image", img_path,
                "--preset", "masterpiece", "--dry-run"
            ])
            assert result.exit_code == 0, result.output
            assert "dry-run" in result.output.lower()
        finally:
            os.unlink(img_path)

    def test_generate_preset_masterpiece_generates_png(self, tmp_path):
        """--preset masterpiece で PNG が生成されること。"""
        from typer.testing import CliRunner
        from backend.cli import app

        runner = CliRunner()
        img_path = self._make_temp_image()
        out_path = str(tmp_path / "out_masterpiece.png")
        try:
            result = runner.invoke(app, [
                "generate", "--image", img_path,
                "--preset", "masterpiece",
                "--output", out_path,
            ])
            assert result.exit_code == 0, result.output
            assert Path(out_path).exists(), "出力 PNG が生成されなかった"
            assert Path(out_path).stat().st_size > 0, "出力 PNG が空"
            assert "masterpiece" in result.output.lower()
        finally:
            os.unlink(img_path)

    def test_generate_invalid_preset_exits_nonzero(self):
        """generate に無効な preset を渡すと exit_code != 0。"""
        from typer.testing import CliRunner
        from backend.cli import app

        runner = CliRunner()
        img_path = self._make_temp_image()
        try:
            result = runner.invoke(app, [
                "generate", "--image", img_path,
                "--preset", "invalid_preset", "--dry-run"
            ])
            assert result.exit_code != 0
        finally:
            os.unlink(img_path)
