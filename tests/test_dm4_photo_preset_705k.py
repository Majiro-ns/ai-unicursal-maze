# -*- coding: utf-8 -*-
"""
cmd_705k_a7: Photo カテゴリ SSIM 根本改善テスト

検証内容:
  A: PHOTO_PRESET 定数検証（設定値の整合性）
  B: generate_dm4_maze_photo() API 基本動作
  C: Photo SSIM 目標達成確認（≥ 0.70 excellent）
  D: 改善前後の比較（ベースライン比 +20%以上）
  E: 他カテゴリへの非破壊確認（既存テスト互換性）
  F: エッジケーステスト（不正入力・均一画像等）
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.core.density.dm4 import (
    DM4Config,
    DM4Result,
    PHOTO_PRESET,
    generate_dm4_maze,
    generate_dm4_maze_photo,
)


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _gradient_image(size: int = 64) -> Image.Image:
    """水平グラデーション（0→255）。"""
    arr = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    return Image.fromarray(arr, mode="L")


def _photo_like_image(size: int = 128) -> Image.Image:
    """
    Photo特性を模倣した画像: mean≈0.5, std≈0.25, ガウス分布輝度。
    実際のsample_photo.pngに近い統計特性。
    """
    rng = np.random.default_rng(42)
    arr = rng.normal(loc=0.5, scale=0.25, size=(size, size))
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _uniform_image(val: int = 128, size: int = 64) -> Image.Image:
    """均一グレー画像。"""
    return Image.fromarray(np.full((size, size), val, dtype=np.uint8), mode="L")


_SAMPLE_PHOTO = _REPO_ROOT / "data" / "input" / "sample_photo.png"
_SAMPLE_LOGO = _REPO_ROOT / "data" / "input" / "sample_logo.png"


# ---------------------------------------------------------------------------
# A: PHOTO_PRESET 定数検証
# ---------------------------------------------------------------------------

class TestPhotoPresetConstants:
    """PHOTO_PRESET の設定値が根本改善に必要な値を持つことを確認。"""

    def test_photo_preset_exists(self):
        """PHOTO_PRESET が定義されている。"""
        assert PHOTO_PRESET is not None
        assert isinstance(PHOTO_PRESET, dict)

    def test_auto_clahe_is_false(self):
        """auto_clahe=False（手動CLAHE制御）。"""
        assert PHOTO_PRESET["auto_clahe"] is False

    def test_clahe_clip_limit_is_small(self):
        """clahe_clip_limit ≤ 0.02（Photo過剰補正を防ぐ低クリップ）。"""
        assert PHOTO_PRESET["clahe_clip_limit"] <= 0.02

    def test_clahe_tile_size_is_8(self):
        """clahe_tile_size = 8（局所コントラスト最小化）。"""
        assert PHOTO_PRESET["clahe_tile_size"] == 8

    def test_density_min_is_around_half(self):
        """density_min ≥ 0.40（中央収束・輝度安定化）。"""
        assert PHOTO_PRESET["density_min"] >= 0.40

    def test_density_max_is_around_half(self):
        """density_max ≤ 0.60（中央収束・輝度安定化）。"""
        assert PHOTO_PRESET["density_max"] <= 0.60

    def test_density_range_is_symmetric(self):
        """density_min と density_max が 0.5 を中心に概ね対称。"""
        dmin = PHOTO_PRESET["density_min"]
        dmax = PHOTO_PRESET["density_max"]
        center = (dmin + dmax) / 2.0
        assert abs(center - 0.5) < 0.05

    def test_render_scale_is_4(self):
        """render_scale = 4（高解像度アンチエイリアス）。"""
        assert PHOTO_PRESET["render_scale"] == 4

    def test_blur_radius_in_optimal_range(self):
        """blur_radius ∈ [2.0, 4.0]（最適ぼかし範囲）。"""
        assert 2.0 <= PHOTO_PRESET["blur_radius"] <= 4.0

    def test_grid_rows_is_32(self):
        """grid_rows = 32（128px画像の最大有効グリッド）。"""
        assert PHOTO_PRESET["grid_rows"] == 32

    def test_grid_cols_equals_grid_rows(self):
        """grid_cols = grid_rows（正方形グリッド）。"""
        assert PHOTO_PRESET["grid_cols"] == PHOTO_PRESET["grid_rows"]

    def test_passage_ratio_is_small(self):
        """passage_ratio ≤ 0.2（MASTERPIECE_PRESET 互換の小さい通路幅）。"""
        assert PHOTO_PRESET["passage_ratio"] <= 0.2

    def test_preset_has_all_required_keys(self):
        """必須キーが全て存在する。"""
        required_keys = {
            "auto_clahe", "clahe_clip_limit", "clahe_tile_size",
            "density_min", "density_max", "render_scale", "blur_radius",
            "grid_rows", "grid_cols", "passage_ratio",
        }
        for key in required_keys:
            assert key in PHOTO_PRESET, f"キー '{key}' が PHOTO_PRESET に存在しない"


# ---------------------------------------------------------------------------
# B: generate_dm4_maze_photo() API 基本動作
# ---------------------------------------------------------------------------

class TestGenerateDm4MazePhotoAPI:
    """generate_dm4_maze_photo の基本動作テスト。"""

    def test_returns_dm4result(self):
        """DM4Result インスタンスを返す。"""
        img = _photo_like_image(64)
        result = generate_dm4_maze_photo(img)
        assert isinstance(result, DM4Result)

    def test_png_bytes_is_valid(self):
        """png_bytes が有効な PNG（PIL で開ける）。"""
        img = _photo_like_image(64)
        result = generate_dm4_maze_photo(img)
        pil = Image.open(io.BytesIO(result.png_bytes))
        assert pil.width > 0
        assert pil.height > 0

    def test_ssim_score_is_float(self):
        """ssim_score が float 型。"""
        img = _photo_like_image(64)
        result = generate_dm4_maze_photo(img)
        assert isinstance(result.ssim_score, float)

    def test_ssim_in_valid_range(self):
        """ssim_score が [-1, 1] の範囲内。"""
        img = _photo_like_image(64)
        result = generate_dm4_maze_photo(img)
        assert -1.0 <= result.ssim_score <= 1.0

    def test_solution_path_non_empty(self):
        """solution_path が非空（迷路に解経路が存在）。"""
        img = _photo_like_image(64)
        result = generate_dm4_maze_photo(img)
        assert len(result.solution_path) > 0

    def test_entrance_exit_distinct(self):
        """entrance と exit_cell が異なるセル。"""
        img = _photo_like_image(64)
        result = generate_dm4_maze_photo(img)
        assert result.entrance != result.exit_cell

    def test_svg_contains_svg_tag(self):
        """svg フィールドに <svg> タグが含まれる。"""
        img = _photo_like_image(64)
        result = generate_dm4_maze_photo(img)
        assert "<svg" in result.svg

    def test_grid_dimensions_positive(self):
        """grid_rows, grid_cols が正の整数。"""
        img = _photo_like_image(64)
        result = generate_dm4_maze_photo(img)
        assert result.grid_rows > 0
        assert result.grid_cols > 0

    def test_no_config_uses_photo_preset(self):
        """config=None のとき PHOTO_PRESET が適用される（grid≥16）。"""
        img = _photo_like_image(64)
        result = generate_dm4_maze_photo(img)
        # 64px → min(32, max(64//4, 1)) = min(32, 16) = 16
        assert result.grid_rows >= 16

    def test_accepts_rgb_image(self):
        """RGB 画像（L に変換されて処理）でも動作する。"""
        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        result = generate_dm4_maze_photo(img)
        assert isinstance(result, DM4Result)


# ---------------------------------------------------------------------------
# C: Photo SSIM 目標達成確認
# ---------------------------------------------------------------------------

class TestPhotoSSIMTarget:
    """Photo 専用プリセットで SSIM ≥ 0.70 (excellent) を達成することを確認。"""

    def test_gradient_photo_ssim_above_070(self):
        """グラデーション画像で Photo プリセット SSIM ≥ 0.70。"""
        img = _gradient_image(64)
        result = generate_dm4_maze_photo(img)
        assert result.ssim_score >= 0.70, (
            f"gradient Photo SSIM={result.ssim_score:.4f} < 0.70"
        )

    @pytest.mark.skipif(
        not _SAMPLE_PHOTO.exists(),
        reason="sample_photo.png が存在しない"
    )
    def test_sample_photo_ssim_above_070(self):
        """sample_photo.png で SSIM ≥ 0.70 達成（根本改善確認）。"""
        img = Image.open(_SAMPLE_PHOTO)
        result = generate_dm4_maze_photo(img)
        assert result.ssim_score >= 0.70, (
            f"sample_photo SSIM={result.ssim_score:.4f} < 0.70 "
            f"(excellent 未達)"
        )

    @pytest.mark.skipif(
        not _SAMPLE_PHOTO.exists(),
        reason="sample_photo.png が存在しない"
    )
    def test_sample_photo_ssim_excellent_rating(self):
        """sample_photo.png で excellent 水準（≥ 0.70）に到達。"""
        img = Image.open(_SAMPLE_PHOTO)
        result = generate_dm4_maze_photo(img)
        EXCELLENT_THRESHOLD = 0.70
        rating = "excellent" if result.ssim_score >= EXCELLENT_THRESHOLD else "below_excellent"
        assert rating == "excellent", (
            f"Photo SSIM={result.ssim_score:.4f}: excellent 未達"
        )

    def test_photo_like_image_ssim_positive(self):
        """Photoらしい均一輝度分布画像で SSIM > 0.0。"""
        img = _photo_like_image(64)
        result = generate_dm4_maze_photo(img)
        assert result.ssim_score > 0.0


# ---------------------------------------------------------------------------
# D: 改善前後比較（ベースライン比 +20%以上）
# ---------------------------------------------------------------------------

class TestPhotoSSIMImprovement:
    """Photo プリセット vs ベースラインの改善を確認。"""

    def test_photo_preset_better_than_default(self):
        """Photo プリセットはデフォルト DM4Config より SSIM が高い。"""
        img = _gradient_image(64)
        baseline = generate_dm4_maze(img, config=DM4Config(passage_ratio=0.10))
        photo_result = generate_dm4_maze_photo(img)
        assert photo_result.ssim_score >= baseline.ssim_score, (
            f"Photo SSIM={photo_result.ssim_score:.4f} < "
            f"baseline={baseline.ssim_score:.4f}"
        )

    @pytest.mark.skipif(
        not _SAMPLE_PHOTO.exists(),
        reason="sample_photo.png が存在しない"
    )
    def test_sample_photo_improvement_over_baseline(self):
        """sample_photo.png で Photo プリセットがベースライン + 0.10 以上改善。"""
        img = Image.open(_SAMPLE_PHOTO)
        baseline = generate_dm4_maze(img, config=DM4Config(passage_ratio=0.10))
        photo_result = generate_dm4_maze_photo(img)
        improvement = photo_result.ssim_score - baseline.ssim_score
        assert improvement >= 0.10, (
            f"改善量 {improvement:.4f} < 0.10 "
            f"(before={baseline.ssim_score:.4f}, after={photo_result.ssim_score:.4f})"
        )

    @pytest.mark.skipif(
        not _SAMPLE_PHOTO.exists(),
        reason="sample_photo.png が存在しない"
    )
    def test_sample_photo_relative_improvement_over_20pct(self):
        """Photo プリセットがベースラインより 20% 以上の相対改善。"""
        img = Image.open(_SAMPLE_PHOTO)
        baseline = generate_dm4_maze(img, config=DM4Config(passage_ratio=0.10))
        photo_result = generate_dm4_maze_photo(img)
        if baseline.ssim_score > 0.0:
            relative = (photo_result.ssim_score - baseline.ssim_score) / baseline.ssim_score
            assert relative >= 0.20, (
                f"相対改善率 {relative:.2%} < 20%"
            )

    def test_photo_preset_non_regressive_on_gradient(self):
        """Photo プリセットがグラデーション画像で SSIM ≥ 0.50 を維持。"""
        img = _gradient_image(64)
        result = generate_dm4_maze_photo(img)
        assert result.ssim_score >= 0.50, (
            f"グラデーション画像で Photo プリセット SSIM={result.ssim_score:.4f} < 0.50"
        )


# ---------------------------------------------------------------------------
# E: 他カテゴリへの非破壊確認
# ---------------------------------------------------------------------------

class TestNonRegressionOtherCategories:
    """Photo プリセットが他カテゴリの既存テストに影響しないことを確認。"""

    @pytest.mark.skipif(
        not _SAMPLE_LOGO.exists(),
        reason="sample_logo.png が存在しない"
    )
    def test_logo_default_config_not_affected(self):
        """Logo画像は DM4 デフォルト設定で SSIM ≥ 0.70 を維持。"""
        img = Image.open(_SAMPLE_LOGO)
        result = generate_dm4_maze(img, config=DM4Config(passage_ratio=0.10))
        assert result.ssim_score >= 0.70, (
            f"Logo SSIM={result.ssim_score:.4f} < 0.70（非破壊確認失敗）"
        )

    def test_gradient_default_config_not_affected(self):
        """グラデーション画像は DM4 デフォルト設定で SSIM ≥ 0.70 を維持。"""
        img = _gradient_image(64)
        result = generate_dm4_maze(img, config=DM4Config(passage_ratio=0.10))
        assert result.ssim_score >= 0.70, (
            f"gradient SSIM={result.ssim_score:.4f} < 0.70（非破壊確認失敗）"
        )

    def test_photo_preset_import_from_density_package(self):
        """PHOTO_PRESET と generate_dm4_maze_photo が density パッケージから import できる。"""
        from backend.core.density import PHOTO_PRESET as pp
        from backend.core.density import generate_dm4_maze_photo as fn
        assert pp is not None
        assert callable(fn)

    def test_dm4_module_still_exports_compute_ssim(self):
        """_compute_ssim が dm4 モジュールからアクセスできる（既存テスト互換）。"""
        from backend.core.density.dm4 import _compute_ssim
        assert callable(_compute_ssim)

    def test_masterpiece_preset_unchanged(self):
        """MASTERPIECE_PRESET が変更されていないことを確認。"""
        from backend.core.density import MASTERPIECE_PRESET
        assert MASTERPIECE_PRESET["passage_ratio"] == pytest.approx(0.10)
        assert MASTERPIECE_PRESET["grid_size"] == 8


# ---------------------------------------------------------------------------
# F: エッジケーステスト
# ---------------------------------------------------------------------------

class TestPhotoPresetEdgeCases:
    """generate_dm4_maze_photo のエッジケーステスト。"""

    def test_small_image_16x16(self):
        """16×16 の小さい画像でもエラーなく動作。"""
        arr = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        result = generate_dm4_maze_photo(img)
        assert isinstance(result, DM4Result)

    def test_uniform_black_image(self):
        """真っ黒（0）の均一画像でも動作する。"""
        img = _uniform_image(0, 64)
        result = generate_dm4_maze_photo(img)
        assert isinstance(result, DM4Result)
        assert isinstance(result.ssim_score, float)

    def test_uniform_white_image(self):
        """真っ白（255）の均一画像でも動作する。"""
        img = _uniform_image(255, 64)
        result = generate_dm4_maze_photo(img)
        assert isinstance(result, DM4Result)
        assert isinstance(result.ssim_score, float)

    def test_custom_config_overrides_blur(self):
        """カスタム config を渡すと blur_radius を上書きできる。"""
        img = _gradient_image(64)
        # blur_radius=0.0 で渡す（デフォルト 2.0 と異なる）
        custom_config = DM4Config(blur_radius=0.0)
        result = generate_dm4_maze_photo(img, config=custom_config)
        assert isinstance(result, DM4Result)
        assert isinstance(result.ssim_score, float)

    def test_ssim_deterministic(self):
        """同一入力で generate_dm4_maze_photo は決定的な SSIM を返す。"""
        img = _gradient_image(64)
        r1 = generate_dm4_maze_photo(img)
        r2 = generate_dm4_maze_photo(img)
        assert r1.ssim_score == pytest.approx(r2.ssim_score, abs=1e-6), (
            f"SSIM が非決定的: {r1.ssim_score:.6f} vs {r2.ssim_score:.6f}"
        )

    def test_result_has_dark_coverage(self):
        """DM4Result に dark_coverage フィールドが存在する。"""
        img = _photo_like_image(64)
        result = generate_dm4_maze_photo(img)
        assert hasattr(result, "dark_coverage")
        assert 0.0 <= result.dark_coverage <= 1.0

    def test_photo_preset_with_none_config_is_consistent(self):
        """config=None を明示的に渡しても同じ結果。"""
        img = _gradient_image(64)
        r1 = generate_dm4_maze_photo(img)
        r2 = generate_dm4_maze_photo(img, config=None)
        assert r1.ssim_score == pytest.approx(r2.ssim_score, abs=1e-6)
