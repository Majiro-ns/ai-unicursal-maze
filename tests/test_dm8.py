"""
tests/test_dm8.py — DM-8 マルチスケール最適化テスト（25件）

テストカテゴリ:
  1. build_multiscale_density_map ユニットテスト (8件)
  2. DM8Config デフォルト値・バリデーション (5件)
  3. DM8Result デフォルト値 (3件)
  4. generate_dm8_maze スモークテスト (6件)
  5. 一意解保証テスト (3件)
  合計: 25件
"""
from __future__ import annotations

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
from backend.core.density.dm6 import DM6Config


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_gradient_image(width: int = 32, height: int = 32) -> Image.Image:
    """テスト用グラデーション画像（RGB）。"""
    arr = np.linspace(0, 255, width * height, dtype=np.uint8).reshape(height, width)
    rgb = np.stack([arr, arr, arr], axis=2)
    return Image.fromarray(rgb, mode="RGB")


def _make_dark_image(width: int = 32, height: int = 32) -> Image.Image:
    """全ピクセルが暗い画像（値=30）。"""
    arr = np.full((height, width), 30, dtype=np.uint8)
    rgb = np.stack([arr, arr, arr], axis=2)
    return Image.fromarray(rgb, mode="RGB")


def _make_bright_image(width: int = 32, height: int = 32) -> Image.Image:
    """全ピクセルが明るい画像（値=230）。"""
    arr = np.full((height, width), 230, dtype=np.uint8)
    rgb = np.stack([arr, arr, arr], axis=2)
    return Image.fromarray(rgb, mode="RGB")


def _make_checkerboard_image(width: int = 32, height: int = 32) -> Image.Image:
    """チェッカーボード画像（暗/明交互）。"""
    arr = np.zeros((height, width), dtype=np.uint8)
    for r in range(height):
        for c in range(width):
            arr[r, c] = 0 if (r + c) % 2 == 0 else 255
    rgb = np.stack([arr, arr, arr], axis=2)
    return Image.fromarray(rgb, mode="RGB")


def _make_gray_array(rows: int = 32, cols: int = 32, fill: float = 0.5) -> np.ndarray:
    """テスト用グレースケール配列 (rows, cols) float 0〜1。"""
    return np.full((rows, cols), fill, dtype=np.float64)


# ===========================================================================
# 1. build_multiscale_density_map ユニットテスト (8件)
# ===========================================================================

class TestBuildMultiscaleDensityMap:
    def test_output_shape(self):
        """出力形状が (target_rows, target_cols) であること。"""
        gray = _make_gray_array(64, 64, fill=0.5)
        result = build_multiscale_density_map(gray, target_rows=16, target_cols=16)
        assert result.shape == (16, 16)

    def test_output_range(self):
        """出力値域が [0.0, 1.0] であること。"""
        gray = _make_gray_array(64, 64, fill=0.7)
        result = build_multiscale_density_map(gray, target_rows=12, target_cols=12)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_uniform_bright_image(self):
        """全ピクセルが同じ明るい値 → マルチスケールマップも均一になる。"""
        gray = _make_gray_array(32, 32, fill=0.8)
        result = build_multiscale_density_map(gray, target_rows=8, target_cols=8)
        # 均一画像はスケールに関わらず値が揃う
        assert result.max() - result.min() < 0.05

    def test_uniform_dark_image(self):
        """全ピクセルが暗い値 → 出力も暗い（低輝度）。"""
        gray = _make_gray_array(32, 32, fill=0.1)
        result = build_multiscale_density_map(gray, target_rows=8, target_cols=8)
        assert result.mean() < 0.3

    def test_scale_weights_normalization(self):
        """scale_weights の合計が 1.0 でなくても自動正規化される。"""
        gray = _make_gray_array(32, 32, fill=0.5)
        result_default = build_multiscale_density_map(
            gray, target_rows=8, target_cols=8, scale_weights=(0.2, 0.3, 0.5)
        )
        # (2, 3, 5) は (0.2, 0.3, 0.5) と等価
        result_scaled = build_multiscale_density_map(
            gray, target_rows=8, target_cols=8, scale_weights=(2.0, 3.0, 5.0)
        )
        np.testing.assert_allclose(result_default, result_scaled, atol=1e-6)

    def test_invalid_scale_weights_raises(self):
        """scale_weights の合計が 0 → ValueError を送出する。"""
        gray = _make_gray_array(32, 32)
        with pytest.raises(ValueError, match="合計は正"):
            build_multiscale_density_map(
                gray, target_rows=8, target_cols=8, scale_weights=(0.0, 0.0, 0.0)
            )

    def test_coarse_only_weights(self):
        """w2=w3=0 の場合、出力はグローバル構造のみを反映する。"""
        gray = np.zeros((32, 32), dtype=np.float64)
        gray[:16, :16] = 1.0  # 左上が明るい
        result = build_multiscale_density_map(
            gray, target_rows=16, target_cols=16,
            coarse_size=4, medium_size=8, scale_weights=(1.0, 0.0, 0.0)
        )
        # 左上エリアが右下より明るいはず
        top_left_mean = result[:8, :8].mean()
        bottom_right_mean = result[8:, 8:].mean()
        assert top_left_mean > bottom_right_mean

    def test_coarse_size_clamped_to_target(self):
        """coarse_size がターゲットより大きい場合もエラーを起こさない。"""
        gray = _make_gray_array(16, 16, fill=0.5)
        # coarse_size=100 はターゲット(4,4)より大きいが、内部でクランプ
        result = build_multiscale_density_map(
            gray, target_rows=4, target_cols=4, coarse_size=100, medium_size=8
        )
        assert result.shape == (4, 4)


# ===========================================================================
# 2. DM8Config デフォルト値・バリデーション (5件)
# ===========================================================================

class TestDM8Config:
    def test_default_coarse_size(self):
        """デフォルト coarse_size は 4。"""
        config = DM8Config()
        assert config.coarse_size == 4

    def test_default_medium_size(self):
        """デフォルト medium_size は 8。"""
        config = DM8Config()
        assert config.medium_size == 8

    def test_default_scale_weights(self):
        """デフォルト scale_weights は (0.2, 0.3, 0.5)。"""
        config = DM8Config()
        assert config.scale_weights == (0.2, 0.3, 0.5)

    def test_dm8config_inherits_dm6config(self):
        """DM8Config は DM6Config を継承している。"""
        assert issubclass(DM8Config, DM6Config)

    def test_dm8config_custom_scale_weights(self):
        """カスタム scale_weights を指定できる。"""
        config = DM8Config(scale_weights=(0.4, 0.3, 0.3))
        assert config.scale_weights == (0.4, 0.3, 0.3)


# ===========================================================================
# 3. DM8Result デフォルト値 (3件)
# ===========================================================================

class TestDM8Result:
    def test_default_scale_weights_used(self):
        """generate_dm8_maze 結果の scale_weights_used はデフォルト (0.2, 0.3, 0.5)。"""
        image = _make_gradient_image(32, 32)
        config = DM8Config(difficulty="easy")
        result = generate_dm8_maze(image, config)
        assert result.scale_weights_used == (0.2, 0.3, 0.5)

    def test_default_coarse_size_used(self):
        """generate_dm8_maze 結果の coarse_size_used はデフォルト 4。"""
        image = _make_gradient_image(32, 32)
        config = DM8Config(difficulty="easy")
        result = generate_dm8_maze(image, config)
        assert result.coarse_size_used == 4

    def test_default_medium_size_used(self):
        """generate_dm8_maze 結果の medium_size_used はデフォルト 8。"""
        image = _make_gradient_image(32, 32)
        config = DM8Config(difficulty="easy")
        result = generate_dm8_maze(image, config)
        assert result.medium_size_used == 8


# ===========================================================================
# 4. generate_dm8_maze スモークテスト (6件)
# ===========================================================================

class TestGenerateDm8Maze:
    def test_smoke_gradient_image(self):
        """グラデーション画像で generate_dm8_maze が正常終了する。"""
        image = _make_gradient_image(32, 32)
        config = DM8Config(difficulty="easy")
        result = generate_dm8_maze(image, config)
        assert isinstance(result, DM8Result)
        assert len(result.png_bytes) > 0

    def test_result_has_svg(self):
        """結果に SVG が含まれる。"""
        image = _make_gradient_image(32, 32)
        config = DM8Config(difficulty="easy")
        result = generate_dm8_maze(image, config)
        assert isinstance(result.svg, str)
        assert len(result.svg) > 0

    def test_grid_size_from_difficulty(self):
        """difficulty="easy" → grid_rows/grid_cols = 6。"""
        image = _make_gradient_image(64, 64)
        config = DM8Config(difficulty="easy")
        result = generate_dm8_maze(image, config)
        assert result.grid_rows == 6
        assert result.grid_cols == 6

    def test_scale_weights_stored_in_result(self):
        """指定した scale_weights が result に保存される。"""
        image = _make_gradient_image(32, 32)
        config = DM8Config(difficulty="easy", scale_weights=(0.3, 0.3, 0.4))
        result = generate_dm8_maze(image, config)
        assert result.scale_weights_used == (0.3, 0.3, 0.4)

    def test_invalid_passage_ratio_raises(self):
        """passage_ratio が範囲外 → ValueError を送出する。"""
        image = _make_gradient_image(32, 32)
        config = DM8Config(difficulty="easy", passage_ratio=0.05)
        with pytest.raises(ValueError, match="passage_ratio"):
            generate_dm8_maze(image, config)

    def test_invalid_difficulty_raises(self):
        """無効な difficulty → ValueError を送出する。"""
        image = _make_gradient_image(32, 32)
        config = DM8Config(difficulty="super_hard")
        with pytest.raises(ValueError, match="Invalid difficulty"):
            generate_dm8_maze(image, config)


# ===========================================================================
# 5. 一意解保証テスト (3件)
# ===========================================================================

class TestUniquePathGuarantee:
    def test_solution_path_not_empty(self):
        """solution_path が空でない（入口から出口への経路が存在する）。"""
        image = _make_gradient_image(32, 32)
        config = DM8Config(difficulty="easy", extra_removal_rate=0.0)
        result = generate_dm8_maze(image, config)
        assert len(result.solution_path) > 0

    def test_entrance_exit_are_different(self):
        """entrance と exit_cell が異なるセルである。"""
        image = _make_gradient_image(32, 32)
        config = DM8Config(difficulty="easy")
        result = generate_dm8_maze(image, config)
        assert result.entrance != result.exit_cell

    def test_solution_count_with_no_extra_removal(self):
        """extra_removal_rate=0 → 一意解 (solution_count==1)。"""
        image = _make_bright_image(32, 32)
        config = DM8Config(difficulty="easy", extra_removal_rate=0.0)
        result = generate_dm8_maze(image, config)
        # extra_removal_rate=0 = perfect spanning tree = 一意解
        assert result.solution_count == 1
