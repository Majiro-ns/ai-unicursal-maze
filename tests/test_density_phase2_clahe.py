# -*- coding: utf-8 -*-
"""
密度迷路 Phase 2 CLAHEコントラスト自動調整テスト。
preprocess_image の contrast_boost パラメータ検証。
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from backend.core.density.preprocess import preprocess_image
from backend.core.density import generate_density_maze


def _make_low_contrast_image(w: int = 64, h: int = 64) -> Image.Image:
    """低コントラスト画像（値が 100〜140 に集中）。"""
    rng = np.random.default_rng(42)
    arr = rng.integers(100, 141, size=(h, w), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def _make_gradient_image(w: int = 64, h: int = 64) -> Image.Image:
    """グラデーション画像（0〜255 の線形勾配）。"""
    arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return Image.fromarray(arr, mode="L")


class TestCLAHEContrastBoost:
    def test_output_range_with_clahe(self):
        """CLAHE 適用後も出力は 0.0〜1.0 に収まること。"""
        img = _make_low_contrast_image()
        result = preprocess_image(img, max_side=64, contrast_boost=1.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_range_without_clahe(self):
        """contrast_boost=0.0（CLAHE無効）でも 0.0〜1.0 に収まること。"""
        img = _make_low_contrast_image()
        result = preprocess_image(img, max_side=64, contrast_boost=0.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_clahe_improves_contrast(self):
        """CLAHE 適用後は標準偏差が増加（コントラスト改善）。"""
        img = _make_low_contrast_image()
        before = preprocess_image(img, max_side=64, contrast_boost=0.0)
        after = preprocess_image(img, max_side=64, contrast_boost=1.0)
        assert after.std() > before.std(), (
            f"CLAHE後の標準偏差 {after.std():.4f} が適用前 {before.std():.4f} 以下"
        )

    def test_contrast_boost_zero_skips_clahe(self):
        """contrast_boost=0.0 はグレースケール変換のみで CLAHE を適用しないこと。"""
        img = _make_gradient_image()
        # CLAHE なし: 線形グラデーションが保持される
        result_no_clahe = preprocess_image(img, max_side=64, contrast_boost=0.0)
        # CLAHE あり: 非線形補正が入る
        result_clahe = preprocess_image(img, max_side=64, contrast_boost=1.0)
        # グラデーション画像で CLAHE なし版は線形に近いはず（max 差が大きい）
        assert not np.allclose(result_no_clahe, result_clahe, atol=1e-3), (
            "contrast_boost=0.0 と 1.0 の結果が同一：CLAHE がスキップされていない可能性"
        )

    def test_contrast_boost_default_is_one(self):
        """デフォルト (contrast_boost=1.0) と明示指定の結果が一致。"""
        img = _make_low_contrast_image()
        default_result = preprocess_image(img, max_side=64)
        explicit_result = preprocess_image(img, max_side=64, contrast_boost=1.0)
        np.testing.assert_array_equal(default_result, explicit_result)

    def test_contrast_boost_nonzero_improves_contrast(self):
        """contrast_boost > 0 であれば無効時より標準偏差が増加すること。"""
        img = _make_low_contrast_image()
        std_none = preprocess_image(img, max_side=64, contrast_boost=0.0).std()
        std_half = preprocess_image(img, max_side=64, contrast_boost=0.5).std()
        std_double = preprocess_image(img, max_side=64, contrast_boost=2.0).std()
        assert std_half > std_none, (
            f"contrast_boost=0.5 の std {std_half:.4f} が 0.0 の std {std_none:.4f} 以下"
        )
        assert std_double > std_none, (
            f"contrast_boost=2.0 の std {std_double:.4f} が 0.0 の std {std_none:.4f} 以下"
        )

    def test_generate_density_maze_accepts_contrast_boost(self):
        """generate_density_maze が contrast_boost パラメータを受け付けること。"""
        img = _make_low_contrast_image()
        result = generate_density_maze(img, grid_size=5, max_side=64, contrast_boost=1.0)
        assert result.entrance >= 0
        assert result.exit_cell >= 0

    def test_generate_density_maze_contrast_boost_zero(self):
        """contrast_boost=0.0 でも generate_density_maze が正常動作すること。"""
        img = _make_low_contrast_image()
        result = generate_density_maze(img, grid_size=5, max_side=64, contrast_boost=0.0)
        assert result.entrance >= 0
        assert result.exit_cell >= 0
