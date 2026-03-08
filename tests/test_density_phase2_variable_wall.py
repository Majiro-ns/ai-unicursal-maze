# -*- coding: utf-8 -*-
"""
密度迷路 Phase 2 可変壁厚レンダリングテスト（masterpiece柱1）。
Xu & Kaplan (SIGGRAPH 2007) 濃淡公式: G = (S-W)/S
暗いセル → 壁厚大 / 明るいセル → 壁厚小。
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from backend.core.density.exporter import _wall_stroke
from backend.core.density import generate_density_maze


def _make_image(value: int, w: int = 64, h: int = 64) -> Image.Image:
    """単一輝度値の均一画像（L モード）。"""
    arr = np.full((h, w), value, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def _make_gradient_image(w: int = 64, h: int = 64) -> Image.Image:
    """左（暗）→右（明）のグラデーション画像。"""
    arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return Image.fromarray(arr, mode="L")


class TestWallStrokeFormula:
    """_wall_stroke() 公式の単体テスト。"""

    def test_dark_cell_gives_thick_wall(self):
        """avg_lum=0（黒）で壁厚が最大（stroke_width * (1 + thickness_range)）。"""
        result = _wall_stroke(2.0, avg_lum=0.0, thickness_range=1.5)
        assert result == pytest.approx(2.0 * (1.0 + 1.5))  # 5.0

    def test_bright_cell_gives_thin_wall(self):
        """avg_lum=1（白）で壁厚が最小（stroke_width * 1.0）。"""
        result = _wall_stroke(2.0, avg_lum=1.0, thickness_range=1.5)
        assert result == pytest.approx(2.0 * 1.0)  # 2.0

    def test_mid_cell_interpolates(self):
        """avg_lum=0.5 で壁厚が中間。"""
        result = _wall_stroke(2.0, avg_lum=0.5, thickness_range=1.5)
        assert result == pytest.approx(2.0 * (1.0 + 1.5 * 0.5))  # 3.5

    def test_thickness_range_zero_is_flat(self):
        """thickness_range=0 のとき全セル同じ壁厚（固定）。"""
        for lum in [0.0, 0.5, 1.0]:
            assert _wall_stroke(2.0, lum, 0.0) == pytest.approx(2.0)

    def test_wall_increases_as_luminance_decreases(self):
        """luminance が下がるほど壁厚が増加する単調性。"""
        lums = [1.0, 0.75, 0.5, 0.25, 0.0]
        widths = [_wall_stroke(2.0, lum, 1.5) for lum in lums]
        assert widths == sorted(widths)  # 単調増加


class TestVariableWallSVG:
    """maze_to_svg における可変壁厚の結合テスト。"""

    def _extract_line_widths(self, svg: str) -> list[float]:
        """<line> 要素の stroke-width のみを抽出（解経路 <path> を除外）。"""
        import re
        return [float(x) for x in re.findall(r'<line [^/]* stroke-width="([\d.]+)"', svg)]

    def test_dark_image_svg_has_thicker_walls(self):
        """全黒画像の SVG 壁線は全白画像よりも stroke-width が大きい。"""
        dark_img = _make_image(0)
        bright_img = _make_image(255)

        # show_solution=False で解経路を除外し壁のみを比較
        dark_result = generate_density_maze(dark_img, grid_size=5, max_side=64,
                                             stroke_width=2.0, thickness_range=1.5,
                                             contrast_boost=0.0, show_solution=False)
        bright_result = generate_density_maze(bright_img, grid_size=5, max_side=64,
                                               stroke_width=2.0, thickness_range=1.5,
                                               contrast_boost=0.0, show_solution=False)

        dark_widths = self._extract_line_widths(dark_result.svg)
        bright_widths = self._extract_line_widths(bright_result.svg)

        if dark_widths and bright_widths:
            assert max(dark_widths) > max(bright_widths), (
                f"暗い画像の最大壁厚 {max(dark_widths):.3f} ≤ 明るい画像 {max(bright_widths):.3f}"
            )

    def test_thickness_range_zero_svg_all_same_width(self):
        """thickness_range=0 のとき SVG の壁線が全て同じ stroke-width。"""
        img = _make_gradient_image()
        result = generate_density_maze(img, grid_size=6, max_side=64,
                                        stroke_width=2.0, thickness_range=0.0,
                                        contrast_boost=0.0, show_solution=False)
        # <line> 要素の stroke-width が全て同一
        widths = set(self._extract_line_widths(result.svg))
        assert len(widths) == 1, (
            f"thickness_range=0 なのに壁厚が異なる値が混在: {widths}"
        )

    def test_svg_contains_variable_widths_for_gradient_image(self):
        """グラデーション画像は SVG 内に複数の異なる stroke-width が現れる。"""
        import re
        img = _make_gradient_image()
        result = generate_density_maze(img, grid_size=8, max_side=64,
                                        stroke_width=2.0, thickness_range=1.5,
                                        contrast_boost=0.0)
        widths = set(re.findall(r'stroke-width="([\d.]+)"', result.svg))
        assert len(widths) > 1, "グラデーション画像なのに壁厚が均一（可変壁厚が機能していない）"


class TestVariableWallPNG:
    """maze_to_png の動作テスト（出力チェックのみ）。"""

    def test_png_returns_bytes(self):
        """PNG 出力が bytes であること。"""
        img = _make_gradient_image()
        result = generate_density_maze(img, grid_size=5, max_side=64,
                                        thickness_range=1.5, contrast_boost=0.0)
        assert isinstance(result.png_bytes, bytes)
        assert len(result.png_bytes) > 0

    def test_dark_png_darker_than_bright(self):
        """全黒画像（壁太）のPNGは全白画像（壁細）より黒ピクセルが多い。"""
        from PIL import Image as PILImage
        import io

        dark_img = _make_image(0)
        bright_img = _make_image(255)

        dark_result = generate_density_maze(dark_img, grid_size=6, max_side=64,
                                             stroke_width=2.0, thickness_range=1.5,
                                             contrast_boost=0.0)
        bright_result = generate_density_maze(bright_img, grid_size=6, max_side=64,
                                               stroke_width=2.0, thickness_range=1.5,
                                               contrast_boost=0.0)

        def count_dark_pixels(png_bytes: bytes) -> int:
            pil = PILImage.open(io.BytesIO(png_bytes)).convert("L")
            arr = np.array(pil)
            return int((arr < 128).sum())

        dark_count = count_dark_pixels(dark_result.png_bytes)
        bright_count = count_dark_pixels(bright_result.png_bytes)
        assert dark_count > bright_count, (
            f"全黒画像の黒ピクセル {dark_count} ≤ 全白画像 {bright_count}"
        )

    def test_generate_density_maze_accepts_thickness_range(self):
        """generate_density_maze が thickness_range パラメータを受け付けること。"""
        img = _make_gradient_image()
        result = generate_density_maze(img, grid_size=5, max_side=64, thickness_range=2.0)
        assert result.entrance >= 0
        assert result.exit_cell >= 0
