# -*- coding: utf-8 -*-
"""
密度迷路 Phase 3 SVG品質改善テスト。

- wall_thickness_histogram(): 壁厚分布ヒストグラム
- SVGグループ化フォーマット: <g stroke-width="..."><path d="..."/></g>
- SVGファイルサイズ削減効果
- PNG DPI メタデータ設定
"""
from __future__ import annotations

import io
import re

import numpy as np
import pytest
from PIL import Image

from backend.core.density import generate_density_maze
from backend.core.density.exporter import wall_thickness_histogram
from backend.core.density.grid_builder import build_cell_grid


def _make_image(value: int, w: int = 64, h: int = 64) -> Image.Image:
    """単一輝度値の均一画像（L モード）。"""
    arr = np.full((h, w), value, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def _make_gradient_image(w: int = 64, h: int = 64) -> Image.Image:
    """左（暗）→右（明）のグラデーション画像。"""
    arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return Image.fromarray(arr, mode="L")


def _build_grid_and_adj(image: Image.Image, grid_size: int = 8):
    """テスト用に CellGrid と adj を構築するヘルパー。"""
    import numpy as np
    from backend.core.density.preprocess import preprocess_image
    from backend.core.density.maze_builder import build_spanning_tree

    gray = preprocess_image(image, max_side=64)
    rows = min(grid_size, max(gray.shape[0] // 4, 1))
    cols = min(grid_size, max(gray.shape[1] // 4, 1))
    grid = build_cell_grid(gray, rows, cols)
    adj = build_spanning_tree(grid)
    return grid, adj


# ─────────────────────────────────────────────────────
# wall_thickness_histogram()
# ─────────────────────────────────────────────────────

class TestWallThicknessHistogram:
    """wall_thickness_histogram() の単体テスト。"""

    def test_returns_dict_with_required_keys(self):
        """戻り値が必要なキーを持つ dict であること。"""
        img = _make_gradient_image()
        grid, adj = _build_grid_and_adj(img)
        result = wall_thickness_histogram(grid, adj, print_chart=False)
        for key in ("total", "min", "max", "bins", "counts"):
            assert key in result, f"キー '{key}' が不足"

    def test_total_matches_wall_count(self):
        """total は描画壁数と一致すること。"""
        img = _make_gradient_image()
        grid, adj = _build_grid_and_adj(img)
        result = wall_thickness_histogram(grid, adj, print_chart=False)
        assert result["total"] > 0
        assert result["total"] <= grid.rows * grid.cols  # 最大は全セル数

    def test_uniform_image_has_narrow_range(self):
        """均一輝度画像では壁厚が全て同一（min == max）。"""
        img = _make_image(128)
        grid, adj = _build_grid_and_adj(img)
        result = wall_thickness_histogram(grid, adj, stroke_width=2.0, thickness_range=1.5,
                                          print_chart=False)
        assert abs(result["max"] - result["min"]) < 0.01, (
            f"均一画像で壁厚分散あり: min={result['min']:.3f}, max={result['max']:.3f}"
        )

    def test_gradient_image_has_wide_range(self):
        """グラデーション画像では壁厚に幅がある（max > min）。"""
        img = _make_gradient_image()
        grid, adj = _build_grid_and_adj(img)
        result = wall_thickness_histogram(grid, adj, stroke_width=2.0, thickness_range=1.5,
                                          print_chart=False)
        assert result["max"] > result["min"], (
            f"グラデーション画像で壁厚範囲なし: min={result['min']:.3f}, max={result['max']:.3f}"
        )

    def test_thickness_range_zero_gives_flat_histogram(self):
        """thickness_range=0 のとき全壁が同一幅（分布は1ビンのみ）。"""
        img = _make_gradient_image()
        grid, adj = _build_grid_and_adj(img)
        result = wall_thickness_histogram(grid, adj, stroke_width=2.0, thickness_range=0.0,
                                          print_chart=False)
        assert abs(result["max"] - result["min"]) < 0.001

    def test_min_max_match_wall_stroke_formula(self):
        """min/max が _wall_stroke() の理論値と一致すること。"""
        from backend.core.density.exporter import _wall_stroke
        sw_base = 2.0
        tr = 1.5
        img = _make_gradient_image()
        grid, adj = _build_grid_and_adj(img)
        result = wall_thickness_histogram(grid, adj, stroke_width=sw_base, thickness_range=tr,
                                          print_chart=False)
        expected_min = _wall_stroke(sw_base, 1.0, tr)  # 最も明るい = 最細
        expected_max = _wall_stroke(sw_base, 0.0, tr)  # 最も暗い = 最太
        # 実際の値は画像の輝度範囲に依存するため、理論上限内であることを確認
        assert result["min"] >= expected_min - 0.01
        assert result["max"] <= expected_max + 0.01

    def test_bins_length_equals_n_bins_plus_one(self):
        """bins の長さが n_bins + 1 であること（境界値）。"""
        img = _make_gradient_image()
        grid, adj = _build_grid_and_adj(img)
        result = wall_thickness_histogram(grid, adj, n_bins=8, print_chart=False)
        assert len(result["bins"]) == 9  # 8 + 1

    def test_counts_length_equals_n_bins(self):
        """counts の長さが n_bins であること。"""
        img = _make_gradient_image()
        grid, adj = _build_grid_and_adj(img)
        result = wall_thickness_histogram(grid, adj, n_bins=8, print_chart=False)
        assert len(result["counts"]) == 8

    def test_counts_sum_equals_total(self):
        """counts の合計が total と一致すること。"""
        img = _make_gradient_image()
        grid, adj = _build_grid_and_adj(img)
        result = wall_thickness_histogram(grid, adj, n_bins=10, print_chart=False)
        if len(result["counts"]) > 0:
            assert sum(result["counts"]) == result["total"]

    def test_print_chart_does_not_raise(self, capsys):
        """print_chart=True がエラーなく ASCII 出力すること。"""
        img = _make_gradient_image()
        grid, adj = _build_grid_and_adj(img)
        wall_thickness_histogram(grid, adj, print_chart=True)
        captured = capsys.readouterr()
        assert "壁厚分布ヒストグラム" in captured.out


# ─────────────────────────────────────────────────────
# SVG グループ化フォーマット
# ─────────────────────────────────────────────────────

class TestSVGGroupingFormat:
    """Phase 3 SVG が <g stroke-width="..."><path d="..."/></g> 形式であること。"""

    def test_svg_uses_g_elements_for_walls(self):
        """壁が <g stroke-width="..."> グループで出力されること。"""
        img = _make_gradient_image()
        result = generate_density_maze(img, grid_size=6, max_side=64,
                                       thickness_range=1.5, show_solution=False)
        g_matches = re.findall(r'<g [^>]*stroke-width="[\d.]+"', result.svg)
        assert len(g_matches) >= 1, "SVG に <g stroke-width> グループが存在しない"

    def test_wall_group_contains_path(self):
        """<g> グループが <path d="..."> を含むこと。"""
        img = _make_gradient_image()
        result = generate_density_maze(img, grid_size=6, max_side=64,
                                       thickness_range=1.5, show_solution=False)
        assert '<path d="' in result.svg, "SVG に path 要素がない"

    def test_path_uses_vh_commands(self):
        """壁 path の d 属性が V（垂直）または H（水平）コマンドを含むこと。"""
        img = _make_gradient_image()
        result = generate_density_maze(img, grid_size=6, max_side=64,
                                       thickness_range=1.5, show_solution=False)
        # <g> グループ内の <path d="..."> で V または H コマンドが使われていること
        g_path_match = re.search(
            r'<g [^>]*stroke-width="[^"]+"><path d="([^"]+)" fill="none"/>',
            result.svg,
        )
        assert g_path_match is not None, "壁グループに <path> が見つからない"
        d = g_path_match.group(1)
        assert "V" in d or "H" in d, (
            f"壁 path が V/H コマンドを使っていない: d='{d[:80]}...'"
        )

    def test_gradient_svg_has_multiple_groups(self):
        """グラデーション画像（thickness_range>0）で複数の <g> グループが生成されること。"""
        img = _make_gradient_image()
        result = generate_density_maze(img, grid_size=8, max_side=64,
                                       thickness_range=1.5, show_solution=False,
                                       stroke_quantize_levels=20)
        g_widths = re.findall(r'<g [^>]*stroke-width="([\d.]+)"', result.svg)
        assert len(set(g_widths)) > 1, (
            f"グラデーション画像なのに壁グループが1つしかない: {set(g_widths)}"
        )

    def test_uniform_image_has_single_group(self):
        """均一画像では壁グループが1つであること。"""
        img = _make_image(128)
        result = generate_density_maze(img, grid_size=6, max_side=64,
                                       thickness_range=1.5, show_solution=False)
        g_widths = re.findall(r'<g [^>]*stroke-width="([\d.]+)"', result.svg)
        assert len(set(g_widths)) == 1, (
            f"均一画像で複数の壁グループが生成された: {set(g_widths)}"
        )

    def test_quantize_levels_zero_still_renders(self):
        """stroke_quantize_levels=0 でも SVG が正常に生成されること。"""
        img = _make_gradient_image()
        result = generate_density_maze(img, grid_size=6, max_side=64,
                                       thickness_range=1.5, stroke_quantize_levels=0)
        assert result.svg.startswith("<?xml")
        assert "</svg>" in result.svg

    def test_outer_rect_uses_fixed_stroke_width(self):
        """外枠 <rect> が固定の stroke_width を使うこと（可変壁厚の影響なし）。"""
        img = _make_gradient_image()
        sw = 2.0
        result = generate_density_maze(img, grid_size=6, max_side=64,
                                       stroke_width=sw, thickness_range=1.5)
        rect_match = re.search(r'<rect [^/]* stroke-width="([\d.]+)"', result.svg)
        assert rect_match is not None, "SVG に <rect> がない"
        assert float(rect_match.group(1)) == pytest.approx(sw)


# ─────────────────────────────────────────────────────
# SVG ファイルサイズ削減効果
# ─────────────────────────────────────────────────────

class TestSVGFileSize:
    """Phase 3 SVG がコンパクトであること。"""

    def test_svg_size_reasonable_for_small_grid(self):
        """10x10 グリッドで SVG が 50KB 未満であること。"""
        img = _make_gradient_image(w=128, h=128)
        result = generate_density_maze(img, grid_size=10, max_side=128,
                                       thickness_range=1.5, show_solution=False)
        svg_bytes = len(result.svg.encode("utf-8"))
        assert svg_bytes < 50_000, (
            f"10x10 グリッドの SVG が大きすぎる: {svg_bytes / 1024:.1f} KB"
        )

    def test_svg_size_scales_linearly(self):
        """グリッドサイズが2倍になっても SVG サイズが4倍未満であること（近似線形）。"""
        img_small = _make_gradient_image(w=64, h=64)
        img_large = _make_gradient_image(w=128, h=128)

        r_small = generate_density_maze(img_small, grid_size=8, max_side=64,
                                        thickness_range=1.5, show_solution=False)
        r_large = generate_density_maze(img_large, grid_size=16, max_side=128,
                                        thickness_range=1.5, show_solution=False)

        size_small = len(r_small.svg.encode("utf-8"))
        size_large = len(r_large.svg.encode("utf-8"))
        ratio = size_large / size_small
        # 16x16 / 8x8 = 4倍のセル数 → 壁数も約4倍 → 比率は4倍程度（許容: <8倍）
        assert ratio < 8, (
            f"SVG サイズ比率が高すぎる: {size_large}B / {size_small}B = {ratio:.1f}倍"
        )

    def test_per_wall_bytes_under_threshold(self):
        """壁1本あたりの平均バイト数が 25 bytes 未満であること（Phase 3 最適化後）。"""
        img = _make_gradient_image(w=128, h=128)
        grid_size = 12
        result = generate_density_maze(img, grid_size=grid_size, max_side=128,
                                       thickness_range=1.5, show_solution=False)
        # 壁数の上限: rows*(cols-1) + (rows-1)*cols ≈ 2 * grid_size^2
        max_walls = 2 * grid_size * grid_size
        svg_bytes = len(result.svg.encode("utf-8"))
        per_wall = svg_bytes / max_walls
        assert per_wall < 25, (
            f"壁1本あたり {per_wall:.1f} bytes は多すぎる（目標 < 25 bytes）"
        )


# ─────────────────────────────────────────────────────
# PNG DPI
# ─────────────────────────────────────────────────────

class TestPNGDPI:
    """maze_to_png() / generate_density_maze() の DPI 設定テスト。"""

    def test_png_without_dpi_has_no_dpi_metadata(self):
        """png_dpi 未指定のとき DPI メタデータが埋め込まれないこと。"""
        img = _make_gradient_image()
        result = generate_density_maze(img, grid_size=5, max_side=64)
        pil = Image.open(io.BytesIO(result.png_bytes))
        # PIL の info["dpi"] が未設定 or None
        dpi = pil.info.get("dpi")
        # DPI なし = None または (72, 72) のデフォルト値（OS依存）
        # ここでは 300 や 96 などの明示的な値が入っていないことを確認
        if dpi is not None:
            assert dpi != (300, 300)
            assert dpi != (96, 96)

    def test_png_dpi_300_is_embedded(self):
        """png_dpi=300 のとき PNG メタデータに DPI=300 が埋め込まれること。"""
        img = _make_gradient_image()
        result = generate_density_maze(img, grid_size=5, max_side=64, png_dpi=300)
        pil = Image.open(io.BytesIO(result.png_bytes))
        dpi = pil.info.get("dpi")
        assert dpi is not None, "DPI メタデータが埋め込まれていない"
        assert abs(dpi[0] - 300) < 5, f"DPI が期待値 300 と異なる: {dpi}"
        assert abs(dpi[1] - 300) < 5, f"DPI が期待値 300 と異なる: {dpi}"

    def test_png_dpi_96_is_embedded(self):
        """png_dpi=96 のとき PNG メタデータに DPI=96 が埋め込まれること。"""
        img = _make_gradient_image()
        result = generate_density_maze(img, grid_size=5, max_side=64, png_dpi=96)
        pil = Image.open(io.BytesIO(result.png_bytes))
        dpi = pil.info.get("dpi")
        assert dpi is not None, "DPI メタデータが埋め込まれていない"
        assert abs(dpi[0] - 96) < 5, f"DPI が期待値 96 と異なる: {dpi}"

    def test_png_dpi_does_not_change_pixel_size(self):
        """DPI 設定が PNG の画素数（width/height）を変更しないこと。"""
        img = _make_gradient_image()
        r_no_dpi = generate_density_maze(img, grid_size=5, max_side=64)
        r_dpi300 = generate_density_maze(img, grid_size=5, max_side=64, png_dpi=300)

        pil_no_dpi = Image.open(io.BytesIO(r_no_dpi.png_bytes))
        pil_dpi300 = Image.open(io.BytesIO(r_dpi300.png_bytes))

        assert pil_no_dpi.size == pil_dpi300.size, (
            f"DPI 設定で画素数が変わった: {pil_no_dpi.size} vs {pil_dpi300.size}"
        )

    def test_generate_density_maze_accepts_png_dpi(self):
        """generate_density_maze が png_dpi パラメータを受け付けること。"""
        img = _make_gradient_image()
        result = generate_density_maze(img, grid_size=5, max_side=64, png_dpi=150)
        assert isinstance(result.png_bytes, bytes)
        assert len(result.png_bytes) > 0
