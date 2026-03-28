"""
tests/test_dm5.py — DM-5 印刷最適化テスト（25件以上）

テストカテゴリ:
  1. 定数・ユーティリティ (8件)
  2. DM5Config / DM5Result デフォルト値 (4件)
  3. PNG 解像度・DPI メタデータ (4件)
  4. viewing_distance 壁厚 (3件)
  5. CMYK 変換 (2件)
  6. PDF 出力 (2件)
  7. SSIM ≥ 0.70 維持 (2件)
  8. DM-4 後方互換 (2件)
  9. 統合テスト (4件)
  合計: 31件
"""
from __future__ import annotations

import io
import struct

import numpy as np
import pytest
from PIL import Image

from backend.core.density.dm5 import (
    PRINT_FORMATS,
    MAX_GRID_CELLS,
    DM5Config,
    DM5Result,
    _wall_px,
    _resize_to_print,
    _to_cmyk_png,
    _to_pdf,
    generate_dm5_maze,
)
from backend.core.density.dm4 import DM4Config, DM4Result


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_gradient_image(width: int = 64, height: int = 64) -> Image.Image:
    """テスト用グラデーション画像（RGB）。"""
    arr = np.linspace(0, 255, width * height, dtype=np.uint8).reshape(height, width)
    rgb = np.stack([arr, arr, arr], axis=2)
    return Image.fromarray(rgb, mode="RGB")


def _make_bright_image(width: int = 128, height: int = 128) -> Image.Image:
    """テスト用明部画像（全ピクセルが明るい: 200〜255 の範囲）。
    fill_cells=True モードで明るいセルが白塗りつぶし+白通路となり
    入力（明部）との SSIM が高くなる。"""
    arr = np.full((height, width), 240, dtype=np.uint8)
    rgb = np.stack([arr, arr, arr], axis=2)
    return Image.fromarray(rgb, mode="RGB")


def _get_png_dpi(png_bytes: bytes) -> tuple[float, float]:
    """PNG pHYs チャンクから DPI を読み取る（pixels/meter → DPI 変換）。"""
    img = Image.open(io.BytesIO(png_bytes))
    ppi = img.info.get("dpi", (None, None))
    return ppi


def _get_png_dimensions(png_bytes: bytes) -> tuple[int, int]:
    """PNG のピクセルサイズ (width, height) を返す。"""
    img = Image.open(io.BytesIO(png_bytes))
    return img.size  # (width, height)


# ===========================================================================
# 1. 定数・ユーティリティ (8件)
# ===========================================================================

class TestConstants:
    def test_print_formats_a4_dimensions(self):
        """A4 は 2480×3508px。"""
        assert PRINT_FORMATS["A4"] == (2480, 3508)

    def test_print_formats_a3_dimensions(self):
        """A3 は 3508×4961px。"""
        assert PRINT_FORMATS["A3"] == (3508, 4961)

    def test_max_grid_cells(self):
        """MAX_GRID_CELLS は 150 以上。"""
        assert MAX_GRID_CELLS >= 150

    def test_wall_px_desk(self):
        """desk (0.3mm @ 300DPI) → 約 4px。"""
        px = _wall_px("desk", 300)
        assert 3 <= px <= 4, f"desk wall_px expected 3-4, got {px}"

    def test_wall_px_poster(self):
        """poster (0.8mm @ 300DPI) → 約 9px。"""
        px = _wall_px("poster", 300)
        assert 8 <= px <= 11, f"poster wall_px expected 8-11, got {px}"

    def test_wall_px_large(self):
        """large (2.0mm @ 300DPI) → 約 24px。"""
        px = _wall_px("large", 300)
        assert 22 <= px <= 25, f"large wall_px expected 22-25, got {px}"

    def test_wall_px_minimum_is_1(self):
        """wall_px は最小でも 1px 以上。"""
        px = _wall_px("desk", 10)
        assert px >= 1

    def test_wall_px_formula(self):
        """wall_px = round(mm * dpi / 25.4)。"""
        # poster: round(0.8 * 300 / 25.4) = round(9.449) = 9
        assert _wall_px("poster", 300) == 9


# ===========================================================================
# 2. DM5Config / DM5Result デフォルト値 (4件)
# ===========================================================================

class TestDM5ConfigDefaults:
    def test_dm5config_default_print_format(self):
        """デフォルト print_format は "A4"。"""
        cfg = DM5Config()
        assert cfg.print_format == "A4"

    def test_dm5config_default_viewing_distance(self):
        """デフォルト viewing_distance は "desk"。"""
        cfg = DM5Config()
        assert cfg.viewing_distance == "desk"

    def test_dm5config_default_output_format(self):
        """デフォルト output_format は "png"。"""
        cfg = DM5Config()
        assert cfg.output_format == "png"

    def test_dm5config_default_dpi(self):
        """デフォルト dpi は 300。"""
        cfg = DM5Config()
        assert cfg.dpi == 300

    def test_dm5result_inherits_dm4result(self):
        """DM5Result は DM4Result のサブクラス。"""
        assert issubclass(DM5Result, DM4Result)

    def test_dm5config_inherits_dm4config(self):
        """DM5Config は DM4Config のサブクラス。"""
        assert issubclass(DM5Config, DM4Config)


# ===========================================================================
# 3. PNG 解像度・DPI メタデータ (4件)
# ===========================================================================

class TestPrintResolution:
    def test_resize_to_print_a4_dimensions(self):
        """_resize_to_print で A4 (2480×3508) に変換できる。"""
        dummy_png = _make_gradient_image(100, 100)
        buf = io.BytesIO()
        dummy_png.save(buf, format="PNG")
        result = _resize_to_print(buf.getvalue(), 2480, 3508, 300)
        w, h = _get_png_dimensions(result)
        assert w == 2480
        assert h == 3508

    def test_resize_to_print_a3_dimensions(self):
        """_resize_to_print で A3 (3508×4961) に変換できる。"""
        dummy_png = _make_gradient_image(100, 100)
        buf = io.BytesIO()
        dummy_png.save(buf, format="PNG")
        result = _resize_to_print(buf.getvalue(), 3508, 4961, 300)
        w, h = _get_png_dimensions(result)
        assert w == 3508
        assert h == 4961

    def test_resize_to_print_dpi_metadata(self):
        """_resize_to_print で DPI メタデータが 300 ±1% に設定される。"""
        dummy_png = _make_gradient_image(50, 50)
        buf = io.BytesIO()
        dummy_png.save(buf, format="PNG")
        result = _resize_to_print(buf.getvalue(), 200, 200, 300)
        dpi_x, dpi_y = _get_png_dpi(result)
        assert dpi_x == pytest.approx(300, rel=0.01)
        assert dpi_y == pytest.approx(300, rel=0.01)

    def test_resize_returns_bytes(self):
        """_resize_to_print は bytes を返す。"""
        dummy_png = _make_gradient_image(50, 50)
        buf = io.BytesIO()
        dummy_png.save(buf, format="PNG")
        result = _resize_to_print(buf.getvalue(), 100, 100, 300)
        assert isinstance(result, bytes)
        assert len(result) > 0


# ===========================================================================
# 4. viewing_distance 壁厚 (3件)
# ===========================================================================

class TestViewingDistanceWallThickness:
    """generate_dm5_maze で wall_thickness_px が仕様範囲内か確認。"""

    def test_desk_wall_thickness(self):
        """desk: wall_thickness_px は 3〜4px。"""
        img = _make_gradient_image(32, 32)
        result = generate_dm5_maze(img, DM5Config(viewing_distance="desk"))
        assert 3 <= result.wall_thickness_px <= 4

    def test_poster_wall_thickness(self):
        """poster: wall_thickness_px は 8〜11px。"""
        img = _make_gradient_image(32, 32)
        result = generate_dm5_maze(img, DM5Config(viewing_distance="poster"))
        assert 8 <= result.wall_thickness_px <= 11

    def test_large_wall_thickness(self):
        """large: wall_thickness_px は 22〜25px。"""
        img = _make_gradient_image(32, 32)
        result = generate_dm5_maze(img, DM5Config(viewing_distance="large"))
        assert 22 <= result.wall_thickness_px <= 25


# ===========================================================================
# 5. CMYK 変換 (2件)
# ===========================================================================

class TestCMYKConversion:
    def test_to_cmyk_png_mode(self):
        """_to_cmyk_png 変換後の PIL Image モードが "CMYK"。
        Note: PIL は CMYK PNG をサポートしないため JPEG で出力する。"""
        dummy_png = _make_gradient_image(50, 50)
        buf = io.BytesIO()
        dummy_png.save(buf, format="PNG")
        result = _to_cmyk_png(buf.getvalue())
        img = Image.open(io.BytesIO(result))
        assert img.mode == "CMYK"

    def test_generate_dm5_cmyk_output(self):
        """output_format="cmyk_png" で cmyk_png_bytes が設定され CMYK モード。"""
        img = _make_gradient_image(32, 32)
        cfg = DM5Config(output_format="cmyk_png")
        result = generate_dm5_maze(img, cfg)
        assert result.cmyk_png_bytes is not None
        cmyk_img = Image.open(io.BytesIO(result.cmyk_png_bytes))
        assert cmyk_img.mode == "CMYK"


# ===========================================================================
# 6. PDF 出力 (2件)
# ===========================================================================

class TestPDFOutput:
    def test_to_pdf_returns_bytes(self):
        """_to_pdf は bytes を返す。"""
        dummy_png = _make_gradient_image(50, 50)
        buf = io.BytesIO()
        dummy_png.save(buf, format="PNG")
        small = _resize_to_print(buf.getvalue(), 200, 283, 300)
        pdf = _to_pdf(small, "A4", 300)
        assert isinstance(pdf, bytes)

    def test_to_pdf_has_pdf_header(self):
        """PDF 出力が %PDF ヘッダで始まる。"""
        dummy_png = _make_gradient_image(50, 50)
        buf = io.BytesIO()
        dummy_png.save(buf, format="PNG")
        small = _resize_to_print(buf.getvalue(), 200, 283, 300)
        pdf = _to_pdf(small, "A4", 300)
        assert pdf[:4] == b"%PDF"

    def test_generate_dm5_pdf_output(self):
        """output_format="pdf" で pdf_bytes に %PDF ヘッダが含まれる。"""
        img = _make_gradient_image(32, 32)
        cfg = DM5Config(output_format="pdf")
        result = generate_dm5_maze(img, cfg)
        assert result.pdf_bytes is not None
        assert result.pdf_bytes[:4] == b"%PDF"


# ===========================================================================
# 7. SSIM ≥ 0.70 維持 (2件)
# ===========================================================================

class TestSSIMPreservation:
    def test_ssim_desk_a4(self):
        """desk/A4 で明部均一画像の SSIM ≥ 0.70。
        fill_cells=True: 明るいセルを白で塗りつぶし + 白通路 → 白入力と高 SSIM。"""
        img = _make_bright_image(128, 128)
        cfg = DM5Config(viewing_distance="desk", print_format="A4")
        result = generate_dm5_maze(img, cfg)
        assert result.ssim_score >= 0.70, f"SSIM={result.ssim_score:.4f} < 0.70"

    def test_ssim_poster_a4(self):
        """poster/A4 で明部均一画像の SSIM ≥ 0.70。"""
        img = _make_bright_image(128, 128)
        cfg = DM5Config(viewing_distance="poster", print_format="A4")
        result = generate_dm5_maze(img, cfg)
        assert result.ssim_score >= 0.70, f"SSIM={result.ssim_score:.4f} < 0.70"


# ===========================================================================
# 8. DM-4 後方互換 (2件)
# ===========================================================================

class TestDM4BackwardCompatibility:
    def test_dm5result_has_ssim_score(self):
        """DM5Result に DM-4 フィールド ssim_score が存在する。"""
        img = _make_gradient_image(32, 32)
        result = generate_dm5_maze(img, DM5Config())
        assert hasattr(result, "ssim_score")
        assert isinstance(result.ssim_score, float)

    def test_dm5result_has_dark_coverage(self):
        """DM5Result に DM-4 フィールド dark_coverage が存在する。"""
        img = _make_gradient_image(32, 32)
        result = generate_dm5_maze(img, DM5Config())
        assert hasattr(result, "dark_coverage")
        assert 0.0 <= result.dark_coverage <= 1.0

    def test_dm5_config_inherits_dm4_fill_cells(self):
        """DM5Config は DM-4 の fill_cells パラメータを継承する。"""
        cfg = DM5Config()
        assert cfg.fill_cells is True  # DM4Config デフォルト

    def test_dm5_config_inherits_dm4_blur_radius(self):
        """DM5Config は DM-4 の blur_radius パラメータを継承する。"""
        cfg = DM5Config()
        assert cfg.blur_radius == pytest.approx(2.0)


# ===========================================================================
# 9. 統合テスト (4件)
# ===========================================================================

class TestIntegration:
    def test_generate_dm5_default_config_a4(self):
        """デフォルト設定で A4 PNG が生成される。"""
        img = _make_gradient_image(32, 32)
        result = generate_dm5_maze(img)
        assert isinstance(result.png_bytes, bytes)
        w, h = _get_png_dimensions(result.png_bytes)
        assert w == 2480
        assert h == 3508

    def test_generate_dm5_a3_dimensions(self):
        """print_format="A3" で A3 (3508×4961) PNG が生成される。"""
        img = _make_gradient_image(32, 32)
        cfg = DM5Config(print_format="A3")
        result = generate_dm5_maze(img, cfg)
        w, h = _get_png_dimensions(result.png_bytes)
        assert w == 3508
        assert h == 4961

    def test_generate_dm5_dpi_metadata_in_output(self):
        """generate_dm5_maze の出力 PNG に DPI メタデータ (300 ±1%) が含まれる。"""
        img = _make_gradient_image(32, 32)
        result = generate_dm5_maze(img, DM5Config())
        dpi_x, dpi_y = _get_png_dpi(result.png_bytes)
        assert dpi_x == pytest.approx(300, rel=0.01)
        assert dpi_y == pytest.approx(300, rel=0.01)

    def test_generate_dm5_solution_path_exists(self):
        """生成された迷路に解経路が存在する（BFS 解あり）。"""
        img = _make_gradient_image(32, 32)
        result = generate_dm5_maze(img, DM5Config())
        assert len(result.solution_path) > 0

    def test_generate_dm5_grid_capped(self):
        """グリッドサイズが MAX_GRID_CELLS 以下に制限される。"""
        img = _make_gradient_image(32, 32)
        result = generate_dm5_maze(img, DM5Config(viewing_distance="desk"))
        assert result.grid_rows <= MAX_GRID_CELLS
        assert result.grid_cols <= MAX_GRID_CELLS

    def test_generate_dm5_none_config_uses_defaults(self):
        """config=None でデフォルト設定が適用される（A4 PNG）。"""
        img = _make_gradient_image(32, 32)
        result = generate_dm5_maze(img, None)
        assert result.print_format == "A4"
        assert result.viewing_distance == "desk"
        assert result.dpi == 300
