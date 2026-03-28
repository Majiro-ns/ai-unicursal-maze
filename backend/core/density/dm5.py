"""
DM-5: 印刷最適化 — A4/A3 300DPI 高解像度出力（Phase DM-5）

設計書: docs/maze-artisan-masterpiece-requirements.md Phase DM-5

DM-4 との差分:
  + 印刷フォーマット指定（A4/A3 @ 300DPI）
  + 観察距離別壁厚最適化（desk/poster/large）
  + RGB→CMYK変換（Pillow 内蔵、商業印刷対応）
  + PDF出力（reportlab）
  + DPIメタデータ埋め込み（PNG pHYs チャンク）

成功基準:
  A4 2480×3508px / A3 3508×4961px @ 300DPI
  desk 壁幅 3〜4px / poster 壁幅 8〜11px / large 壁幅 22〜25px
  SSIM ≥ 0.70 維持（DM-4 後方互換）
"""
from __future__ import annotations

import dataclasses
import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .dm4 import DM4Config, DM4Result, generate_dm4_maze


# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

# 印刷フォーマット → (width_px, height_px) @ 300DPI
PRINT_FORMATS: Dict[str, Tuple[int, int]] = {
    "A4": (2480, 3508),   # 210×297mm @ 300DPI
    "A3": (3508, 4961),   # 297×420mm @ 300DPI
}

# viewing_distance → 壁幅 (mm)
_WALL_MM: Dict[str, float] = {
    "desk":   0.3,   # 30cm 観察距離
    "poster": 0.8,   # 1m 観察距離
    "large":  2.0,   # 3m 観察距離
}

# グリッドサイズ上限（パフォーマンス保護）
MAX_GRID_CELLS: int = 150


# ---------------------------------------------------------------------------
# ユーティリティ関数
# ---------------------------------------------------------------------------

def _wall_px(viewing_distance: str, dpi: int = 300) -> int:
    """
    viewing_distance から壁幅（px）を計算する。

    formula: round(wall_mm * dpi / 25.4)
    結果: desk→4px / poster→9px / large→24px (@300DPI)
    """
    mm = _WALL_MM[viewing_distance]
    return max(1, round(mm * dpi / 25.4))


def _resize_to_print(
    png_bytes: bytes,
    width: int,
    height: int,
    dpi: int = 300,
) -> bytes:
    """
    PNG を印刷サイズ（width×height px）に LANCZOS リサイズし、
    DPI メタデータ（pHYs チャンク）を埋め込む。

    Note: PNG の pHYs チャンクは pixels/meter (int) で保存されるため、
    読み返し時に ~299.9994 DPI になる（浮動小数誤差）。
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    resized = img.resize((width, height), Image.LANCZOS)
    buf = io.BytesIO()
    resized.save(buf, format="PNG", dpi=(dpi, dpi))
    return buf.getvalue()


def _to_cmyk_png(png_bytes: bytes) -> bytes:
    """
    PNG を CMYK モードに変換して JPEG bytes を返す（Pillow 内蔵変換）。

    Note: PNG フォーマットは CMYK モードをサポートしないため JPEG で出力する。
    JPEG は CMYK モードを完全サポートしており、商業印刷ワークフローでも
    CMYK JPEG は標準的な入稿形式として広く利用される。
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB").convert("CMYK")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _to_pdf(
    png_bytes: bytes,
    print_format: str = "A4",
    dpi: int = 300,
) -> bytes:
    """
    PNG bytes から印刷用 PDF を生成する（reportlab 使用）。

    Args:
        png_bytes   : 印刷サイズ PNG（_resize_to_print 後の bytes）。
        print_format: "A4" / "A3"
        dpi         : メタデータ用（reportlab 内部では point 単位で処理）。

    Returns:
        PDF bytes（%PDF ヘッダで始まる）。
    """
    from reportlab.lib.pagesizes import A4 as RL_A4, A3 as RL_A3
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.utils import ImageReader

    page_size_map = {"A4": RL_A4, "A3": RL_A3}
    page_w, page_h = page_size_map[print_format]

    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(page_w, page_h))
    img_reader = ImageReader(io.BytesIO(png_bytes))
    c.drawImage(img_reader, 0, 0, width=page_w, height=page_h)
    c.showPage()
    c.save()

    return buf.getvalue()


# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

@dataclass
class DM5Config(DM4Config):
    """
    DM-5 設定。DM4Config を継承し、印刷最適化パラメータを追加。

    主要パラメータ:
        print_format     : "A4" (2480×3508px) / "A3" (3508×4961px) @ 300DPI
        viewing_distance : "desk"(0.3mm壁) / "poster"(0.8mm壁) / "large"(2.0mm壁)
        output_format    : "png" / "cmyk_png" / "pdf"
        dpi              : 出力 DPI（デフォルト 300）

    Note: cell_size_px / grid_rows / grid_cols は generate_dm5_maze() が
    viewing_distance と print_format から自動計算するため、
    ここに設定した値は上書きされる。
    """
    print_format: str = "A4"
    viewing_distance: str = "desk"
    output_format: str = "png"
    dpi: int = 300


# ---------------------------------------------------------------------------
# 結果
# ---------------------------------------------------------------------------

@dataclass
class DM5Result(DM4Result):
    """DM-5 生成結果。DM-4 互換フィールド + 印刷最適化情報。"""
    print_format: str = "A4"
    viewing_distance: str = "desk"
    dpi: int = 300
    wall_thickness_px: int = 0
    pdf_bytes: Optional[bytes] = None
    cmyk_png_bytes: Optional[bytes] = None


# ---------------------------------------------------------------------------
# メイン API
# ---------------------------------------------------------------------------

def generate_dm5_maze(
    image: Image.Image,
    config: Optional[DM5Config] = None,
) -> DM5Result:
    """
    DM-5 印刷最適化迷路を生成する。

    DM-4 パイプラインで迷路を生成し、A4/A3 @ 300DPI にリサイズして
    DPI メタデータを埋め込む。output_format に応じて CMYK 変換または
    PDF 出力も行う。

    Args:
        image : 入力画像（任意サイズ・任意形式）。
        config: DM5Config。None の場合はデフォルト値を使用。

    Returns:
        DM5Result（DM4Result 互換フィールド + 印刷最適化情報）

    Raises:
        KeyError: 無効な print_format / viewing_distance が指定された場合。
    """
    if config is None:
        config = DM5Config()

    # ------------------------------------------------------------------
    # 印刷サイズ・壁幅の計算
    # ------------------------------------------------------------------
    canvas_w, canvas_h = PRINT_FORMATS[config.print_format]
    wall_px = _wall_px(config.viewing_distance, config.dpi)
    cell_size_px = wall_px * 2

    target_grid_cols = min(canvas_w // cell_size_px, MAX_GRID_CELLS)
    target_grid_rows = min(canvas_h // cell_size_px, MAX_GRID_CELLS)

    # ------------------------------------------------------------------
    # DM-4 実行用設定を構築（viewing_distance/印刷パラメータで上書き）
    # dataclasses.replace() で DM5Config フィールドを保持しつつ上書き
    # ------------------------------------------------------------------
    dm4_config = dataclasses.replace(
        config,
        cell_size_px=cell_size_px,
        grid_rows=target_grid_rows,
        grid_cols=target_grid_cols,
    )

    # ------------------------------------------------------------------
    # DM-4 パイプライン実行
    # ------------------------------------------------------------------
    dm4_result = generate_dm4_maze(image, dm4_config)

    # ------------------------------------------------------------------
    # 印刷サイズにリサイズ + DPI メタデータ埋め込み
    # ------------------------------------------------------------------
    print_png = _resize_to_print(dm4_result.png_bytes, canvas_w, canvas_h, config.dpi)

    # ------------------------------------------------------------------
    # 出力フォーマット変換
    # ------------------------------------------------------------------
    pdf_bytes: Optional[bytes] = None
    cmyk_png_bytes: Optional[bytes] = None

    if config.output_format == "pdf":
        pdf_bytes = _to_pdf(print_png, config.print_format, config.dpi)
    elif config.output_format == "cmyk_png":
        cmyk_png_bytes = _to_cmyk_png(print_png)

    return DM5Result(
        # DM-4 継承フィールド
        svg=dm4_result.svg,
        png_bytes=print_png,
        entrance=dm4_result.entrance,
        exit_cell=dm4_result.exit_cell,
        solution_path=dm4_result.solution_path,
        grid_rows=dm4_result.grid_rows,
        grid_cols=dm4_result.grid_cols,
        density_map=dm4_result.density_map,
        adj=dm4_result.adj,
        edge_map=dm4_result.edge_map,
        solution_count=dm4_result.solution_count,
        clahe_clip_limit_used=dm4_result.clahe_clip_limit_used,
        clahe_n_tiles_used=dm4_result.clahe_n_tiles_used,
        ssim_score=dm4_result.ssim_score,
        dark_coverage=dm4_result.dark_coverage,
        # DM-5 追加フィールド
        print_format=config.print_format,
        viewing_distance=config.viewing_distance,
        dpi=config.dpi,
        wall_thickness_px=wall_px,
        pdf_bytes=pdf_bytes,
        cmyk_png_bytes=cmyk_png_bytes,
    )
