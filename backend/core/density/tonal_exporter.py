"""
DM-4: 多値トーン壁表現 (Tonal Exporter)

8段階グレースケール壁により SSIM 天井（~0.63）を突破し、0.70+ を目指す。

設計書: docs/maze-artisan-masterpiece-requirements.md Phase DM-4

原理:
  現行の二値壁（色域 [40, 175]）は平均輝度の再現精度が限定的。
  DM-4 では壁色を元画像局所輝度に基づく 8 段階（[0..255] 全域）に量子化し、
  画素輝度分布を原画像に近づけ SSIM を改善する。

  暗部（avg_lum≈0）: grade=0 (黒)  → 濃い壁 → 暗部ピクセルが増加
  明部（avg_lum≈1）: grade=255 (白) → 壁が背景と同化 → 明部ピクセルが白に近くなる

8 段階グレースケール:
  grades = [0, 36, 73, 109, 146, 182, 219, 255]

アンチエイリアス:
  render_scale 倍サイズで描画し、LANCZOS リサンプリングで縮小 → ジャギー低減。

壁厚変動:
  thickness_range > 0 のとき、暗部の壁を太く描画 → 暗部の黒ピクセル密度増加 → SSIM 向上。

通路: 常に白 (255)。
"""
from __future__ import annotations

import io
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .grid_builder import CellGrid

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

TONAL_GRADES: List[int] = [0, 36, 73, 109, 146, 182, 219, 255]
"""8 段階グレースケール値（昇順）。DM-4 の壁色パレット。"""


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _quantize_lum_to_grade(lum: float, grades: List[int] = TONAL_GRADES) -> int:
    """
    avg_lum [0, 1] を grades の最近傍要素に量子化する。

    Args:
        lum   : 0.0（黒）〜 1.0（白）の輝度値。
        grades: 量子化先のグレー値リスト（昇順）。

    Returns:
        grades の要素（int, 0-255）。

    Examples:
        >>> _quantize_lum_to_grade(0.0)   # → 0
        >>> _quantize_lum_to_grade(1.0)   # → 255
        >>> _quantize_lum_to_grade(0.5)   # → 146
    """
    n = len(grades)
    idx = int(round(float(lum) * (n - 1)))
    idx = max(0, min(n - 1, idx))
    return grades[idx]


def compute_dark_coverage(png_bytes: bytes, threshold: int = 128) -> float:
    """
    PNG バイト列中の「暗い」ピクセル（< threshold）の割合を返す。

    Args:
        png_bytes : PNG 画像バイト列。
        threshold : 暗部閾値 (0-255)。デフォルト 128。

    Returns:
        0.0〜1.0。threshold 未満のピクセル割合。
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("L")
    arr = np.asarray(img, dtype=np.uint8)
    return float(np.mean(arr < threshold))


# ---------------------------------------------------------------------------
# PNG レンダリング（アンチエイリアス付き）
# ---------------------------------------------------------------------------

def maze_to_png_tonal(
    grid: CellGrid,
    adj: Dict[int, List[int]],
    entrance: int,
    exit_id: int,
    solution_path: List[int],
    width: int = 800,
    height: int = 600,
    show_solution: bool = False,
    render_scale: int = 2,
    grades: Optional[List[int]] = None,
    wall_thickness_base: float = 1.5,
    thickness_range: float = 0.5,
    fill_cells: bool = True,
    blur_radius: float = 2.0,
) -> bytes:
    """
    8 段階グレースケール壁で PNG 迷路を描画する（アンチエイリアス付き）。

    fill_cells=True（デフォルト）の場合:
      セルを 8 段階グレーで塗りつぶし、白い通路（パッセージ）を彫り込む方式。
      Gaussian blur を適用することで元画像の輝度勾配を忠実に再現し、SSIM を向上させる。
      gradient SSIM ≥ 0.70 を達成するための主要手法。

    fill_cells=False の場合:
      白背景に 8 段階グレーの壁線を描画する従来方式（DM-2 互換ライン描画）。

    Args:
        grid              : CellGrid（luminance マップ含む）。
        adj               : 隣接リスト（除去済み壁 = 通路）。
        entrance, exit_id : 入口・出口セル ID（描画では未使用、インターフェース統一のため保持）。
        solution_path     : 解経路セル ID リスト。
        width, height     : 出力画像サイズ（ピクセル）。
        show_solution     : True のとき解経路を白線で描画（masterpiece モード）。
        render_scale      : アンチエイリアス倍率（1=なし, 2=2倍描画→縮小）。
        grades            : グレー値パレット（デフォルト: TONAL_GRADES）。
        wall_thickness_base: 基本壁厚（fill_cells=False 時のみ使用）。
        thickness_range   : 壁厚変動範囲（fill_cells=False 時のみ使用）。
        fill_cells        : True=セル塗りつぶし+通路彫り込み（高 SSIM）, False=壁線描画。
        blur_radius       : Gaussian blur 半径（fill_cells=True 時に適用）。0 で無効。

    Returns:
        PNG バイト列（L モード = グレースケール）。
    """
    if grades is None:
        grades = TONAL_GRADES

    # 超解像キャンバスサイズ
    sw = width * render_scale
    sh = height * render_scale

    # 除去済み壁（通路）のセット
    removed: set = set()
    for u, neighbors in adj.items():
        for v in neighbors:
            removed.add((min(u, v), max(u, v)))

    if fill_cells:
        # ------------------------------------------------------------------
        # fill_cells モード: セル塗りつぶし + 白通路彫り込み + Gaussian blur
        # マージンなしで全面を使い輝度分布の再現精度を最大化する。
        # ------------------------------------------------------------------
        cs_x = sw / grid.cols
        cs_y = sh / grid.rows
        cell_size = min(cs_x, cs_y)
        x_offsets = np.arange(grid.cols + 1, dtype=np.float64) * cell_size
        y_offsets = np.arange(grid.rows + 1, dtype=np.float64) * cell_size

        img = Image.new("L", (sw, sh), 255)
        draw = ImageDraw.Draw(img)

        # セルをグレードで塗りつぶし
        for r in range(grid.rows):
            for c in range(grid.cols):
                grade = _quantize_lum_to_grade(float(grid.luminance[r, c]), grades)
                x0 = int(x_offsets[c])
                y0 = int(y_offsets[r])
                x1 = int(x_offsets[c + 1])
                y1 = int(y_offsets[r + 1])
                draw.rectangle([x0, y0, x1, y1], fill=grade)

        # 通路部分を白(255)で彫り込む
        passage_width = max(render_scale, int(cell_size * 0.5))
        hw = passage_width // 2

        for r in range(grid.rows):
            for c in range(grid.cols):
                cid = grid.cell_id(r, c)
                # 右通路（垂直）
                if c + 1 < grid.cols:
                    cid2 = grid.cell_id(r, c + 1)
                    if (min(cid, cid2), max(cid, cid2)) in removed:
                        x1 = int(x_offsets[c + 1])
                        y0 = int(y_offsets[r])
                        y1 = int(y_offsets[r + 1])
                        draw.rectangle([x1 - hw, y0 + hw, x1 + hw, y1 - hw], fill=255)
                # 下通路（水平）
                if r + 1 < grid.rows:
                    cid2 = grid.cell_id(r + 1, c)
                    if (min(cid, cid2), max(cid, cid2)) in removed:
                        x0 = int(x_offsets[c])
                        x1 = int(x_offsets[c + 1])
                        y1 = int(y_offsets[r + 1])
                        draw.rectangle([x0 + hw, y1 - hw, x1 - hw, y1 + hw], fill=255)

        # LANCZOS アンチエイリアス縮小
        if render_scale > 1:
            img = img.resize((width, height), Image.LANCZOS)

        # Gaussian blur で量子化アーチファクトを平滑化
        if blur_radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    else:
        # ------------------------------------------------------------------
        # 壁線描画モード（従来方式、fill_cells=False）:
        # 白背景に 8 段階グレーの壁を線として描画する。
        # ------------------------------------------------------------------
        margin = 20 * render_scale
        w_inner = sw - 2 * margin
        h_inner = sh - 2 * margin

        cs_x = w_inner / grid.cols
        cs_y = h_inner / grid.rows
        cell_size = min(cs_x, cs_y)

        x_offsets = np.arange(grid.cols + 1, dtype=np.float64) * cell_size
        y_offsets = np.arange(grid.rows + 1, dtype=np.float64) * cell_size

        img = Image.new("L", (sw, sh), 255)
        draw = ImageDraw.Draw(img)

        for r in range(grid.rows):
            for c in range(grid.cols):
                cid = grid.cell_id(r, c)
                x0 = int(margin + x_offsets[c])
                y0 = int(margin + y_offsets[r])

                # 右壁（垂直線）
                if c + 1 < grid.cols:
                    cid2 = grid.cell_id(r, c + 1)
                    if (min(cid, cid2), max(cid, cid2)) not in removed:
                        avg_lum = float(
                            (grid.luminance[r, c] + grid.luminance[r, c + 1]) / 2.0
                        )
                        gray_val = _quantize_lum_to_grade(avg_lum, grades)
                        sw_px = max(
                            render_scale,
                            round(
                                (wall_thickness_base + thickness_range * (1.0 - avg_lum))
                                * render_scale
                            ),
                        )
                        x1 = int(margin + x_offsets[c + 1])
                        y1 = int(margin + y_offsets[r + 1])
                        draw.line([(x1, y0), (x1, y1)], fill=gray_val, width=sw_px)

                # 下壁（水平線）
                if r + 1 < grid.rows:
                    cid2 = grid.cell_id(r + 1, c)
                    if (min(cid, cid2), max(cid, cid2)) not in removed:
                        avg_lum = float(
                            (grid.luminance[r, c] + grid.luminance[r + 1, c]) / 2.0
                        )
                        gray_val = _quantize_lum_to_grade(avg_lum, grades)
                        sw_px = max(
                            render_scale,
                            round(
                                (wall_thickness_base + thickness_range * (1.0 - avg_lum))
                                * render_scale
                            ),
                        )
                        x1 = int(margin + x_offsets[c + 1])
                        y1 = int(margin + y_offsets[r + 1])
                        draw.line([(x0, y1), (x1, y1)], fill=gray_val, width=sw_px)

        # 解経路（白線オーバーレイ: masterpiece モード）
        if show_solution and solution_path:
            corridor_px = max(2 * render_scale, int(cell_size * 0.85))
            pts = []
            for cid in solution_path:
                r2, c2 = grid.cell_rc(cid)
                x = int(margin + x_offsets[c2] + cell_size / 2.0)
                y = int(margin + y_offsets[r2] + cell_size / 2.0)
                pts.append((x, y))
            for i in range(len(pts) - 1):
                draw.line([pts[i], pts[i + 1]], fill=255, width=corridor_px)

        # LANCZOS アンチエイリアス縮小
        if render_scale > 1:
            img = img.resize((width, height), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# SVG レンダリング（8 段階グレースケール）
# ---------------------------------------------------------------------------

def maze_to_svg_tonal(
    grid: CellGrid,
    adj: Dict[int, List[int]],
    entrance: int,
    exit_id: int,
    solution_path: List[int],
    width: int = 800,
    height: int = 600,
    show_solution: bool = False,
    grades: Optional[List[int]] = None,
    wall_thickness_base: float = 1.5,
    thickness_range: float = 0.5,
) -> str:
    """
    8 段階グレースケール壁で SVG 迷路を描画する。

    壁を (color, stroke-width) キーでグループ化し
    <g stroke="..."><path d="..."/></g> 形式で出力。

    Args:
        grades            : グレー値パレット（デフォルト: TONAL_GRADES）。
        wall_thickness_base: 基本壁厚（px）。
        thickness_range   : 暗部での追加壁厚比率。

    Returns:
        SVG 文字列。
    """
    if grades is None:
        grades = TONAL_GRADES

    margin = 20.0
    w = width - 2 * margin
    h = height - 2 * margin
    cs_x = w / grid.cols
    cs_y = h / grid.rows
    cell_size = min(cs_x, cs_y)

    x_offsets = np.arange(grid.cols + 1, dtype=np.float64) * cell_size
    y_offsets = np.arange(grid.rows + 1, dtype=np.float64) * cell_size

    removed: set = set()
    for u, neighbors in adj.items():
        for v in neighbors:
            removed.add((min(u, v), max(u, v)))

    wall_cmds: Dict[tuple, List[str]] = defaultdict(list)

    for r in range(grid.rows):
        for c in range(grid.cols):
            cid = grid.cell_id(r, c)
            x0 = margin + x_offsets[c]
            y0 = margin + y_offsets[r]

            # 右壁
            if c + 1 < grid.cols:
                cid2 = grid.cell_id(r, c + 1)
                if (min(cid, cid2), max(cid, cid2)) not in removed:
                    avg_lum = float(
                        (grid.luminance[r, c] + grid.luminance[r, c + 1]) / 2.0
                    )
                    gray = _quantize_lum_to_grade(avg_lum, grades)
                    sw = wall_thickness_base + thickness_range * (1.0 - avg_lum)
                    color = f"rgb({gray},{gray},{gray})"
                    x1 = margin + x_offsets[c + 1]
                    y1 = margin + y_offsets[r + 1]
                    wall_cmds[(color, f"{sw:.2f}")].append(
                        f"M{x1:.1f} {y0:.1f}V{y1:.1f}"
                    )

            # 下壁
            if r + 1 < grid.rows:
                cid2 = grid.cell_id(r + 1, c)
                if (min(cid, cid2), max(cid, cid2)) not in removed:
                    avg_lum = float(
                        (grid.luminance[r, c] + grid.luminance[r + 1, c]) / 2.0
                    )
                    gray = _quantize_lum_to_grade(avg_lum, grades)
                    sw = wall_thickness_base + thickness_range * (1.0 - avg_lum)
                    color = f"rgb({gray},{gray},{gray})"
                    x1 = margin + x_offsets[c + 1]
                    y1 = margin + y_offsets[r + 1]
                    wall_cmds[(color, f"{sw:.2f}")].append(
                        f"M{x0:.1f} {y1:.1f}H{x1:.1f}"
                    )

    parts: List[str] = []
    # 外枠
    parts.append(
        f'<rect x="{margin:.1f}" y="{margin:.1f}" '
        f'width="{x_offsets[-1]:.1f}" height="{y_offsets[-1]:.1f}" '
        f'fill="none" stroke="black" stroke-width="2"/>'
    )

    for (color, sw_key) in sorted(wall_cmds.keys()):
        d = " ".join(wall_cmds[(color, sw_key)])
        parts.append(
            f'<g stroke="{color}" stroke-width="{sw_key}">'
            f'<path d="{d}" fill="none"/>'
            f'</g>'
        )

    # 解経路（白線）
    if show_solution and solution_path:
        path_d = []
        for i, cid in enumerate(solution_path):
            r2, c2 = grid.cell_rc(cid)
            x = margin + x_offsets[c2] + cell_size / 2.0
            y = margin + y_offsets[r2] + cell_size / 2.0
            if i == 0:
                path_d.append(f"M{x:.1f} {y:.1f}")
            else:
                path_d.append(f"L{x:.1f} {y:.1f}")
        corridor_w = max(2.0, cell_size * 0.85)
        parts.append(
            f'<path d="{" ".join(path_d)}" fill="none" stroke="white" '
            f'stroke-width="{corridor_w:.2f}" stroke-linecap="round" '
            f'stroke-linejoin="round"/>'
        )

    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n'
        + "\n".join(parts)
        + "\n</svg>"
    )
