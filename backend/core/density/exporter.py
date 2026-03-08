"""
密度迷路 Phase 1/2: 迷路グラフ → SVG / PNG。
直角グリッドで壁を描画し、解経路をオプションで表示。
Phase 2: 廊下風の解経路描画（太線 + 丸キャップ）と入口・出口マーカーを追加。
Phase 2b: solution_highlight=False（デフォルト）で白線塗りつぶしモード。
          解経路が白く浮かぶ = masterpiece「白い道を塗りつぶした完成形」。
"""
from __future__ import annotations

import io
from typing import Dict, List, Optional

from PIL import Image, ImageDraw

from .grid_builder import CellGrid
from .maze_builder import build_spanning_tree


def _cell_center(grid: CellGrid, cell_id: int, cell_size: float, margin: float) -> tuple[float, float]:
    r, c = grid.cell_rc(cell_id)
    x = margin + (c + 0.5) * cell_size
    y = margin + (r + 0.5) * cell_size
    return x, y


def _wall_stroke(stroke_width_base: float, avg_lum: float, thickness_range: float) -> float:
    """
    Xu & Kaplan (SIGGRAPH 2007) 濃淡公式:
      G = (S - W) / S  →  暗い部分は W（壁厚）を太く、明るい部分は細く。
    wall_thickness = stroke_width_base * (1.0 + thickness_range * (1.0 - avg_lum))
    avg_lum=0（黒）: stroke_width_base * (1.0 + thickness_range)  → 最大
    avg_lum=1（白）: stroke_width_base * 1.0                       → 最小
    """
    return stroke_width_base * (1.0 + thickness_range * (1.0 - avg_lum))


def maze_to_svg(
    grid: CellGrid,
    adj: Dict[int, List[int]],
    entrance: int,
    exit_id: int,
    solution_path: List[int],
    width: int = 800,
    height: int = 600,
    stroke_width: float = 2.0,
    show_solution: bool = True,
    thickness_range: float = 1.5,
    solution_highlight: bool = False,
) -> str:
    """
    迷路を SVG で描画。セルは矩形、隣接かつ壁除去済みでない境界に線を引く。
    thickness_range > 0 のとき、隣接2セルの平均輝度に応じて壁厚を変える（可変壁厚）。

    解経路の描画モード（show_solution=True 時）:
      solution_highlight=False (デフォルト / masterpiece モード):
        白線（壁色と同化）で太く描画。経路部分が白く浮かぶ。
        corridor_width = cell_size × 0.85。
      solution_highlight=True (デバッグ/プレビュー):
        オレンジ廊下 + 入口(緑丸)・出口(赤丸)マーカー。
    """
    margin = 20.0
    w = width - 2 * margin
    h = height - 2 * margin
    cs_x = w / grid.cols
    cs_y = h / grid.rows
    cell_size = min(cs_x, cs_y)

    lines: List[str] = []
    # 外枠（ベース stroke_width 固定）
    lines.append(f'<rect x="{margin}" y="{margin}" width="{grid.cols * cell_size}" height="{grid.rows * cell_size}" fill="none" stroke="black" stroke-width="{stroke_width}"/>')

    removed = set()
    for u, neighbors in adj.items():
        for v in neighbors:
            removed.add((min(u, v), max(u, v)))

    # 各セルの右・下の壁（境界）を描画（除去されていなければ線を引く）
    for r in range(grid.rows):
        for c in range(grid.cols):
            cid = grid.cell_id(r, c)
            x0 = margin + c * cell_size
            y0 = margin + r * cell_size
            # 右壁
            if c + 1 < grid.cols:
                cid2 = grid.cell_id(r, c + 1)
                if (min(cid, cid2), max(cid, cid2)) not in removed:
                    avg_lum = float((grid.luminance[r, c] + grid.luminance[r, c + 1]) / 2.0)
                    sw = _wall_stroke(stroke_width, avg_lum, thickness_range)
                    lines.append(f'<line x1="{x0 + cell_size}" y1="{y0}" x2="{x0 + cell_size}" y2="{y0 + cell_size}" stroke="black" stroke-width="{sw:.3f}"/>')
            # 下壁
            if r + 1 < grid.rows:
                cid2 = grid.cell_id(r + 1, c)
                if (min(cid, cid2), max(cid, cid2)) not in removed:
                    avg_lum = float((grid.luminance[r, c] + grid.luminance[r + 1, c]) / 2.0)
                    sw = _wall_stroke(stroke_width, avg_lum, thickness_range)
                    lines.append(f'<line x1="{x0}" y1="{y0 + cell_size}" x2="{x0 + cell_size}" y2="{y0 + cell_size}" stroke="black" stroke-width="{sw:.3f}"/>')

    if show_solution and solution_path:
        path_d = []
        for i, cid in enumerate(solution_path):
            x, y = _cell_center(grid, cid, cell_size, margin)
            if i == 0:
                path_d.append(f"M {x:.2f} {y:.2f}")
            else:
                path_d.append(f"L {x:.2f} {y:.2f}")

        if solution_highlight:
            # デバッグ/プレビューモード: オレンジ廊下 + 入口・出口マーカー
            corridor_w = max(stroke_width * 1.5, cell_size * 0.40)
            lines.append(
                f'<path d="{" ".join(path_d)}" fill="none" stroke="#E05000" '
                f'stroke-width="{corridor_w:.2f}" stroke-linecap="round" '
                f'stroke-linejoin="round" opacity="0.65"/>'
            )
            ex, ey = _cell_center(grid, solution_path[0], cell_size, margin)
            r_marker = max(cell_size * 0.3, stroke_width)
            lines.append(
                f'<circle cx="{ex:.2f}" cy="{ey:.2f}" r="{r_marker:.2f}" '
                f'fill="#00AA44" opacity="0.85"/>'
            )
            if len(solution_path) > 1:
                gx_pt, gy_pt = _cell_center(grid, solution_path[-1], cell_size, margin)
                lines.append(
                    f'<circle cx="{gx_pt:.2f}" cy="{gy_pt:.2f}" r="{r_marker:.2f}" '
                    f'fill="#CC2222" opacity="0.85"/>'
                )
        else:
            # masterpiece モード: 白線で経路を塗りつぶし（経路部分が白く浮かぶ）
            corridor_w = max(stroke_width * 2.0, cell_size * 0.85)
            lines.append(
                f'<path d="{" ".join(path_d)}" fill="none" stroke="white" '
                f'stroke-width="{corridor_w:.2f}" stroke-linecap="round" '
                f'stroke-linejoin="round"/>'
            )

    return f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n' + "\n".join(lines) + "\n</svg>"


def maze_to_png(
    grid: CellGrid,
    adj: Dict[int, List[int]],
    entrance: int,
    exit_id: int,
    solution_path: List[int],
    width: int = 800,
    height: int = 600,
    stroke_width: int = 2,
    show_solution: bool = True,
    thickness_range: float = 1.5,
    solution_highlight: bool = False,
) -> bytes:
    """迷路を PNG バイト列で出力。thickness_range > 0 で可変壁厚を適用。

    解経路の描画モード（show_solution=True 時）:
      solution_highlight=False (デフォルト / masterpiece モード):
        白線（(255,255,255)）で太く描画。corridor_width = cell_size × 0.85。
      solution_highlight=True (デバッグ/プレビュー):
        オレンジ廊下 + 入口(緑丸)・出口(赤丸)マーカー。
    """
    margin = 20
    w = width - 2 * margin
    h = height - 2 * margin
    cs_x = w / grid.cols
    cs_y = h / grid.rows
    cell_size = min(cs_x, cs_y)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    removed = set()
    for u, neighbors in adj.items():
        for v in neighbors:
            removed.add((min(u, v), max(u, v)))

    def to_px(x: float, y: float) -> tuple[int, int]:
        return int(margin + x * cell_size), int(margin + y * cell_size)

    for r in range(grid.rows):
        for c in range(grid.cols):
            cid = grid.cell_id(r, c)
            # 右壁
            if c + 1 < grid.cols:
                cid2 = grid.cell_id(r, c + 1)
                if (min(cid, cid2), max(cid, cid2)) not in removed:
                    avg_lum = float((grid.luminance[r, c] + grid.luminance[r, c + 1]) / 2.0)
                    sw = max(1, round(_wall_stroke(stroke_width, avg_lum, thickness_range)))
                    a = to_px(c + 1, r)
                    b = to_px(c + 1, r + 1)
                    draw.line([a, b], fill="black", width=sw)
            # 下壁
            if r + 1 < grid.rows:
                cid2 = grid.cell_id(r + 1, c)
                if (min(cid, cid2), max(cid, cid2)) not in removed:
                    avg_lum = float((grid.luminance[r, c] + grid.luminance[r + 1, c]) / 2.0)
                    sw = max(1, round(_wall_stroke(stroke_width, avg_lum, thickness_range)))
                    a = to_px(c, r + 1)
                    b = to_px(c + 1, r + 1)
                    draw.line([a, b], fill="black", width=sw)

    if show_solution and solution_path:
        pts = [_cell_center(grid, cid, cell_size, margin) for cid in solution_path]
        px_pts = [(int(x), int(y)) for x, y in pts]

        if solution_highlight:
            # デバッグ/プレビューモード: オレンジ廊下 + 入口・出口マーカー
            corridor_w = max(stroke_width + 2, int(cell_size * 0.40))
            for i in range(len(px_pts) - 1):
                draw.line([px_pts[i], px_pts[i + 1]], fill=(224, 80, 0), width=corridor_w)
            r_m = max(int(cell_size * 0.30), 3)
            ex, ey = px_pts[0]
            draw.ellipse([ex - r_m, ey - r_m, ex + r_m, ey + r_m], fill=(0, 170, 68))
            if len(px_pts) > 1:
                gx_p, gy_p = px_pts[-1]
                draw.ellipse([gx_p - r_m, gy_p - r_m, gx_p + r_m, gy_p + r_m], fill=(204, 34, 34))
        else:
            # masterpiece モード: 白線で経路を塗りつぶし（経路部分が白く浮かぶ）
            corridor_w = max(stroke_width + 2, int(cell_size * 0.85))
            for i in range(len(px_pts) - 1):
                draw.line([px_pts[i], px_pts[i + 1]], fill=(255, 255, 255), width=corridor_w)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
