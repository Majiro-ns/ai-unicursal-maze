"""
密度迷路 Phase 1/2: 迷路グラフ → SVG / PNG。
直角グリッドで壁を描画し、解経路をオプションで表示。
Phase 2: 廊下風の解経路描画（太線 + 丸キャップ）と入口・出口マーカーを追加。
Phase 2b: solution_highlight=False（デフォルト）で白線塗りつぶしモード。
          解経路が白く浮かぶ = masterpiece「白い道を塗りつぶした完成形」。
Phase 3 SVG品質改善:
  - stroke_quantize_levels で壁厚を離散化し <g> グループ化 → ファイルサイズ削減
  - wall_thickness_histogram() で壁厚分布を可視化
  - maze_to_png() に dpi パラメータを追加
"""
from __future__ import annotations

import io
from collections import defaultdict
from typing import Dict, List, Optional

from PIL import Image, ImageDraw

from .grid_builder import CellGrid


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


def _wall_color(avg_lum: float) -> str:
    """
    壁の描画色をグレースケールで返す（SVG stroke / PNG fill 共通）。
      avg_lum=0（黒画素）→ rgb(0,0,0)     = 黒（最大コントラスト）
      avg_lum=1（白画素）→ rgb(220,220,220)= 淡灰（白背景に消えない範囲で最淡）
    完全白(255)にしないのは白背景との区別がつかなくなるため。
    公式: v = int(avg_lum * 220)
    """
    v = int(avg_lum * 220)
    return f"rgb({v},{v},{v})"


def wall_thickness_histogram(
    grid: CellGrid,
    adj: Dict[int, List[int]],
    stroke_width: float = 2.0,
    thickness_range: float = 1.5,
    n_bins: int = 10,
    print_chart: bool = True,
) -> dict:
    """
    壁厚分布のヒストグラムを計算し、オプションで ASCII 表示する。

    Returns:
        {
            "total": int,         # 描画された壁の総数
            "min": float,         # 最小壁厚
            "max": float,         # 最大壁厚
            "bins": List[float],  # ビン境界 (len = n_bins + 1)
            "counts": List[int],  # 各ビンの壁数 (len = n_bins)
        }
    """
    removed: set = set()
    for u, neighbors in adj.items():
        for v in neighbors:
            removed.add((min(u, v), max(u, v)))

    widths: List[float] = []
    for r in range(grid.rows):
        for c in range(grid.cols):
            cid = grid.cell_id(r, c)
            if c + 1 < grid.cols:
                cid2 = grid.cell_id(r, c + 1)
                if (min(cid, cid2), max(cid, cid2)) not in removed:
                    avg_lum = float((grid.luminance[r, c] + grid.luminance[r, c + 1]) / 2.0)
                    widths.append(_wall_stroke(stroke_width, avg_lum, thickness_range))
            if r + 1 < grid.rows:
                cid2 = grid.cell_id(r + 1, c)
                if (min(cid, cid2), max(cid, cid2)) not in removed:
                    avg_lum = float((grid.luminance[r, c] + grid.luminance[r + 1, c]) / 2.0)
                    widths.append(_wall_stroke(stroke_width, avg_lum, thickness_range))

    if not widths:
        return {"total": 0, "min": 0.0, "max": 0.0, "bins": [], "counts": []}

    min_w = min(widths)
    max_w = max(widths)

    if max_w == min_w:
        # 全壁が同一幅（均一画像 or thickness_range=0）
        bins = [min_w, min_w]
        counts = [len(widths)]
    else:
        bin_size = (max_w - min_w) / n_bins
        bins = [min_w + i * bin_size for i in range(n_bins + 1)]
        counts = [0] * n_bins
        for w in widths:
            idx = min(int((w - min_w) / bin_size), n_bins - 1)
            counts[idx] += 1

    if print_chart:
        max_count = max(counts) if counts else 1
        bar_max = 40
        print(
            f"壁厚分布ヒストグラム "
            f"(total={len(widths)} walls, stroke={stroke_width}, range={thickness_range})"
        )
        print(f"  min={min_w:.3f}, max={max_w:.3f}")
        for lo, hi, cnt in zip(bins[:-1], bins[1:], counts):
            bar = "#" * int(cnt / max_count * bar_max)
            print(f"  [{lo:.3f}-{hi:.3f}] {bar:<{bar_max}} {cnt}")

    return {
        "total": len(widths),
        "min": min_w,
        "max": max_w,
        "bins": list(bins),
        "counts": list(counts),
    }


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
    stroke_quantize_levels: int = 20,
) -> str:
    """
    迷路を SVG で描画。セルは矩形、隣接かつ壁除去済みでない境界に線を引く。
    thickness_range > 0 のとき、隣接2セルの平均輝度に応じて壁厚を変える（可変壁厚）。

    SVG最適化（Phase 3）:
      壁を stroke-width でグループ化し <g stroke-width="..."><path d="..."/></g> 形式で出力。
      M/V/H コマンドで座標を簡略化。座標精度 .1f。
      stroke_quantize_levels > 0 のとき avg_lum を N 段階に量子化してグループ数を削減。

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

    parts: List[str] = []

    # 外枠（固定 stroke_width）
    parts.append(
        f'<rect x="{margin:.1f}" y="{margin:.1f}" '
        f'width="{grid.cols * cell_size:.1f}" height="{grid.rows * cell_size:.1f}" '
        f'fill="none" stroke="black" stroke-width="{stroke_width}"/>'
    )

    removed: set = set()
    for u, neighbors in adj.items():
        for v in neighbors:
            removed.add((min(u, v), max(u, v)))

    # 壁を (color, stroke-width) キーでグループ化（M/V/H コマンドで座標簡略化）
    wall_cmds: Dict[tuple, List[str]] = defaultdict(list)

    for r in range(grid.rows):
        for c in range(grid.cols):
            cid = grid.cell_id(r, c)
            x0 = margin + c * cell_size
            y0 = margin + r * cell_size

            # 右壁（垂直線: x1=x0+cs, y: y0→y0+cs）
            if c + 1 < grid.cols:
                cid2 = grid.cell_id(r, c + 1)
                if (min(cid, cid2), max(cid, cid2)) not in removed:
                    avg_lum = float((grid.luminance[r, c] + grid.luminance[r, c + 1]) / 2.0)
                    if thickness_range > 0 and stroke_quantize_levels > 0:
                        q_lum = round(avg_lum * stroke_quantize_levels) / stroke_quantize_levels
                        sw = _wall_stroke(stroke_width, q_lum, thickness_range)
                        color = _wall_color(q_lum)
                    else:
                        sw = _wall_stroke(stroke_width, avg_lum, thickness_range)
                        color = _wall_color(avg_lum)
                    x1 = x0 + cell_size
                    wall_cmds[(color, f"{sw:.3f}")].append(
                        f"M{x1:.1f} {y0:.1f}V{y0 + cell_size:.1f}"
                    )

            # 下壁（水平線: y1=y0+cs, x: x0→x0+cs）
            if r + 1 < grid.rows:
                cid2 = grid.cell_id(r + 1, c)
                if (min(cid, cid2), max(cid, cid2)) not in removed:
                    avg_lum = float((grid.luminance[r, c] + grid.luminance[r + 1, c]) / 2.0)
                    if thickness_range > 0 and stroke_quantize_levels > 0:
                        q_lum = round(avg_lum * stroke_quantize_levels) / stroke_quantize_levels
                        sw = _wall_stroke(stroke_width, q_lum, thickness_range)
                        color = _wall_color(q_lum)
                    else:
                        sw = _wall_stroke(stroke_width, avg_lum, thickness_range)
                        color = _wall_color(avg_lum)
                    y1 = y0 + cell_size
                    wall_cmds[(color, f"{sw:.3f}")].append(
                        f"M{x0:.1f} {y1:.1f}H{x0 + cell_size:.1f}"
                    )

    # (color, stroke-width) 順に <g> グループとして出力
    for (color, sw_key) in sorted(wall_cmds.keys()):
        d = " ".join(wall_cmds[(color, sw_key)])
        parts.append(
            f'<g stroke="{color}" stroke-width="{sw_key}">'
            f'<path d="{d}" fill="none"/>'
            f'</g>'
        )

    # 解経路（座標精度 .1f）
    if show_solution and solution_path:
        path_d = []
        for i, cid in enumerate(solution_path):
            x, y = _cell_center(grid, cid, cell_size, margin)
            if i == 0:
                path_d.append(f"M{x:.1f} {y:.1f}")
            else:
                path_d.append(f"L{x:.1f} {y:.1f}")

        if solution_highlight:
            # デバッグ/プレビューモード: オレンジ廊下 + 入口・出口マーカー
            corridor_w = max(stroke_width * 1.5, cell_size * 0.40)
            parts.append(
                f'<path d="{" ".join(path_d)}" fill="none" stroke="#E05000" '
                f'stroke-width="{corridor_w:.2f}" stroke-linecap="round" '
                f'stroke-linejoin="round" opacity="0.65"/>'
            )
            ex, ey = _cell_center(grid, solution_path[0], cell_size, margin)
            r_marker = max(cell_size * 0.3, stroke_width)
            parts.append(
                f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="{r_marker:.2f}" '
                f'fill="#00AA44" opacity="0.85"/>'
            )
            if len(solution_path) > 1:
                gx_pt, gy_pt = _cell_center(grid, solution_path[-1], cell_size, margin)
                parts.append(
                    f'<circle cx="{gx_pt:.1f}" cy="{gy_pt:.1f}" r="{r_marker:.2f}" '
                    f'fill="#CC2222" opacity="0.85"/>'
                )
        else:
            # masterpiece モード: 白線で経路を塗りつぶし（経路部分が白く浮かぶ）
            corridor_w = max(stroke_width * 2.0, cell_size * 0.85)
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
    dpi: Optional[int] = None,
) -> bytes:
    """迷路を PNG バイト列で出力。thickness_range > 0 で可変壁厚を適用。

    解経路の描画モード（show_solution=True 時）:
      solution_highlight=False (デフォルト / masterpiece モード):
        白線（(255,255,255)）で太く描画。corridor_width = cell_size × 0.85。
      solution_highlight=True (デバッグ/プレビュー):
        オレンジ廊下 + 入口(緑丸)・出口(赤丸)マーカー。

    dpi: None のとき DPI メタデータなし。整数を指定すると PNG に DPI を埋め込む。
         印刷用 300、Web 用 96 など。
    """
    margin = 20
    w = width - 2 * margin
    h = height - 2 * margin
    cs_x = w / grid.cols
    cs_y = h / grid.rows
    cell_size = min(cs_x, cs_y)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    removed: set = set()
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
                    v = int(avg_lum * 220)
                    a = to_px(c + 1, r)
                    b = to_px(c + 1, r + 1)
                    draw.line([a, b], fill=(v, v, v), width=sw)
            # 下壁
            if r + 1 < grid.rows:
                cid2 = grid.cell_id(r + 1, c)
                if (min(cid, cid2), max(cid, cid2)) not in removed:
                    avg_lum = float((grid.luminance[r, c] + grid.luminance[r + 1, c]) / 2.0)
                    sw = max(1, round(_wall_stroke(stroke_width, avg_lum, thickness_range)))
                    v = int(avg_lum * 220)
                    a = to_px(c, r + 1)
                    b = to_px(c + 1, r + 1)
                    draw.line([a, b], fill=(v, v, v), width=sw)

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
    save_kwargs: dict = {"format": "PNG"}
    if dpi is not None:
        save_kwargs["dpi"] = (dpi, dpi)
    img.save(buf, **save_kwargs)
    return buf.getvalue()
