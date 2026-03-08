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

import numpy as np
from PIL import Image, ImageDraw

from .grid_builder import CellGrid


def _cell_center(
    grid: CellGrid,
    cell_id: int,
    cell_size: float,
    margin: float,
    x_offsets: Optional[np.ndarray] = None,
    y_offsets: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """セル中心座標を返す。x_offsets/y_offsets が与えられた場合は累積和ベースで計算。"""
    r, c = grid.cell_rc(cell_id)
    if x_offsets is not None and y_offsets is not None:
        x = margin + (x_offsets[c] + x_offsets[c + 1]) / 2.0
        y = margin + (y_offsets[r] + y_offsets[r + 1]) / 2.0
    else:
        x = margin + (c + 0.5) * cell_size
        y = margin + (r + 0.5) * cell_size
    return x, y


def _compute_offsets(
    grid: CellGrid,
    w: float,
    h: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    描画用の cell_size・x_offsets・y_offsets を計算して返す。

    可変セルサイズモード（grid.col_widths/row_heights が設定済み）:
        x_offsets = [0, col_widths[0]*w, ..., w]  len=cols+1
        y_offsets = 同様                           len=rows+1
        cell_size = min(w/cols, h/rows)  ← corridor_width 計算の参考値
    均一モード（後方互換・None の場合）:
        cell_size = min(w/cols, h/rows)
        x_offsets = [0, cs, 2cs, ..., cols*cs]
        y_offsets = [0, cs, 2cs, ..., rows*cs]
    """
    cs_x = w / grid.cols
    cs_y = h / grid.rows
    cell_size = min(cs_x, cs_y)

    if grid.col_widths is not None and grid.row_heights is not None:
        x_offsets = np.concatenate([[0.0], np.cumsum(grid.col_widths * w)])
        y_offsets = np.concatenate([[0.0], np.cumsum(grid.row_heights * h)])
    else:
        x_offsets = np.arange(grid.cols + 1, dtype=np.float64) * cell_size
        y_offsets = np.arange(grid.rows + 1, dtype=np.float64) * cell_size

    return cell_size, x_offsets, y_offsets


def _wall_stroke(stroke_width_base: float, avg_lum: float, thickness_range: float) -> float:
    """
    Xu & Kaplan (SIGGRAPH 2007) 濃淡公式:
      G = (S - W) / S  →  暗い部分は W（壁厚）を太く、明るい部分は細く。
    wall_thickness = stroke_width_base * (1.0 + thickness_range * (1.0 - avg_lum))
    avg_lum=0（黒）: stroke_width_base * (1.0 + thickness_range)  → 最大
    avg_lum=1（白）: stroke_width_base * 1.0                       → 最小
    """
    return stroke_width_base * (1.0 + thickness_range * (1.0 - avg_lum))


def _wall_v(avg_lum: float, v_min: int = 40, v_max: int = 175) -> int:
    """
    壁色グレー値（整数 0-255）を返す。
    公式: v = v_min + int((v_max - v_min) * avg_lum)
    デフォルト (v_min=40, v_max=175):
      背景白(255)との差 = 255 - v_max = 80 → 全輝度範囲で Δv≥80 を保証。
      avg_lum=0（黒画素）→ 40  (濃灰: Δv=215)
      avg_lum=1（白画素）→ 175 (淡灰: Δv=80 — 最低コントラスト保証)
    """
    return v_min + int((v_max - v_min) * avg_lum)


def _wall_color(avg_lum: float, v_min: int = 40, v_max: int = 175) -> str:
    """
    壁の描画色をグレースケールで返す（SVG stroke 用）。
    背景白(255)との最低コントラスト差 Δv≥(255-v_max) を保証。
    デフォルト: v_max=175 → Δv≥80 を全輝度範囲で保証（cmd_361k_a2修正）。
      avg_lum=0（黒画素）→ rgb(40,40,40)   (v_min=40)
      avg_lum=1（白画素）→ rgb(175,175,175) (v_max=175, Δv=80)
    """
    v = _wall_v(avg_lum, v_min, v_max)
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
    wall_color_min: int = 40,
    wall_color_max: int = 175,
    use_gradient_walls: bool = False,
    # G1: per-segment variable path width (path-first V2)
    cell_luminance: Optional[np.ndarray] = None,
    path_thickness_dark: float = 6.0,
    path_thickness_bright: float = 1.0,
) -> str:
    """
    迷路を SVG で描画。セルは矩形、隣接かつ壁除去済みでない境界に線を引く。
    thickness_range > 0 のとき、隣接2セルの平均輝度に応じて壁厚を変える（可変壁厚）。

    SVG最適化（Phase 3）:
      壁を stroke-width でグループ化し <g stroke-width="..."><path d="..."/></g> 形式で出力。
      M/V/H コマンドで座標を簡略化。座標精度 .1f。
      stroke_quantize_levels > 0 のとき avg_lum を N 段階に量子化してグループ数を削減。

    壁色グラデーション（use_gradient_walls=True）:
      各壁セグメントを SVG linearGradient で描画。<defs> に linearGradient を挿入し、
      各壁を <rect fill="url(#gid)"/> として出力。
      垂直壁: 上=左セル色 → 下=右セル色 (objectBoundingBox, x1=0 y1=0 x2=0 y2=1)
      水平壁: 左=上セル色 → 右=下セル色 (objectBoundingBox, x1=0 y1=0 x2=1 y2=0)
      stroke_quantize_levels で量子化しグラデーション数を制限。

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
    cell_size, x_offsets, y_offsets = _compute_offsets(grid, w, h)

    parts: List[str] = []

    # 外枠（固定 stroke_width）
    parts.append(
        f'<rect x="{margin:.1f}" y="{margin:.1f}" '
        f'width="{x_offsets[-1]:.1f}" height="{y_offsets[-1]:.1f}" '
        f'fill="none" stroke="black" stroke-width="{stroke_width}"/>'
    )

    removed: set = set()
    for u, neighbors in adj.items():
        for v in neighbors:
            removed.add((min(u, v), max(u, v)))

    if use_gradient_walls:
        # --- グラデーション壁モード ---
        # stroke_quantize_levels で量子化（グラデーション数の爆発を防ぐ）
        nlevels = stroke_quantize_levels if stroke_quantize_levels > 0 else 20
        grad_defs: set = set()   # ('v'/'h', qi1, qi2)
        grad_rects: List[str] = []

        for r in range(grid.rows):
            for c in range(grid.cols):
                cid = grid.cell_id(r, c)
                x0 = margin + x_offsets[c]
                y0 = margin + y_offsets[r]

                # 右壁（垂直 rect: 上=左セル色, 下=右セル色）
                if c + 1 < grid.cols:
                    cid2 = grid.cell_id(r, c + 1)
                    if (min(cid, cid2), max(cid, cid2)) not in removed:
                        lum1 = float(grid.luminance[r, c])
                        lum2 = float(grid.luminance[r, c + 1])
                        qi1 = round(lum1 * nlevels)
                        qi2 = round(lum2 * nlevels)
                        ql1 = qi1 / nlevels
                        ql2 = qi2 / nlevels
                        avg_q = (ql1 + ql2) / 2.0
                        sw = _wall_stroke(stroke_width, avg_q, thickness_range)
                        gid = f"gw_v_{qi1}_{qi2}"
                        grad_defs.add(("v", qi1, qi2))
                        x1 = margin + x_offsets[c + 1]
                        h_seg = float(y_offsets[r + 1] - y_offsets[r])
                        grad_rects.append(
                            f'<rect x="{x1 - sw / 2:.1f}" y="{y0:.1f}" '
                            f'width="{sw:.2f}" height="{h_seg:.1f}" fill="url(#{gid})"/>'
                        )

                # 下壁（水平 rect: 左=上セル色, 右=下セル色）
                if r + 1 < grid.rows:
                    cid2 = grid.cell_id(r + 1, c)
                    if (min(cid, cid2), max(cid, cid2)) not in removed:
                        lum1 = float(grid.luminance[r, c])
                        lum2 = float(grid.luminance[r + 1, c])
                        qi1 = round(lum1 * nlevels)
                        qi2 = round(lum2 * nlevels)
                        ql1 = qi1 / nlevels
                        ql2 = qi2 / nlevels
                        avg_q = (ql1 + ql2) / 2.0
                        sw = _wall_stroke(stroke_width, avg_q, thickness_range)
                        gid = f"gw_h_{qi1}_{qi2}"
                        grad_defs.add(("h", qi1, qi2))
                        y1 = margin + y_offsets[r + 1]
                        w_seg = float(x_offsets[c + 1] - x_offsets[c])
                        grad_rects.append(
                            f'<rect x="{x0:.1f}" y="{y1 - sw / 2:.1f}" '
                            f'width="{w_seg:.1f}" height="{sw:.2f}" fill="url(#{gid})"/>'
                        )

        # <defs> に linearGradient 要素を構築
        defs_elements: List[str] = []
        for (direction, qi1, qi2) in sorted(grad_defs):
            ql1 = qi1 / nlevels
            ql2 = qi2 / nlevels
            c1 = _wall_color(ql1, wall_color_min, wall_color_max)
            c2 = _wall_color(ql2, wall_color_min, wall_color_max)
            gid = f"gw_{direction}_{qi1}_{qi2}"
            if direction == "v":
                # 垂直壁: 上→下グラデーション
                defs_elements.append(
                    f'<linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1"'
                    f' gradientUnits="objectBoundingBox">'
                    f'<stop offset="0%" stop-color="{c1}"/>'
                    f'<stop offset="100%" stop-color="{c2}"/>'
                    f'</linearGradient>'
                )
            else:
                # 水平壁: 左→右グラデーション
                defs_elements.append(
                    f'<linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="0"'
                    f' gradientUnits="objectBoundingBox">'
                    f'<stop offset="0%" stop-color="{c1}"/>'
                    f'<stop offset="100%" stop-color="{c2}"/>'
                    f'</linearGradient>'
                )

        if defs_elements:
            # <defs> を SVG 先頭（外枠より前）に挿入
            parts.insert(0, "<defs>\n" + "\n".join(defs_elements) + "\n</defs>")
        parts.extend(grad_rects)

    else:
        # --- 既存の path/group モード（後方互換）---
        # 壁を (color, stroke-width) キーでグループ化（M/V/H コマンドで座標簡略化）
        wall_cmds: Dict[tuple, List[str]] = defaultdict(list)

        for r in range(grid.rows):
            for c in range(grid.cols):
                cid = grid.cell_id(r, c)
                x0 = margin + x_offsets[c]
                y0 = margin + y_offsets[r]

                # 右壁（垂直線: x=x_offsets[c+1], y: y_offsets[r]→y_offsets[r+1]）
                if c + 1 < grid.cols:
                    cid2 = grid.cell_id(r, c + 1)
                    if (min(cid, cid2), max(cid, cid2)) not in removed:
                        avg_lum = float((grid.luminance[r, c] + grid.luminance[r, c + 1]) / 2.0)
                        if thickness_range > 0 and stroke_quantize_levels > 0:
                            q_lum = round(avg_lum * stroke_quantize_levels) / stroke_quantize_levels
                            sw = _wall_stroke(stroke_width, q_lum, thickness_range)
                            color = _wall_color(q_lum, wall_color_min, wall_color_max)
                        else:
                            sw = _wall_stroke(stroke_width, avg_lum, thickness_range)
                            color = _wall_color(avg_lum, wall_color_min, wall_color_max)
                        x1 = margin + x_offsets[c + 1]
                        y1 = margin + y_offsets[r + 1]
                        wall_cmds[(color, f"{sw:.3f}")].append(
                            f"M{x1:.1f} {y0:.1f}V{y1:.1f}"
                        )

                # 下壁（水平線: y=y_offsets[r+1], x: x_offsets[c]→x_offsets[c+1]）
                if r + 1 < grid.rows:
                    cid2 = grid.cell_id(r + 1, c)
                    if (min(cid, cid2), max(cid, cid2)) not in removed:
                        avg_lum = float((grid.luminance[r, c] + grid.luminance[r + 1, c]) / 2.0)
                        if thickness_range > 0 and stroke_quantize_levels > 0:
                            q_lum = round(avg_lum * stroke_quantize_levels) / stroke_quantize_levels
                            sw = _wall_stroke(stroke_width, q_lum, thickness_range)
                            color = _wall_color(q_lum, wall_color_min, wall_color_max)
                        else:
                            sw = _wall_stroke(stroke_width, avg_lum, thickness_range)
                            color = _wall_color(avg_lum, wall_color_min, wall_color_max)
                        x1 = margin + x_offsets[c + 1]
                        y1 = margin + y_offsets[r + 1]
                        wall_cmds[(color, f"{sw:.3f}")].append(
                            f"M{x0:.1f} {y1:.1f}H{x1:.1f}"
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
        if solution_highlight:
            # デバッグ/プレビューモード: オレンジ廊下 + 入口・出口マーカー
            path_d = []
            for i, cid in enumerate(solution_path):
                x, y = _cell_center(grid, cid, cell_size, margin, x_offsets, y_offsets)
                if i == 0:
                    path_d.append(f"M{x:.1f} {y:.1f}")
                else:
                    path_d.append(f"L{x:.1f} {y:.1f}")
            corridor_w = max(stroke_width * 1.5, cell_size * 0.40)
            parts.append(
                f'<path d="{" ".join(path_d)}" fill="none" stroke="#E05000" '
                f'stroke-width="{corridor_w:.2f}" stroke-linecap="round" '
                f'stroke-linejoin="round" opacity="0.65"/>'
            )
            ex, ey = _cell_center(grid, solution_path[0], cell_size, margin, x_offsets, y_offsets)
            r_marker = max(cell_size * 0.3, stroke_width)
            parts.append(
                f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="{r_marker:.2f}" '
                f'fill="#00AA44" opacity="0.85"/>'
            )
            if len(solution_path) > 1:
                gx_pt, gy_pt = _cell_center(grid, solution_path[-1], cell_size, margin, x_offsets, y_offsets)
                parts.append(
                    f'<circle cx="{gx_pt:.1f}" cy="{gy_pt:.1f}" r="{r_marker:.2f}" '
                    f'fill="#CC2222" opacity="0.85"/>'
                )
        elif cell_luminance is not None and len(solution_path) > 1:
            # G1: per-segment variable width (path-first V2 masterpiece mode)
            flat_lum = cell_luminance.flatten()
            for i in range(len(solution_path) - 1):
                cid_a = solution_path[i]
                cid_b = solution_path[i + 1]
                x1, y1 = _cell_center(grid, cid_a, cell_size, margin, x_offsets, y_offsets)
                x2, y2 = _cell_center(grid, cid_b, cell_size, margin, x_offsets, y_offsets)
                avg_lum = (float(flat_lum[cid_a]) + float(flat_lum[cid_b])) / 2.0
                # Dark cells -> thick path, Bright cells -> thin path
                seg_width = path_thickness_bright + (path_thickness_dark - path_thickness_bright) * (1.0 - avg_lum)
                parts.append(
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                    f'stroke="white" stroke-width="{seg_width:.2f}" stroke-linecap="round"/>'
                )
        else:
            # masterpiece モード: 白線で経路を塗りつぶし（経路部分が白く浮かぶ）
            path_d = []
            for i, cid in enumerate(solution_path):
                x, y = _cell_center(grid, cid, cell_size, margin, x_offsets, y_offsets)
                if i == 0:
                    path_d.append(f"M{x:.1f} {y:.1f}")
                else:
                    path_d.append(f"L{x:.1f} {y:.1f}")
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
    wall_color_min: int = 40,
    wall_color_max: int = 175,
    # G1: per-segment variable path width (path-first V2)
    cell_luminance: Optional[np.ndarray] = None,
    path_thickness_dark: float = 6.0,
    path_thickness_bright: float = 1.0,
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
    cell_size, x_offsets, y_offsets = _compute_offsets(grid, w, h)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    removed: set = set()
    for u, neighbors in adj.items():
        for v in neighbors:
            removed.add((min(u, v), max(u, v)))

    def to_px(col: float, row: float) -> tuple[int, int]:
        return (
            int(margin + x_offsets[int(col)]),
            int(margin + y_offsets[int(row)]),
        )

    for r in range(grid.rows):
        for c in range(grid.cols):
            cid = grid.cell_id(r, c)
            # 右壁
            if c + 1 < grid.cols:
                cid2 = grid.cell_id(r, c + 1)
                if (min(cid, cid2), max(cid, cid2)) not in removed:
                    avg_lum = float((grid.luminance[r, c] + grid.luminance[r, c + 1]) / 2.0)
                    sw = max(1, round(_wall_stroke(stroke_width, avg_lum, thickness_range)))
                    v = _wall_v(avg_lum, wall_color_min, wall_color_max)
                    a = to_px(c + 1, r)
                    b = to_px(c + 1, r + 1)
                    draw.line([a, b], fill=(v, v, v), width=sw)
            # 下壁
            if r + 1 < grid.rows:
                cid2 = grid.cell_id(r + 1, c)
                if (min(cid, cid2), max(cid, cid2)) not in removed:
                    avg_lum = float((grid.luminance[r, c] + grid.luminance[r + 1, c]) / 2.0)
                    sw = max(1, round(_wall_stroke(stroke_width, avg_lum, thickness_range)))
                    v = _wall_v(avg_lum, wall_color_min, wall_color_max)
                    a = to_px(c, r + 1)
                    b = to_px(c + 1, r + 1)
                    draw.line([a, b], fill=(v, v, v), width=sw)

    if show_solution and solution_path:
        pts = [_cell_center(grid, cid, cell_size, margin, x_offsets, y_offsets) for cid in solution_path]
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
        elif cell_luminance is not None and len(solution_path) > 1:
            # G1: per-segment variable width (path-first V2 masterpiece mode)
            flat_lum = cell_luminance.flatten()
            for i in range(len(px_pts) - 1):
                cid_a = solution_path[i]
                cid_b = solution_path[i + 1]
                avg_lum = (float(flat_lum[cid_a]) + float(flat_lum[cid_b])) / 2.0
                seg_width = path_thickness_bright + (path_thickness_dark - path_thickness_bright) * (1.0 - avg_lum)
                draw.line([px_pts[i], px_pts[i + 1]], fill=(255, 255, 255), width=max(1, int(seg_width)))
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
