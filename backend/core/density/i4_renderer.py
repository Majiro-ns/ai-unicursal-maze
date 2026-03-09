"""
I4レンダラー: G1経路太さ連続化によるPNG/SVG高品質出力 (DM-1 Part B: cmd_376k_a7)

I4MazeResultインターフェース:
  - walls: set of ((r1,c1),(r2,c2)) pairs — 壁として存在するセル境界
  - solution_path: list of (row, col) tuples — 解経路
  - density_map: np.ndarray (rows, cols) float 0.0〜1.0 — 0=明部, 1=暗部

G1 線幅:
  - density > 0.7 (暗部): 太線（4〜8px）
  - density < 0.3 (明部): 細線（1px）
  - 中間: 線形補間  → thickness = 1.0 + 7.0 * avg_density
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Optional, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------

@dataclass
class I4MazeResult:
    """I4逆転パイプライン（path-first）の出力データ。

    Attributes:
        grid_width:    横方向セル数（cols）。
        grid_height:   縦方向セル数（rows）。
        cell_size_px:  1セル当たりのピクセルサイズ。
        walls:         壁として残るセル境界の集合。
                       各要素は ((r1,c1),(r2,c2)) タプル（正規化済みでなくてもよい）。
        solution_path: 解経路のセル座標リスト（(row, col) タプルの順序列）。
        density_map:   (grid_height, grid_width) の float 配列。
                       0.0 = 明部（白）、1.0 = 暗部（黒）。
                       luminance の逆数相当（density = 1 - luminance）。
        entrance:      入口セルの (row, col)。
        exit:          出口セルの (row, col)。
    """
    grid_width: int
    grid_height: int
    cell_size_px: int
    walls: Set[Tuple[Tuple[int, int], Tuple[int, int]]]
    solution_path: list
    density_map: np.ndarray
    entrance: Tuple[int, int]
    exit: Tuple[int, int]


# ---------------------------------------------------------------------------
# 描画定数
# ---------------------------------------------------------------------------

_WALL_COLOR_PNG = (255, 255, 255)       # 白 #FFFFFF
_PATH_COLOR_PNG = (204, 204, 204)       # 灰 #CCCCCC（非解経路）
_SOL_COLOR_PNG  = (0, 0, 0)            # 黒 #000000（解経路）
_BG_COLOR_PNG   = (255, 255, 255)       # 白背景

_PATH_COLOR_SVG = "#CCCCCC"
_SOL_COLOR_SVG  = "#000000"

_G1_THICK_DARK   = 8.0  # density=1.0 のときの線幅 (px)
_G1_THICK_BRIGHT = 1.0  # density=0.0 のときの線幅 (px)


# ---------------------------------------------------------------------------
# 内部ユーティリティ
# ---------------------------------------------------------------------------

def _norm_key(
    cell1: Tuple[int, int], cell2: Tuple[int, int]
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """壁キーを正規化（小さい方を先に）。"""
    return (min(cell1, cell2), max(cell1, cell2))


def _build_walls_normalized(
    walls: Set[Tuple[Tuple[int, int], Tuple[int, int]]]
) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """walls セットを正規化キーで再構築。"""
    return {_norm_key(a, b) for a, b in walls}


def _build_solution_set(
    solution_path: list,
) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """解経路の隣接セルペア集合（正規化済み）を構築。"""
    s: Set = set()
    for i in range(len(solution_path) - 1):
        s.add(_norm_key(solution_path[i], solution_path[i + 1]))
    return s


def _g1_thickness(avg_density: float) -> float:
    """G1 線幅計算。density=0→1px、density=1→8px、線形補間。"""
    clamped = max(0.0, min(1.0, avg_density))
    return _G1_THICK_BRIGHT + (_G1_THICK_DARK - _G1_THICK_BRIGHT) * clamped


# ---------------------------------------------------------------------------
# PNG レンダリング
# ---------------------------------------------------------------------------

def _render_png(result: I4MazeResult) -> bytes:
    """I4MazeResult → PNG バイト列。"""
    cs = result.cell_size_px
    img_w = result.grid_width * cs
    img_h = result.grid_height * cs

    img = Image.new("RGB", (img_w, img_h), _BG_COLOR_PNG)
    draw = ImageDraw.Draw(img)

    walls_norm = _build_walls_normalized(result.walls)
    sol_set = _build_solution_set(result.solution_path)
    density = result.density_map

    for r in range(result.grid_height):
        for c in range(result.grid_width):
            # セル中心ピクセル座標
            cx1 = int((c + 0.5) * cs)
            cy1 = int((r + 0.5) * cs)

            # 右隣 (r, c+1)
            if c + 1 < result.grid_width:
                key = _norm_key((r, c), (r, c + 1))
                if key not in walls_norm:
                    cx2 = int((c + 1.5) * cs)
                    cy2 = cy1
                    if key in sol_set:
                        avg_d = (float(density[r, c]) + float(density[r, c + 1])) / 2.0
                        w = max(1, round(_g1_thickness(avg_d)))
                        draw.line([(cx1, cy1), (cx2, cy2)],
                                  fill=_SOL_COLOR_PNG, width=w)
                    else:
                        draw.line([(cx1, cy1), (cx2, cy2)],
                                  fill=_PATH_COLOR_PNG, width=1)

            # 下隣 (r+1, c)
            if r + 1 < result.grid_height:
                key = _norm_key((r, c), (r + 1, c))
                if key not in walls_norm:
                    cx2 = cx1
                    cy2 = int((r + 1.5) * cs)
                    if key in sol_set:
                        avg_d = (float(density[r, c]) + float(density[r + 1, c])) / 2.0
                        w = max(1, round(_g1_thickness(avg_d)))
                        draw.line([(cx1, cy1), (cx2, cy2)],
                                  fill=_SOL_COLOR_PNG, width=w)
                    else:
                        draw.line([(cx1, cy1), (cx2, cy2)],
                                  fill=_PATH_COLOR_PNG, width=1)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# SVG レンダリング
# ---------------------------------------------------------------------------

def _render_svg(result: I4MazeResult) -> str:
    """I4MazeResult → SVG 文字列。"""
    cs = result.cell_size_px
    svg_w = result.grid_width * cs
    svg_h = result.grid_height * cs

    walls_norm = _build_walls_normalized(result.walls)
    sol_set = _build_solution_set(result.solution_path)
    density = result.density_map

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{svg_w}" height="{svg_h}" '
        f'viewBox="0 0 {svg_w} {svg_h}">',
        f'<rect width="{svg_w}" height="{svg_h}" fill="white"/>',
    ]

    # 非解経路をまとめて描画 (固定 stroke-width=1)
    path_cmds = []
    # 解経路は per-segment で描画（可変幅）
    sol_lines = []

    for r in range(result.grid_height):
        for c in range(result.grid_width):
            x1 = (c + 0.5) * cs
            y1 = (r + 0.5) * cs

            # 右隣
            if c + 1 < result.grid_width:
                key = _norm_key((r, c), (r, c + 1))
                if key not in walls_norm:
                    x2 = (c + 1.5) * cs
                    y2 = y1
                    if key in sol_set:
                        avg_d = (float(density[r, c]) + float(density[r, c + 1])) / 2.0
                        sw = round(_g1_thickness(avg_d), 2)
                        sol_lines.append(
                            f'<line x1="{x1:.1f}" y1="{y1:.1f}" '
                            f'x2="{x2:.1f}" y2="{y2:.1f}" '
                            f'stroke="{_SOL_COLOR_SVG}" stroke-width="{sw}" '
                            f'stroke-linecap="round"/>'
                        )
                    else:
                        path_cmds.append(f"M{x1:.1f} {y1:.1f}H{x2:.1f}")

            # 下隣
            if r + 1 < result.grid_height:
                key = _norm_key((r, c), (r + 1, c))
                if key not in walls_norm:
                    x2 = x1
                    y2 = (r + 1.5) * cs
                    if key in sol_set:
                        avg_d = (float(density[r, c]) + float(density[r + 1, c])) / 2.0
                        sw = round(_g1_thickness(avg_d), 2)
                        sol_lines.append(
                            f'<line x1="{x1:.1f}" y1="{y1:.1f}" '
                            f'x2="{x2:.1f}" y2="{y2:.1f}" '
                            f'stroke="{_SOL_COLOR_SVG}" stroke-width="{sw}" '
                            f'stroke-linecap="round"/>'
                        )
                    else:
                        path_cmds.append(f"M{x1:.1f} {y1:.1f}V{y2:.1f}")

    # 非解経路: <g> グループ化で ファイルサイズ圧縮
    if path_cmds:
        d = " ".join(path_cmds)
        parts.append(
            f'<g stroke="{_PATH_COLOR_SVG}" stroke-width="1">'
            f'<path d="{d}" fill="none"/></g>'
        )

    # 解経路: 個別 <line> (G1 可変幅)
    parts.extend(sol_lines)
    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def render_i4_maze(
    result: I4MazeResult,
    output_path: str,
    format: str = "png",
) -> str:
    """I4MazeResult を PNG または SVG ファイルに出力する。

    Args:
        result:      I4MazeResult — path_designer.build_walls_around_path() の出力。
        output_path: 出力ファイルパス（拡張子を含む）。
        format:      "png" または "svg"（大文字小文字不問）。

    Returns:
        書き込んだファイルの絶対パス。

    Raises:
        ValueError: format が "png" でも "svg" でもない場合。
    """
    fmt = format.lower()
    if fmt not in ("png", "svg"):
        raise ValueError(f"サポート外のフォーマット: {format!r}。'png' または 'svg' を指定してください。")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if fmt == "png":
        data = _render_png(result)
        with open(output_path, "wb") as f:
            f.write(data)
    else:
        svg_text = _render_svg(result)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(svg_text)

    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# ヘルパー: 既存パイプライン出力 → I4MazeResult 変換
# ---------------------------------------------------------------------------

def density_maze_result_to_i4(
    grid,                    # backend.core.density.grid_builder.CellGrid
    adj: dict,               # 隣接リスト（open passages, cell id ベース）
    solution_path_ids: list, # セル id のリスト
    entrance_id: int,
    exit_id: int,
    cell_size_px: int = 3,
) -> "I4MazeResult":
    """既存 generate_density_maze() の内部データを I4MazeResult に変換する。

    density_map = 1 - grid.luminance として構築（luminance 0=黒→density 1.0=暗部）。

    Args:
        grid:             CellGrid（grid.luminance, grid.rows, grid.cols）。
        adj:              build_spanning_tree 等が返す隣接リスト（open passage）。
        solution_path_ids: 解経路のセル id リスト。
        entrance_id:      入口セル id。
        exit_id:          出口セル id。
        cell_size_px:     レンダリング時の 1 セルピクセルサイズ（デフォルト 3）。

    Returns:
        I4MazeResult
    """
    rows, cols = grid.rows, grid.cols

    # open passages を cell id ペア集合に変換
    open_passages: set = set()
    for u, neighbors in adj.items():
        for v in neighbors:
            open_passages.add((min(u, v), max(u, v)))

    # walls: 隣接セルのうち open passages に含まれないペア → (row, col) 形式
    walls: set = set()
    for r in range(rows):
        for c in range(cols):
            cid = r * cols + c
            if c + 1 < cols:
                cid2 = r * cols + (c + 1)
                if (min(cid, cid2), max(cid, cid2)) not in open_passages:
                    walls.add(((r, c), (r, c + 1)))
            if r + 1 < rows:
                cid2 = (r + 1) * cols + c
                if (min(cid, cid2), max(cid, cid2)) not in open_passages:
                    walls.add(((r, c), (r + 1, c)))

    # solution_path: セル id → (row, col)
    solution_path_rc = [grid.cell_rc(cid) for cid in solution_path_ids]

    # density_map: 1 - luminance（明部=0, 暗部=1）
    density_map = 1.0 - grid.luminance.astype(np.float64)

    return I4MazeResult(
        grid_width=cols,
        grid_height=rows,
        cell_size_px=cell_size_px,
        walls=walls,
        solution_path=solution_path_rc,
        density_map=density_map,
        entrance=grid.cell_rc(entrance_id),
        exit=grid.cell_rc(exit_id),
    )
