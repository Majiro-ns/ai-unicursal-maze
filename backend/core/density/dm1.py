"""
DM-1: 密度制御迷路 — 明示的インターフェース（Phase DM-1 基盤実装）

設計書: docs/maze-artisan-masterpiece-requirements.md Phase DM-1

5段階パイプライン:
  Stage 1: 前処理（グレースケール + CLAHE リサイズ）
  Stage 2: density_map 生成（グレースケール→0.0〜1.0 正規化）
  Stage 3: density_mapに基づくエッジ重み設定 + Kruskal迷路生成
           暗部(低輝度) = 低壁重み = 先に除去 = 通路が多い(密)
           明部(高輝度) = 高壁重み = 後から除去 = 通路が少ない(疎)
  Stage 4: 入口・出口決定 + BFSソルバーによる解経路検証
  Stage 5: PNG/SVGレンダリング（cell_size_px指定）

推奨パラメータ:
  grid_rows/grid_cols : 200〜400
  cell_size_px        : 2〜5
  density_min/max     : 0.1〜0.9
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .entrance_exit import find_entrance_exit_and_path
from .exporter import maze_to_png, maze_to_svg
from .grid_builder import CellGrid, build_density_map
from .maze_builder import build_spanning_tree
from .preprocess import preprocess_image
from .solver import bfs_has_path


# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

@dataclass
class DM1Config:
    """
    DM-1密度制御迷路の生成パラメータ。

    設計書 § 2.3 重要パラメータに対応する明示的インターフェース。
    """
    # グリッドサイズ（推奨 200〜400）
    grid_rows: int = 50
    grid_cols: int = 50
    # 1セルのピクセルサイズ（推奨 2〜5）
    cell_size_px: int = 3
    # 通路密度範囲
    #   density_min: 明部（輝度=1.0）の壁重み下限 → 通路が少なくなりやすい
    #   density_max: 暗部（輝度=0.0）の壁重み上限 → 通路が多くなりやすい
    #   ※ weight = density_min + lum_avg * (density_max - density_min)
    #      weight 小 → Kruskal で先に除去 → 通路が多い
    density_min: float = 0.1
    density_max: float = 0.9
    # 出力オプション
    show_solution: bool = False
    # 前処理リサイズ最大辺長（0=リサイズなし）
    max_side: int = 512


# ---------------------------------------------------------------------------
# 結果
# ---------------------------------------------------------------------------

@dataclass
class DM1Result:
    """DM-1生成結果。"""
    svg: str
    png_bytes: bytes
    entrance: int
    exit_cell: int
    solution_path: List[int]
    grid_rows: int
    grid_cols: int
    density_map: np.ndarray          # (grid_rows, grid_cols) float 輝度マップ
    adj: Dict[int, List[int]]        # 隣接リスト（検証・デバッグ用）


# ---------------------------------------------------------------------------
# 内部ユーティリティ
# ---------------------------------------------------------------------------

def _build_dm1_walls(
    lum: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    density_min: float,
    density_max: float,
) -> List[Tuple[int, int, float]]:
    """
    density_min〜density_max にスケールした壁重みリストを生成。

    weight = density_min + avg_lum * (density_max - density_min)

    暗部 (avg_lum → 0) → weight ≈ density_min → Kruskal で先に除去 → 通路多(密)
    明部 (avg_lum → 1) → weight ≈ density_max → Kruskal で後から除去 → 通路少(疎)

    density_min < density_max の場合、壁重みの相対順序は輝度順と同一。
    density_min = density_max の場合、全壁の重みが等しくなり均一迷路になる。
    """
    if density_min > density_max:
        raise ValueError(
            f"density_min ({density_min}) は density_max ({density_max}) 以下でなければなりません"
        )
    ids = np.arange(grid_rows * grid_cols, dtype=np.int32).reshape(grid_rows, grid_cols)

    # 右壁（水平隣接）
    h_cid1 = ids[:, :-1].ravel()
    h_cid2 = ids[:, 1:].ravel()
    h_avg = (lum[:, :-1] + lum[:, 1:]) / 2.0
    h_w = density_min + h_avg.ravel() * (density_max - density_min)

    # 下壁（垂直隣接）
    v_cid1 = ids[:-1, :].ravel()
    v_cid2 = ids[1:, :].ravel()
    v_avg = (lum[:-1, :] + lum[1:, :]) / 2.0
    v_w = density_min + v_avg.ravel() * (density_max - density_min)

    c1 = np.concatenate([h_cid1, v_cid1]).tolist()
    c2 = np.concatenate([h_cid2, v_cid2]).tolist()
    w  = np.concatenate([h_w, v_w]).tolist()

    return list(zip(c1, c2, w))


# ---------------------------------------------------------------------------
# メイン API
# ---------------------------------------------------------------------------

def generate_dm1_maze(
    image: Image.Image,
    config: Optional[DM1Config] = None,
) -> DM1Result:
    """
    DM-1密度制御迷路を生成する。

    設計書 § 2.2 パイプライン設計の5段階を明示的に実装。

    Args:
        image : 入力画像（任意サイズ・任意形式）。
        config: DM1Config。None の場合はデフォルト値を使用。

    Returns:
        DM1Result（svg / png_bytes / entrance / exit_cell /
                  solution_path / grid_rows / grid_cols / density_map / adj）

    Raises:
        ValueError: density_min > density_max の場合。
        RuntimeError: BFS でも解経路が見つからない場合（実装バグ）。
    """
    if config is None:
        config = DM1Config()

    # ------------------------------------------------------------------
    # Stage 1: 前処理（グレースケール + CLAHE + リサイズ）
    # ------------------------------------------------------------------
    gray = preprocess_image(image, max_side=config.max_side)

    # ------------------------------------------------------------------
    # Stage 2: density_map 生成
    # ------------------------------------------------------------------
    # グリッドサイズは画像サイズに応じて上限クリップ（最小1）
    grid_rows = min(config.grid_rows, max(gray.shape[0] // 4, 1))
    grid_cols = min(config.grid_cols, max(gray.shape[1] // 4, 1))
    density_map = build_density_map(gray, grid_rows, grid_cols)

    # ------------------------------------------------------------------
    # Stage 3: エッジ重み設定（density_min/max スケール）+ Kruskal 迷路生成
    # ------------------------------------------------------------------
    walls = _build_dm1_walls(
        density_map, grid_rows, grid_cols,
        config.density_min, config.density_max,
    )
    grid = CellGrid(rows=grid_rows, cols=grid_cols, luminance=density_map, walls=walls)
    adj = build_spanning_tree(grid)

    # ------------------------------------------------------------------
    # Stage 4: 入口・出口・解経路 + BFS 検証
    # ------------------------------------------------------------------
    entrance, exit_cell, solution_path = find_entrance_exit_and_path(adj, grid.num_cells)
    if not bfs_has_path(adj, entrance, exit_cell):
        raise RuntimeError(
            f"BFS: 入口({entrance})→出口({exit_cell}) の解経路が存在しません。"
            "これは実装バグです。"
        )

    # ------------------------------------------------------------------
    # Stage 5: PNG/SVG レンダリング
    # ------------------------------------------------------------------
    out_w = grid_cols * config.cell_size_px
    out_h = grid_rows * config.cell_size_px

    svg = maze_to_svg(
        grid, adj, entrance, exit_cell, solution_path,
        width=out_w,
        height=out_h,
        show_solution=config.show_solution,
    )
    png_bytes = maze_to_png(
        grid, adj, entrance, exit_cell, solution_path,
        width=out_w,
        height=out_h,
        show_solution=config.show_solution,
    )

    return DM1Result(
        svg=svg,
        png_bytes=png_bytes,
        entrance=entrance,
        exit_cell=exit_cell,
        solution_path=solution_path,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        density_map=density_map,
        adj=adj,
    )
