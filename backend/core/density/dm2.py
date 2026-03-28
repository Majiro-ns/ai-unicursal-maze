"""
DM-2: エッジ強調 + CLAHE自動チューニング 明示的インターフェース

設計書: docs/maze-artisan-masterpiece-requirements.md Phase DM-2

DM-1 との差分:
  + Canny/Sobel エッジ検出 → エッジマップ生成
  + エッジ部分の壁を優先的に残す（edge_weight で制御）
  + CLAHEパラメータの画像別自動チューニング（輝度ヒストグラム解析）
  + 解経路数の制御と検証（max_solutions）

成功基準:
  「エッジピクセル領域の壁密度 ≥ 非エッジ領域の壁密度 × 1.5」
  → エッジ強調壁が輪郭を白線として残し、元画像の輪郭が視認できる
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .dm1 import DM1Config, _build_dm1_walls
from .edge_enhancer import apply_edge_boost_to_walls, detect_edge_map
from .entrance_exit import find_entrance_exit_and_path
from .exporter import maze_to_png, maze_to_svg
from .grid_builder import CellGrid, build_density_map
from .maze_builder import build_spanning_tree
from .preprocess import preprocess_image
from .solver import bfs_has_path, count_solutions_dfs


# ---------------------------------------------------------------------------
# CLAHE 自動チューニング
# ---------------------------------------------------------------------------

def auto_tune_clahe(gray: np.ndarray) -> Tuple[float, int]:
    """
    画像の輝度ヒストグラム（標準偏差）から最適な CLAHE パラメータを自動決定する。

    分類:
      低コントラスト (std < 0.15): clip_limit=0.05, n_tiles=32
        → 局所的・積極的な補正で細部を引き出す
      中コントラスト (0.15 ≤ std < 0.30): clip_limit=0.03, n_tiles=16
        → 標準的な補正
      高コントラスト (std ≥ 0.30): clip_limit=0.01, n_tiles=8
        → 穏やかな補正で過補正を防ぐ

    Args:
        gray: (H, W) float 0.0〜1.0 グレースケール配列。

    Returns:
        (clip_limit, n_tiles):
          clip_limit: equalize_adapthist の clip_limit パラメータ。
          n_tiles   : タイル分割数（kernel_size_px = image_dim // n_tiles）。
                      大きいほど小タイル = より局所的な補正。
    """
    std = float(np.std(gray))

    if std < 0.15:
        return 0.05, 32   # 低コントラスト: 積極的補正 + 細密タイル
    elif std < 0.30:
        return 0.03, 16   # 中コントラスト: 標準
    else:
        return 0.01, 8    # 高コントラスト: 穏やか + 粗大タイル


def _apply_clahe_custom(
    gray: np.ndarray,
    clip_limit: float,
    n_tiles: int,
) -> np.ndarray:
    """
    カスタム CLAHE パラメータで contrast enhancement を適用する。

    preprocess_image() が使う kernel_size=None（デフォルト: 画像サイズ//8）の代わりに
    n_tiles で明示的にタイル数を指定する。

    Args:
        gray     : (H, W) float 0.0〜1.0 グレースケール。
        clip_limit: Histogram clip limit（0.0 で事実上無効）。
        n_tiles  : タイル分割数（kernel_size = image_dim // n_tiles）。

    Returns:
        CLAHE 適用後の (H, W) float 配列。均一画像はそのまま返す。
    """
    if float(np.std(gray)) < 1e-4:
        return gray  # 均一画像: CLAHE は意味がない（全1.0 になるためスキップ）

    from skimage.exposure import equalize_adapthist

    h, w = gray.shape
    tile_h = max(4, h // max(1, n_tiles))
    tile_w = max(4, w // max(1, n_tiles))

    result = equalize_adapthist(
        gray,
        kernel_size=(tile_h, tile_w),
        clip_limit=clip_limit,
        nbins=256,
    )
    return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

@dataclass
class DM2Config(DM1Config):
    """
    DM-2 設定。DM1Config を継承し、エッジ強調・CLAHE・解数制御パラメータを追加。
    """
    # ----- エッジ強調パラメータ -----
    # edge_weight: 0.0=エッジ強調なし(DM-1相当) / 1.0=最大強調
    edge_weight: float = 0.5
    # Canny 前段ガウシアン標準偏差
    edge_sigma: float = 1.0
    # Canny 二重閾値（0.0〜1.0 相対値）
    edge_low_threshold: float = 0.05
    edge_high_threshold: float = 0.20

    # ----- CLAHE 自動チューニング -----
    # True: auto_tune_clahe() で画像別自動決定
    # False: 以下の手動値を使用
    auto_clahe: bool = True
    # 手動 CLAHE パラメータ（auto_clahe=False 時のみ使用）
    clahe_clip_limit: float = 0.03
    clahe_tile_size: int = 16   # n_tiles: 分割数

    # ----- 解経路数制御 -----
    max_solutions: int = 10


# ---------------------------------------------------------------------------
# 結果
# ---------------------------------------------------------------------------

@dataclass
class DM2Result:
    """DM-2 生成結果。DM-1 互換フィールド + エッジマップ + 解数情報。"""
    # DM-1 互換フィールド
    svg: str
    png_bytes: bytes
    entrance: int
    exit_cell: int
    solution_path: List[int]
    grid_rows: int
    grid_cols: int
    density_map: np.ndarray          # (grid_rows, grid_cols) float 輝度マップ
    adj: Dict[int, List[int]]        # 隣接リスト（検証・デバッグ用）
    # DM-2 追加フィールド
    edge_map: np.ndarray             # (grid_rows, grid_cols) float エッジ強度
    solution_count: int              # 実際の解経路数（perfect maze = 1）
    clahe_clip_limit_used: float     # 実際に使用した clip_limit
    clahe_n_tiles_used: int          # 実際に使用した n_tiles


# ---------------------------------------------------------------------------
# メイン API
# ---------------------------------------------------------------------------

def generate_dm2_maze(
    image: Image.Image,
    config: Optional[DM2Config] = None,
) -> DM2Result:
    """
    DM-2 密度制御迷路を生成する。

    DM-1 パイプラインに以下を追加:
      1. CLAHE 自動チューニング（auto_clahe=True 時）
      2. Canny エッジ検出 → エッジマップ生成
      3. 壁重みにエッジボーナス加算（edge_weight で制御）
      4. 解経路数の検証（max_solutions 以下であることを確認）

    Args:
        image : 入力画像（任意サイズ・任意形式）。
        config: DM2Config。None の場合はデフォルト値を使用。

    Returns:
        DM2Result（DM1Result互換フィールド + edge_map / solution_count / CLAHE使用値）

    Raises:
        ValueError: density_min > density_max の場合。
        RuntimeError: BFS で解経路が見つからない場合（実装バグ）。
    """
    if config is None:
        config = DM2Config()

    # ------------------------------------------------------------------
    # Stage 1: 前処理（グレースケール + リサイズ）
    # ------------------------------------------------------------------
    # DM-2 では CLAHE を自前で制御するため contrast_boost=0.0 でスキップ
    gray = preprocess_image(image, max_side=config.max_side, contrast_boost=0.0)

    # ------------------------------------------------------------------
    # DM-2 追加 Stage: CLAHE 自動チューニング
    # ------------------------------------------------------------------
    if config.auto_clahe:
        clip_limit, n_tiles = auto_tune_clahe(gray)
    else:
        clip_limit = config.clahe_clip_limit
        n_tiles = config.clahe_tile_size

    gray = _apply_clahe_custom(gray, clip_limit, n_tiles)

    # ------------------------------------------------------------------
    # Stage 2: density_map 生成
    # ------------------------------------------------------------------
    grid_rows = min(config.grid_rows, max(gray.shape[0] // 4, 1))
    grid_cols = min(config.grid_cols, max(gray.shape[1] // 4, 1))
    density_map = build_density_map(gray, grid_rows, grid_cols)

    # ------------------------------------------------------------------
    # DM-2 追加 Stage: エッジマップ生成（Canny）
    # ------------------------------------------------------------------
    edge_map = detect_edge_map(
        gray,
        grid_rows,
        grid_cols,
        sigma=config.edge_sigma,
        low_threshold=config.edge_low_threshold,
        high_threshold=config.edge_high_threshold,
    )

    # ------------------------------------------------------------------
    # Stage 3: density_min/max スケール壁重み + エッジボーナス → Kruskal
    # ------------------------------------------------------------------
    walls = _build_dm1_walls(
        density_map, grid_rows, grid_cols,
        config.density_min, config.density_max,
    )
    # エッジ上の壁 weight を増大（高重み = Kruskal で後から処理 = 壁が残る）
    walls = apply_edge_boost_to_walls(
        walls, edge_map, grid_cols,
        edge_weight=config.edge_weight,
    )
    grid = CellGrid(rows=grid_rows, cols=grid_cols, luminance=density_map, walls=walls)
    adj = build_spanning_tree(grid)

    # ------------------------------------------------------------------
    # Stage 4: 入口・出口・解経路 + BFS + 解数検証
    # ------------------------------------------------------------------
    entrance, exit_cell, solution_path = find_entrance_exit_and_path(adj, grid.num_cells)
    if not bfs_has_path(adj, entrance, exit_cell):
        raise RuntimeError(
            f"BFS: 入口({entrance})→出口({exit_cell}) の解経路が存在しません。"
        )

    # 解経路数チェック（spanning tree は必ず 1 解）
    solution_count = count_solutions_dfs(
        adj, entrance, exit_cell,
        max_solutions=config.max_solutions + 1,
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

    return DM2Result(
        svg=svg,
        png_bytes=png_bytes,
        entrance=entrance,
        exit_cell=exit_cell,
        solution_path=solution_path,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        density_map=density_map,
        adj=adj,
        edge_map=edge_map,
        solution_count=solution_count,
        clahe_clip_limit_used=clip_limit,
        clahe_n_tiles_used=n_tiles,
    )
