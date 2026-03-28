"""
DM-4: 多値トーン壁表現 — 明示的インターフェース（Phase DM-4）

設計書: docs/maze-artisan-masterpiece-requirements.md Phase DM-4

DM-2 との差分:
  + 8 段階グレースケール壁（TONAL_GRADES: [0,36,73,109,146,182,219,255]）
  + アンチエイリアス（render_scale 倍描画 → LANCZOS 縮小）
  + 壁厚変動強化（暗部を太く → 黒ピクセル密度増加 → SSIM 向上）
  + SSIM スコア計算（vs 入力画像）
  + dark_coverage 計算（画素値 < 128 の割合）

成功基準:
  gradient SSIM ≥ 0.70 / circle SSIM ≥ 0.65 / dark_coverage ≥ 0.75（黒画像）
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .dm2 import DM2Config, DM2Result, _apply_clahe_custom, auto_tune_clahe
from .dm1 import _build_dm1_walls
from .edge_enhancer import apply_edge_boost_to_walls, detect_edge_map
from .entrance_exit import find_entrance_exit_and_path
from .grid_builder import CellGrid, build_density_map
from .maze_builder import build_spanning_tree
from .preprocess import preprocess_image
from .solver import bfs_has_path, count_solutions_dfs
from .tonal_exporter import (
    TONAL_GRADES,
    compute_dark_coverage,
    maze_to_png_tonal,
    maze_to_svg_tonal,
)


# ---------------------------------------------------------------------------
# SSIM 計算
# ---------------------------------------------------------------------------

def _compute_ssim(
    original_gray: np.ndarray,
    png_bytes: bytes,
    target_size: Tuple[int, int] = (256, 256),
) -> float:
    """
    元画像グレースケール配列（float 0.0〜1.0）と迷路 PNG の SSIM を計算する。

    両画像を target_size にリサイズしてから skimage.metrics.structural_similarity を適用。
    skimage が利用不可の場合は 0.0 を返す。
    """
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return 0.0

    orig_pil = Image.fromarray(
        (np.clip(original_gray, 0.0, 1.0) * 255).astype(np.uint8)
    ).convert("L")
    orig_arr = np.asarray(
        orig_pil.resize(target_size, Image.LANCZOS), dtype=np.float64
    ) / 255.0

    maze_pil = Image.open(io.BytesIO(png_bytes)).convert("L")
    maze_arr = np.asarray(
        maze_pil.resize(target_size, Image.LANCZOS), dtype=np.float64
    ) / 255.0

    return float(structural_similarity(orig_arr, maze_arr, data_range=1.0))


# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

@dataclass
class DM4Config(DM2Config):
    """
    DM-4 設定。DM2Config を継承し、多値トーンレンダリングパラメータを追加。

    主要パラメータ:
        tonal_grades        : 壁色パレット（デフォルト: TONAL_GRADES の 8 値）
        render_scale        : アンチエイリアス倍率（1=なし, 2=2倍→LANCZOS縮小）
        tonal_thickness_range: 暗部での壁厚追加比率（0=均一, 1.0=暗部で2倍）
        ssim_target_size    : SSIM 計算時のリサイズ解像度
    """
    # 8 段階グレースケールパレット
    tonal_grades: List[int] = field(
        default_factory=lambda: list(TONAL_GRADES)
    )
    # アンチエイリアス倍率
    render_scale: int = 2
    # 壁厚変動範囲（暗部: base * (1 + range)、明部: base * 1.0）
    # DM-4 では暗部を太く描画して黒ピクセル密度を増加させ SSIM を向上させる
    tonal_thickness_range: float = 2.0
    # SSIM 計算ターゲットサイズ
    ssim_target_size: Tuple[int, int] = (256, 256)
    # fill_cells=True: セル塗りつぶし+通路彫り込み+blur（高 SSIM 手法）
    # fill_cells=False: 白背景+壁線描画（従来手法）
    fill_cells: bool = True
    # Gaussian blur 半径（fill_cells=True 時のみ有効。0=無効）
    blur_radius: float = 2.0


# ---------------------------------------------------------------------------
# 結果
# ---------------------------------------------------------------------------

@dataclass
class DM4Result(DM2Result):
    """DM-4 生成結果。DM-2 互換フィールド + SSIM + dark_coverage。"""
    # SSIM スコア（vs 入力画像、生成時に計算）
    ssim_score: float = 0.0
    # 画素値 < 128 の割合（暗部ピクセル密度の指標）
    dark_coverage: float = 0.0


# ---------------------------------------------------------------------------
# メイン API
# ---------------------------------------------------------------------------

def generate_dm4_maze(
    image: Image.Image,
    config: Optional[DM4Config] = None,
) -> DM4Result:
    """
    DM-4 多値トーン迷路を生成する。

    DM-2 パイプライン（前処理 / CLAHE / エッジ検出 / Kruskal）で迷路グラフを生成し、
    tonal_exporter の 8 段階グレースケール + アンチエイリアスレンダラーで描画する。
    生成後に SSIM スコアと dark_coverage を計算して DM4Result に格納する。

    Args:
        image : 入力画像（任意サイズ・任意形式）。
        config: DM4Config。None の場合はデフォルト値を使用。

    Returns:
        DM4Result（DM2Result 互換フィールド + ssim_score + dark_coverage）

    Raises:
        ValueError: density_min > density_max の場合。
        RuntimeError: BFS で解経路が見つからない場合（実装バグ）。
    """
    if config is None:
        config = DM4Config()

    # ------------------------------------------------------------------
    # Stage 1: 前処理（グレースケール + リサイズ、CLAHE は自前で制御）
    # ------------------------------------------------------------------
    gray = preprocess_image(image, max_side=config.max_side, contrast_boost=0.0)

    # ------------------------------------------------------------------
    # CLAHE 自動チューニング（DM-2 継承）
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
    # DM-2 継承: エッジマップ生成
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
    # Stage 3: 壁重み設定 + エッジボーナス + Kruskal
    # ------------------------------------------------------------------
    walls = _build_dm1_walls(
        density_map, grid_rows, grid_cols,
        config.density_min, config.density_max,
    )
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

    solution_count = count_solutions_dfs(
        adj, entrance, exit_cell,
        max_solutions=config.max_solutions + 1,
    )

    # ------------------------------------------------------------------
    # Stage 5: DM-4 トーンレンダリング（8 段階グレー + アンチエイリアス）
    # ------------------------------------------------------------------
    out_w = grid_cols * config.cell_size_px
    out_h = grid_rows * config.cell_size_px

    png_bytes = maze_to_png_tonal(
        grid, adj, entrance, exit_cell, solution_path,
        width=out_w,
        height=out_h,
        show_solution=config.show_solution,
        render_scale=config.render_scale,
        grades=list(config.tonal_grades),
        wall_thickness_base=1.5,
        thickness_range=config.tonal_thickness_range,
        fill_cells=config.fill_cells,
        blur_radius=config.blur_radius,
    )

    svg = maze_to_svg_tonal(
        grid, adj, entrance, exit_cell, solution_path,
        width=out_w,
        height=out_h,
        show_solution=config.show_solution,
        grades=list(config.tonal_grades),
        wall_thickness_base=1.5,
        thickness_range=config.tonal_thickness_range,
    )

    # ------------------------------------------------------------------
    # SSIM・dark_coverage 計算
    # ------------------------------------------------------------------
    ssim_score = _compute_ssim(gray, png_bytes, target_size=config.ssim_target_size)
    dark_cov = compute_dark_coverage(png_bytes, threshold=128)

    return DM4Result(
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
        ssim_score=ssim_score,
        dark_coverage=dark_cov,
    )
