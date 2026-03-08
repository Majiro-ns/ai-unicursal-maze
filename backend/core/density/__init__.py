"""
密度迷路 Phase 1/2: 濃度マップ → グリッド → Kruskal → 入口・出口 → SVG/PNG。

Phase 2 追加パラメータ:
  preset      : "generic" / "face" / "landscape"（テクスチャプリセット）
  n_segments  : K-means クラスタ数（デフォルト 4）
  use_texture : True でテクスチャ・セグメンテーション機能を有効化
  use_heuristic: True で Phase 2 解ヒューリスティクスを使用
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .entrance_exit import find_entrance_exit_and_path, find_entrance_exit_heuristic, find_image_guided_path
from .exporter import maze_to_png, maze_to_svg
from .grid_builder import build_cell_grid, build_cell_grid_with_edges, build_cell_grid_with_texture, CellGrid
from .maze_builder import build_spanning_tree, post_process_density
from .preprocess import preprocess_image


@dataclass
class DensityMazeResult:
    maze_id: str
    svg: str
    png_bytes: bytes
    entrance: int          # セル id
    exit_cell: int
    solution_path: List[int]   # セル id のリスト
    grid_rows: int
    grid_cols: int
    # Phase 2 追加フィールド
    segment_map: Optional[np.ndarray] = field(default=None)   # (H,W) int ラベルマップ
    texture_map: Optional[np.ndarray] = field(default=None)   # (rows,cols) TextureType


def generate_density_maze(
    image: Image.Image,
    *,
    grid_size: int = 50,
    max_side: int = 512,
    width: int = 800,
    height: int = 600,
    stroke_width: float = 2.0,
    show_solution: bool = True,
    density_factor: float = 1.0,
    maze_id: Optional[str] = None,
    # Phase 2 パラメータ
    preset: str = "generic",
    n_segments: int = 4,
    use_texture: bool = False,
    use_heuristic: bool = False,
    bias_strength: float = 0.5,
    # Phase 2 Stage 4: エッジ強調
    edge_weight: float = 0.0,
    edge_sigma: float = 1.0,
    edge_low_threshold: float = 0.05,
    edge_high_threshold: float = 0.20,
    # Phase 2 CLAHEコントラスト補正
    contrast_boost: float = 1.0,
    # Phase 2 可変壁厚（masterpiece柱1）
    thickness_range: float = 1.5,
    # 解経路描画モード: False=masterpiece白線 / True=デバッグ用オレンジ+マーカー
    solution_highlight: bool = False,
    # Phase 2b: 解経路画像適応ルーティング（masterpiece柱3）
    use_image_guided: bool = False,
    # Phase 2b: 密度制御（ループ許容）
    extra_removal_rate: float = 0.0,
    dark_threshold: float = 0.3,
    light_threshold: float = 0.7,
    # Phase 3 SVG品質改善: stroke-width 量子化レベル（0=無効, 推奨=20）
    stroke_quantize_levels: int = 20,
    # Phase 3 PNG DPI: None=メタデータなし, 300=印刷用, 96=Web用
    png_dpi: Optional[int] = None,
) -> DensityMazeResult:
    """
    密度迷路パイプライン（Phase 1/2 共用）。

    use_texture=False（デフォルト）: Phase 1 相当（後方互換）。
    use_texture=True: Phase 2 — セグメンテーション + テクスチャ割り当て。
    use_heuristic=True: Phase 2 — 解ヒューリスティクスで美しい解経路を選択。
    edge_weight>0: Phase 2 Stage 4 — Canny エッジ検出で輪郭部分の壁を保持。
    """
    import uuid

    mid = maze_id or f"density-{uuid.uuid4().hex[:8]}"
    gray = preprocess_image(image, max_side=max_side, contrast_boost=contrast_boost)
    grid_rows = min(grid_size, max(gray.shape[0] // 4, 1))
    grid_cols = min(grid_size, max(gray.shape[1] // 4, 1))

    segment_map: Optional[np.ndarray] = None
    texture_map: Optional[np.ndarray] = None

    if use_texture:
        # Phase 2: セグメンテーション + テクスチャ割り当て
        from .segment import segment_by_luminance
        from .texture import (
            assign_cell_textures,
            compute_gradient_angles,
            PRESET_FACE,
            PRESET_GENERIC,
            PRESET_LANDSCAPE,
        )

        preset_map: Dict = {
            "face": PRESET_FACE,
            "landscape": PRESET_LANDSCAPE,
        }.get(preset, PRESET_GENERIC)

        segment_map = segment_by_luminance(gray, n_clusters=n_segments)
        texture_map = assign_cell_textures(
            CellGrid(rows=grid_rows, cols=grid_cols,
                     luminance=np.zeros((grid_rows, grid_cols)),
                     walls=[]),
            segment_map,
            preset_map,
        )
        gradient_angles = compute_gradient_angles(gray, grid_rows, grid_cols)

        grid = build_cell_grid_with_texture(
            gray, grid_rows, grid_cols,
            cell_textures=texture_map,
            gradient_angles=gradient_angles,
            density_factor=density_factor,
            bias_strength=bias_strength,
        )
    else:
        # Phase 1 相当（+ エッジ強調オプション）
        if edge_weight > 0.0:
            grid = build_cell_grid_with_edges(
                gray, grid_rows, grid_cols,
                density_factor=density_factor,
                edge_weight=edge_weight,
                edge_sigma=edge_sigma,
                edge_low_threshold=edge_low_threshold,
                edge_high_threshold=edge_high_threshold,
            )
        else:
            grid = build_cell_grid(gray, grid_rows, grid_cols, density_factor=density_factor)

    adj = build_spanning_tree(grid)

    # Phase 2b: ループ許容密度後処理
    if extra_removal_rate > 0.0 or light_threshold < 1.0:
        adj = post_process_density(
            adj,
            grid,
            extra_removal_rate=extra_removal_rate,
            dark_threshold=dark_threshold,
            light_threshold=light_threshold,
        )

    if use_image_guided:
        # Phase 2b: 画像適応ルーティング（明部を通る Dijkstra 最短経路）
        entrance, exit_cell, solution_path = find_image_guided_path(
            adj, grid.num_cells, grid.luminance, grid_rows, grid_cols
        )
    elif use_heuristic:
        entrance, exit_cell, solution_path = find_entrance_exit_heuristic(
            adj, grid.num_cells, grid.luminance
        )
    else:
        entrance, exit_cell, solution_path = find_entrance_exit_and_path(adj, grid.num_cells)

    svg = maze_to_svg(
        grid, adj, entrance, exit_cell, solution_path,
        width=width, height=height,
        stroke_width=stroke_width,
        show_solution=show_solution,
        thickness_range=thickness_range,
        solution_highlight=solution_highlight,
        stroke_quantize_levels=stroke_quantize_levels,
    )
    png_bytes = maze_to_png(
        grid, adj, entrance, exit_cell, solution_path,
        width=width, height=height,
        stroke_width=int(max(1, stroke_width)),
        show_solution=show_solution,
        thickness_range=thickness_range,
        solution_highlight=solution_highlight,
        dpi=png_dpi,
    )

    return DensityMazeResult(
        maze_id=mid,
        svg=svg,
        png_bytes=png_bytes,
        entrance=entrance,
        exit_cell=exit_cell,
        solution_path=solution_path,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        segment_map=segment_map,
        texture_map=texture_map,
    )
