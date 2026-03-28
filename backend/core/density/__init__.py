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

from .dm5 import DM5Config, DM5Result, generate_dm5_maze, PRINT_FORMATS
from .dm6 import DM6Config, DM6Result, generate_dm6_maze, DIFFICULTY_PARAMS, VALID_DIFFICULTIES
from .dm6_optimizer import optimize_for_image, generate_preset, load_preset, save_preset, CATEGORY_CONSTRAINTS, VALID_CATEGORIES
from .entrance_exit import find_entrance_exit_and_path, find_entrance_exit_heuristic, find_image_guided_path
from .exporter import maze_to_png, maze_to_svg
from .grid_builder import build_cell_grid, build_cell_grid_with_edges, build_cell_grid_with_texture, CellGrid
from .maze_builder import build_spanning_tree, post_process_density
from .preprocess import preprocess_image


# masterpiece 黄金設定（SSIM探索 cmd_358k_a2 結果: grid_size=8 が SSIM と視認性のバランス点）
# P3 (cmd_368k_a8_p3): thickness_range 1.5 → 2.5 に拡張（G1 線幅最適化）
MASTERPIECE_PRESET: dict = {
    "grid_size": 8,
    "thickness_range": 2.5,
    "extra_removal_rate": 0.5,
    "dark_threshold": 0.3,
    "light_threshold": 0.7,
    "use_image_guided": True,
    "solution_highlight": False,
    "show_solution": False,
    "edge_weight": 0.5,
    "stroke_width": 2.0,
    "wall_color_min": 40,
    "wall_color_max": 175,
    "variable_cell_size": True,
    "use_gradient_walls": True,  # Phase 4: SVG linearGradient 壁色グラデーション
}

# Path-First Masterpiece V2: 経路優先パイプライン（I4+F3+G1+H2+K1）
# V6: no-shortcut spanning tree — BFS solution IS the image-tracing path
MASTERPIECE_V2_PRESET: dict = {
    **MASTERPIECE_PRESET,
    "use_path_first": True,
    "grid_size": 50,         # override grid_size=8 for image tracing resolution
    "dark_threshold": 0.3,
    "bright_threshold": 0.7,
    "path_thickness_dark": 6.0,
    "path_thickness_bright": 1.0,
}


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
    # 壁色コントラスト保証: 背景(255)との差≥(255-wall_color_max)
    wall_color_min: int = 40,
    wall_color_max: int = 175,
    # MAZE-Q2: Xu-Kaplan 可変セルサイズ（False=均一・後方互換デフォルト）
    variable_cell_size: bool = False,
    # Phase 4: SVG linearGradient 壁色グラデーション（False=既存挙動・後方互換）
    use_gradient_walls: bool = False,
    # Path-First Masterpiece V2（I4+F3+G1+H2+K1）
    use_path_first: bool = False,
    bright_threshold: float = 0.7,
    path_thickness_dark: float = 6.0,
    path_thickness_bright: float = 1.0,
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
            variable_cell_size=variable_cell_size,
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
                variable_cell_size=variable_cell_size,
            )
        else:
            grid = build_cell_grid(gray, grid_rows, grid_cols, density_factor=density_factor,
                                   variable_cell_size=variable_cell_size)

    if use_path_first:
        # === Path-First Masterpiece V2 Pipeline (I4+F3+G1+H2+K1) ===
        from .edge_enhancer import detect_edge_map, extract_edge_waypoints
        from .path_designer import (
            classify_cells,
            find_dark_blobs,
            find_entrance_exit_path_first,
            order_blobs_for_path,
            design_masterpiece_path,
            build_walls_around_path,
        )

        # K1: Edge detection for waypoints
        edge_map = detect_edge_map(
            gray, grid_rows, grid_cols,
            sigma=edge_sigma,
            low_threshold=edge_low_threshold,
            high_threshold=edge_high_threshold,
        )
        edge_waypoints = extract_edge_waypoints(edge_map, grid_cols, threshold=0.3)

        # Classify cells and find dark blobs
        cell_classes = classify_cells(
            grid.luminance,
            dark_thresh=dark_threshold,
            bright_thresh=bright_threshold,
        )
        blobs = find_dark_blobs(cell_classes, grid)

        # Find entrance/exit
        entrance, exit_cell = find_entrance_exit_path_first(grid, cell_classes)

        # Order blobs and design path (F3 serpentine fill)
        ordered_blobs = order_blobs_for_path(blobs, entrance, exit_cell, grid)
        solution_path, path_edges = design_masterpiece_path(
            grid, cell_classes, ordered_blobs, entrance, exit_cell, edge_waypoints
        )

        # I4: Build walls around designed path (V6 no-shortcut)
        adj = build_walls_around_path(
            grid, path_edges, cell_classes,
            extra_removal_rate=extra_removal_rate,
            solution_cells=set(solution_path),
        )
    else:
        # === Original Pipeline (V1) ===
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

    # G1: Pass luminance info for per-segment variable path width
    cell_luminance = grid.luminance if use_path_first else None

    svg = maze_to_svg(
        grid, adj, entrance, exit_cell, solution_path,
        width=width, height=height,
        stroke_width=stroke_width,
        show_solution=show_solution,
        thickness_range=thickness_range,
        solution_highlight=solution_highlight,
        stroke_quantize_levels=stroke_quantize_levels,
        wall_color_min=wall_color_min,
        wall_color_max=wall_color_max,
        use_gradient_walls=use_gradient_walls,
        cell_luminance=cell_luminance,
        path_thickness_dark=path_thickness_dark,
        path_thickness_bright=path_thickness_bright,
    )
    png_bytes = maze_to_png(
        grid, adj, entrance, exit_cell, solution_path,
        width=width, height=height,
        stroke_width=int(max(1, stroke_width)),
        show_solution=show_solution,
        thickness_range=thickness_range,
        solution_highlight=solution_highlight,
        dpi=png_dpi,
        wall_color_min=wall_color_min,
        wall_color_max=wall_color_max,
        cell_luminance=cell_luminance,
        path_thickness_dark=path_thickness_dark,
        path_thickness_bright=path_thickness_bright,
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
