# -*- coding: utf-8 -*-
"""
密度迷路 Phase 1 プロトタイプのテスト（M-12）。
01a §6.1 受け入れ基準: 入口1・出口1・解経路1本、連結性。
"""
from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image

from backend.core.density import generate_density_maze
from backend.core.density.grid_builder import build_cell_grid
from backend.core.density.maze_builder import build_spanning_tree
from backend.core.density.entrance_exit import find_entrance_exit_and_path


def _make_small_image(w: int = 64, h: int = 64) -> Image.Image:
    arr = np.ones((h, w), dtype=np.uint8) * 128
    return Image.fromarray(arr, mode="L")


def test_density_maze_returns_one_entrance_one_exit():
    """入口が1つ、出口が1つであること。"""
    img = _make_small_image(64, 64)
    result = generate_density_maze(img, grid_size=5, max_side=64)
    assert result.entrance >= 0
    assert result.exit_cell >= 0
    assert result.entrance < result.grid_rows * result.grid_cols
    assert result.exit_cell < result.grid_rows * result.grid_cols


def test_density_maze_solution_path_is_single_path():
    """解経路が1本（入口から出口までのセル列）であること。"""
    img = _make_small_image(64, 64)
    result = generate_density_maze(img, grid_size=6, max_side=64)
    path = result.solution_path
    assert len(path) >= 1
    assert path[0] == result.entrance
    assert path[-1] == result.exit_cell
    # 経路は重複なし（木の単純路）
    assert len(path) == len(set(path))


def test_density_maze_connectivity():
    """全セルが連結であること（spanning tree）。"""
    img = _make_small_image(32, 32)
    result = generate_density_maze(img, grid_size=4, max_side=32)
    from backend.core.density.preprocess import preprocess_image
    from backend.core.density.grid_builder import build_cell_grid
    gray = preprocess_image(img, max_side=32)
    grid = build_cell_grid(gray, result.grid_rows, result.grid_cols)
    adj = build_spanning_tree(grid)
    n = grid.num_cells
    # BFS で 0 から到達可能なセル数 = n なら連結
    visited = set()
    q = deque([0])
    visited.add(0)
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                q.append(v)
    assert len(visited) == n, "グラフが連結でない"


def test_density_maze_svg_and_png_non_empty():
    """SVG と PNG が空でないこと。"""
    img = _make_small_image(48, 48)
    result = generate_density_maze(img, grid_size=4, max_side=48)
    assert len(result.svg) > 100
    assert "svg" in result.svg.lower()
    assert len(result.png_bytes) > 100


def test_density_maze_1x1_grid():
    """1x1 グリッドでも落ちずに入口=出口=0、解経路=[0] で返ること。"""
    img = _make_small_image(8, 8)
    result = generate_density_maze(img, grid_size=1, max_side=8)
    assert result.grid_rows == 1 and result.grid_cols == 1
    assert result.entrance == 0 and result.exit_cell == 0
    assert result.solution_path == [0]
    assert len(result.svg) > 50
    assert len(result.png_bytes) > 50


def test_density_maze_unique_path_between_entrance_exit():
    """入口から出口までの経路がグラフ上で唯一であること（perfect maze）。"""
    from backend.core.density.preprocess import preprocess_image
    img = _make_small_image(32, 32)
    result = generate_density_maze(img, grid_size=3, max_side=32)
    gray = preprocess_image(img, max_side=32)
    grid = build_cell_grid(gray, result.grid_rows, result.grid_cols)
    adj = build_spanning_tree(grid)
    # 木なので任意2点間の経路は唯一。solution_path が入口→出口の経路になっているか確認
    entrance, exit_cell, path = find_entrance_exit_and_path(adj, grid.num_cells)
    assert path[0] == entrance
    assert path[-1] == exit_cell
    # 隣接性
    for i in range(len(path) - 1):
        assert path[i + 1] in adj.get(path[i], []), f"path[{i}] and path[{i+1}] are not adjacent"
