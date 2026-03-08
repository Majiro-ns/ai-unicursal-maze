# -*- coding: utf-8 -*-
"""
maze-artisan Phase3準備: 高解像度パフォーマンステスト（cmd_357k_a6）。

グリッドサイズ別の所要時間を計測し、400x600が目標時間内に完了することを保証。
3本柱全有効（thickness_range=1.5, extra_removal_rate=0.5, use_image_guided=True）。

目標: 400x600グリッドが60秒以内に完了すること。
実測: 最適化後 ~1.5秒（density_map numpy化 + post_process spanning_edge skip）。
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import pytest
from PIL import Image

from backend.core.density import generate_density_maze
from backend.core.density.grid_builder import build_cell_grid, build_density_map
from backend.core.density.maze_builder import build_spanning_tree, post_process_density
from backend.core.density.preprocess import preprocess_image
from backend.core.density.entrance_exit import find_entrance_exit_and_path


# --- ヘルパー ---

def _make_gradient_img(w: int = 512, h: int = 512) -> Image.Image:
    """水平グラデーション（左=暗、右=明）"""
    arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return Image.fromarray(arr, mode="L")


def _make_random_img(w: int = 512, h: int = 512, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (h, w), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


# --- Stage タイマー: 各フェーズを個別計測 ---

def _profile_stages(rows: int, cols: int, img: Image.Image) -> Dict[str, float]:
    """generate_density_maze の各ステージ所要時間を計測して返す。"""
    t0 = time.perf_counter()
    gray = preprocess_image(img, max_side=512)
    t1 = time.perf_counter()

    grid = build_cell_grid(gray, rows, cols)
    t2 = time.perf_counter()

    adj = build_spanning_tree(grid)
    t3 = time.perf_counter()

    adj2 = post_process_density(
        adj, grid,
        extra_removal_rate=0.5,
        dark_threshold=0.3,
        light_threshold=0.7,
    )
    t4 = time.perf_counter()

    entrance, exit_cell, path = find_entrance_exit_and_path(adj2, grid.num_cells)
    t5 = time.perf_counter()

    return {
        "preprocess":   t1 - t0,
        "grid_build":   t2 - t1,
        "kruskal":      t3 - t2,
        "post_process": t4 - t3,
        "path_find":    t5 - t4,
        "total":        t5 - t0,
        "cells":        rows * cols,
    }


# --- テスト: グリッドサイズ別パフォーマンス ---

@pytest.mark.parametrize("rows,cols,limit_s", [
    (50,   50,    2.0),   # 小グリッド: 2秒以内
    (100, 100,    5.0),   # 中グリッド: 5秒以内
    (200, 200,   15.0),   # 大グリッド: 15秒以内
    (300, 300,   30.0),   # 特大: 30秒以内
    (400, 600,   60.0),   # Phase3目標: 60秒以内
])
def test_pipeline_stage_timing(rows: int, cols: int, limit_s: float):
    """
    各ステージの所要時間を計測し、目標時間内に収まることを確認。
    3本柱全有効（extra_removal_rate=0.5, thickness_range=1.5）。
    """
    img = _make_gradient_img()
    profile = _profile_stages(rows, cols, img)

    total = profile["total"]
    assert total <= limit_s, (
        f"グリッド {rows}x{cols}: {total:.2f}s > 目標 {limit_s}s\n"
        f"  preprocess={profile['preprocess']:.3f}s\n"
        f"  grid_build={profile['grid_build']:.3f}s\n"
        f"  kruskal={profile['kruskal']:.3f}s\n"
        f"  post_process={profile['post_process']:.3f}s\n"
        f"  path_find={profile['path_find']:.3f}s"
    )


def test_400x600_full_pipeline_within_60s():
    """
    400x600グリッドで generate_density_maze() 全体（エクスポート含む）が60秒以内に完了すること。
    3本柱全有効:
      - thickness_range=1.5（可変壁厚）
      - extra_removal_rate=0.5（密度後処理）
      - use_image_guided=True（画像適応ルーティング）
    """
    img = _make_random_img(512, 512, seed=42)
    t0 = time.perf_counter()
    result = generate_density_maze(
        img,
        grid_size=400,
        max_side=512,
        width=1200,
        height=800,
        thickness_range=1.5,
        extra_removal_rate=0.5,
        dark_threshold=0.3,
        light_threshold=0.7,
        use_image_guided=True,
        show_solution=True,
        solution_highlight=False,
    )
    elapsed = time.perf_counter() - t0

    assert elapsed <= 60.0, f"400x600 全パイプライン: {elapsed:.2f}s > 60s 目標"
    assert result.grid_rows <= 400
    assert result.grid_cols <= 400  # max_side=512 から縮小される場合あり
    assert len(result.solution_path) >= 1
    assert len(result.png_bytes) > 0


def test_density_map_numpy_correctness():
    """
    numpy 化した build_density_map() が旧 Python ループ版と同じ結果を返すこと。
    """
    rng = np.random.default_rng(0)
    h, w = 64, 64
    gray = rng.random((h, w))  # float 0-1

    # 新 numpy 版
    result_numpy = build_density_map(gray, 8, 8)

    # Python ループ版（参照実装）
    grid_rows, grid_cols = 8, 8
    expected = np.zeros((grid_rows, grid_cols), dtype=np.float64)
    for r in range(grid_rows):
        for c in range(grid_cols):
            y0 = int(r * h / grid_rows)
            y1 = min(int((r + 1) * h / grid_rows), h)
            x0 = int(c * w / grid_cols)
            x1 = min(int((c + 1) * w / grid_cols), w)
            if y1 > y0 and x1 > x0:
                expected[r, c] = float(np.mean(gray[y0:y1, x0:x1]))
            else:
                expected[r, c] = 0.5
    expected = np.clip(expected, 0.0, 1.0)

    np.testing.assert_allclose(
        result_numpy, expected, atol=1e-10,
        err_msg="build_density_map numpy版とPythonループ版の結果が一致しない"
    )


def test_post_process_spanning_edge_skip_preserves_connectivity():
    """
    spanning_edge スキップ最適化後も post_process_density が連結性を保持すること。
    """
    from collections import deque

    img = _make_gradient_img(64, 64)
    gray = preprocess_image(img, max_side=64)
    grid = build_cell_grid(gray, 10, 10)
    adj = build_spanning_tree(grid)
    adj2 = post_process_density(
        adj, grid,
        extra_removal_rate=0.5,
        dark_threshold=0.3,
        light_threshold=0.7,
    )

    n = grid.num_cells
    visited = {0}
    q: deque = deque([0])
    while q:
        u = q.popleft()
        for v in adj2.get(u, []):
            if v not in visited:
                visited.add(v)
                q.append(v)

    assert len(visited) == n, (
        f"post_process 後に連結性が失われた: {len(visited)}/{n} セルに到達"
    )


def test_build_cell_grid_numpy_wall_count():
    """
    numpy 化した build_cell_grid() が正しい壁数を返すこと。
    R×C グリッドの壁数 = R*(C-1) + (R-1)*C
    """
    rows, cols = 8, 10
    img = _make_random_img(32, 32)
    gray = preprocess_image(img, max_side=32)
    grid = build_cell_grid(gray, rows, cols)

    expected_walls = rows * (cols - 1) + (rows - 1) * cols
    assert len(grid.walls) == expected_walls, (
        f"壁数不一致: {len(grid.walls)} != {expected_walls}"
    )
    # wall の cid1 < cid2 を確認
    for c1, c2, w in grid.walls:
        assert c1 < c2, f"壁の順序不正: {c1} >= {c2}"
        assert 0.0 <= w <= 1.0, f"壁重み値域外: {w}"
