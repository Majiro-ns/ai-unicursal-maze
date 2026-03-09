# -*- coding: utf-8 -*-
"""
I4 Reverse Pipeline テスト (DM-1 Part A).

generate_i4_maze() の動作と I4MazeResult の正当性を検証する。

テスト方針:
  - すべて合成画像（PIL.Image）を tmp_path 経由で使用（実画像不要）
  - 小グリッド（10〜20行×12〜20列）で高速実行
  - 暗部=左半分, 明部=右半分の勾配画像で CLAHE 影響を抑える
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pytest
from PIL import Image

from backend.core.density.i4_pipeline import I4MazeResult, generate_i4_maze


# ---------------------------------------------------------------------------
# テスト用ヘルパー
# ---------------------------------------------------------------------------

def _gradient_image(w: int = 32, h: int = 32) -> Image.Image:
    """
    左半分=暗(30), 右半分=明(220) のグレースケール画像。
    ノイズを加えて CLAHE が均一画像と判定しないようにする。
    """
    rng = np.random.default_rng(42)
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[:, : w // 2] = rng.integers(20, 40, size=(h, w // 2), dtype=np.uint8)
    arr[:, w // 2 :] = rng.integers(210, 230, size=(h, w - w // 2), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def _run(
    tmp_path,
    w: int = 16,
    h: int = 12,
    img: Image.Image | None = None,
) -> I4MazeResult:
    """合成画像を tmp_path に保存して generate_i4_maze を実行する。"""
    if img is None:
        img = _gradient_image()
    p = tmp_path / "test.png"
    img.save(str(p))
    # contrast_boost=0.0 で CLAHE をオフにして勾配を安定させる
    return generate_i4_maze(
        str(p),
        grid_width=w,
        grid_height=h,
        cell_size_px=3,
        contrast_boost=0.0,
    )


# ---------------------------------------------------------------------------
# CR-1: 型 / 基本フィールド検証
# ---------------------------------------------------------------------------

def test_i4_pipeline_produces_valid_maze(tmp_path):
    """generate_i4_maze は I4MazeResult インスタンスを返し、基本フィールドが正しい。"""
    result = _run(tmp_path)
    assert isinstance(result, I4MazeResult)
    assert result.grid_width == 16
    assert result.grid_height == 12
    assert result.cell_size_px == 3
    assert isinstance(result.walls, set)
    assert isinstance(result.solution_path, list)
    assert isinstance(result.density_map, np.ndarray)


def test_i4_maze_result_all_fields_present(tmp_path):
    """I4MazeResult の全フィールドが None でなく適切な型を持つ。"""
    result = _run(tmp_path)
    # walls: set of 2-tuples of 2-tuples
    for wall in list(result.walls)[:5]:
        assert len(wall) == 2
        assert len(wall[0]) == 2
        assert len(wall[1]) == 2
    # solution_path: list of (row, col) tuples
    for rc in result.solution_path[:5]:
        assert len(rc) == 2
    # density_map: 2-D array [0,1]
    assert result.density_map.ndim == 2
    # entrance / exit_pos: (row, col) int tuples
    assert len(result.entrance) == 2
    assert len(result.exit_pos) == 2


# ---------------------------------------------------------------------------
# CR-2: 入口・出口検証
# ---------------------------------------------------------------------------

def test_entrance_exit_exist(tmp_path):
    """入口・出口はグリッド範囲内で、互いに異なる座標を持つ。"""
    result = _run(tmp_path)
    r_ent, c_ent = result.entrance
    r_ext, c_ext = result.exit_pos
    assert 0 <= r_ent < result.grid_height, f"entrance row {r_ent} out of bounds"
    assert 0 <= c_ent < result.grid_width, f"entrance col {c_ent} out of bounds"
    assert 0 <= r_ext < result.grid_height, f"exit row {r_ext} out of bounds"
    assert 0 <= c_ext < result.grid_width, f"exit col {c_ext} out of bounds"
    assert result.entrance != result.exit_pos, "entrance and exit must differ"


def test_solution_path_starts_at_entrance(tmp_path):
    """solution_path[0] == entrance でなければならない。"""
    result = _run(tmp_path)
    assert result.solution_path[0] == result.entrance, (
        f"Path starts at {result.solution_path[0]}, expected entrance {result.entrance}"
    )


def test_solution_path_ends_at_exit(tmp_path):
    """solution_path[-1] == exit_pos でなければならない。"""
    result = _run(tmp_path)
    assert result.solution_path[-1] == result.exit_pos, (
        f"Path ends at {result.solution_path[-1]}, expected exit {result.exit_pos}"
    )


# ---------------------------------------------------------------------------
# CR-3: 解経路の構造検証
# ---------------------------------------------------------------------------

def test_solution_path_nonempty(tmp_path):
    """解経路は少なくとも 2 セル（入口 + 出口）を含む。"""
    result = _run(tmp_path)
    assert len(result.solution_path) >= 2, (
        f"solution_path has only {len(result.solution_path)} cell(s)"
    )


def test_solution_path_4neighbor_steps(tmp_path):
    """解経路の連続するセルはすべて 4 近傍（マンハッタン距離 = 1）。"""
    result = _run(tmp_path)
    path = result.solution_path
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        dist = abs(r1 - r2) + abs(c1 - c2)
        assert dist == 1, (
            f"Step {i}: ({r1},{c1}) -> ({r2},{c2}) is not 4-neighbor (dist={dist})"
        )


def test_solution_path_no_revisit(tmp_path):
    """解経路にセルの重複はない（ループなし単純パス）。"""
    result = _run(tmp_path)
    path = result.solution_path
    assert len(path) == len(set(path)), (
        f"Duplicate cells in solution_path: {len(path)} steps, {len(set(path))} unique"
    )


# ---------------------------------------------------------------------------
# CR-4: 壁配置の正当性
# ---------------------------------------------------------------------------

def test_wall_placement_respects_path(tmp_path):
    """解経路の連続セル間に壁は存在しない（経路が塞がれていない）。"""
    result = _run(tmp_path)
    path = result.solution_path
    walls = result.walls
    cols = result.grid_width

    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        # 正規化: セル ID 小さい方を先に
        cid1 = r1 * cols + c1
        cid2 = r2 * cols + c2
        if cid1 <= cid2:
            w = ((r1, c1), (r2, c2))
        else:
            w = ((r2, c2), (r1, c1))
        assert w not in walls, (
            f"Wall found between consecutive path cells {path[i]} -> {path[i+1]}"
        )


def test_walls_count_in_valid_range(tmp_path):
    """壁数は (0, 最大隣接ペア数) の範囲内。完全壁なし/完全壁ありは不正。"""
    result = _run(tmp_path)
    rows, cols = result.grid_height, result.grid_width
    max_walls = rows * (cols - 1) + (rows - 1) * cols
    assert 0 < len(result.walls) < max_walls, (
        f"Wall count {len(result.walls)} out of expected range (0, {max_walls})"
    )


# ---------------------------------------------------------------------------
# CR-5: 密度マップ
# ---------------------------------------------------------------------------

def test_density_map_shape_and_range(tmp_path):
    """density_map は (grid_height, grid_width) の形状で値が [0, 1] に収まる。"""
    result = _run(tmp_path)
    assert result.density_map.shape == (result.grid_height, result.grid_width), (
        f"Expected shape ({result.grid_height}, {result.grid_width}), "
        f"got {result.density_map.shape}"
    )
    assert float(result.density_map.min()) >= 0.0
    assert float(result.density_map.max()) <= 1.0


# ---------------------------------------------------------------------------
# CR-6: 連結性検証
# ---------------------------------------------------------------------------

def test_maze_is_connected(tmp_path):
    """入口から BFS ですべてのセルに到達できる（迷路が連結）。"""
    result = _run(tmp_path, w=12, h=10)
    rows, cols = result.grid_height, result.grid_width
    n = rows * cols

    # 全隣接ペアを通路とし、walls を差し引いて隣接グラフを構築
    adj: dict = {i: set() for i in range(n)}
    for r in range(rows):
        for c in range(cols):
            cid = r * cols + c
            if c + 1 < cols:
                cid2 = r * cols + (c + 1)
                adj[cid].add(cid2)
                adj[cid2].add(cid)
            if r + 1 < rows:
                cid2 = (r + 1) * cols + c
                adj[cid].add(cid2)
                adj[cid2].add(cid)

    for (r1, c1), (r2, c2) in result.walls:
        cid1 = r1 * cols + c1
        cid2 = r2 * cols + c2
        adj[cid1].discard(cid2)
        adj[cid2].discard(cid1)

    # BFS from entrance
    ent_r, ent_c = result.entrance
    start = ent_r * cols + ent_c
    visited = {start}
    queue: deque = deque([start])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                queue.append(v)

    assert len(visited) == n, (
        f"Only {len(visited)}/{n} cells reachable from entrance "
        f"({result.entrance}) — maze is not connected"
    )


# ---------------------------------------------------------------------------
# CR-7: 暗部セルの経路密度検証
# ---------------------------------------------------------------------------

def test_dark_cells_have_dense_paths(tmp_path):
    """
    暗部セル（左半分）の経路密度 > 明部セル（右半分）の経路密度。

    経路密度 = その領域で経路が通るセルの割合。
    F3 serpentine fill が暗部を優先的にカバーすることを検証する。
    """
    result = _run(tmp_path, w=20, h=14)
    path_set = set(result.solution_path)
    cols = result.grid_width
    rows = result.grid_height
    half = cols // 2

    dark_total = rows * half
    bright_total = rows * (cols - half)

    dark_in_path = sum(
        1 for r in range(rows) for c in range(half) if (r, c) in path_set
    )
    bright_in_path = sum(
        1 for r in range(rows) for c in range(half, cols) if (r, c) in path_set
    )

    dark_density = dark_in_path / dark_total if dark_total > 0 else 0.0
    bright_density = bright_in_path / bright_total if bright_total > 0 else 0.0

    assert dark_density >= bright_density, (
        f"Dark density {dark_density:.3f} should be >= bright density "
        f"{bright_density:.3f} — F3 should prioritize dark cells"
    )
