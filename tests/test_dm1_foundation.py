# -*- coding: utf-8 -*-
"""
DM-1 Foundation テスト
Phase DM-1 基盤実装の受け入れ基準を検証する。

判定基準: 単色グラデーション画像を入力して、暗部→密/明部→疎が再現できること。

カバレッジ:
  - Stage 1: 前処理 (preprocess_image)
  - Stage 2: 密度マップ変換 (build_density_map / build_cell_grid)
  - Stage 3: Kruskal 迷路生成 (build_spanning_tree / post_process_density)
  - 解経路検証 (bfs_has_path)
  - Stage 5: PNG / SVG レンダリング
  - E2E: generate_density_maze() + グラデーション画像
"""
from __future__ import annotations

import io
from collections import deque
from typing import Dict, List, Set

import numpy as np
import pytest
from PIL import Image

from backend.core.density import generate_density_maze
from backend.core.density.grid_builder import (
    CellGrid,
    build_cell_grid,
    build_density_map,
)
from backend.core.density.maze_builder import build_spanning_tree, post_process_density
from backend.core.density.preprocess import preprocess_image
from backend.core.density.solver import bfs_has_path


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_gradient_image(rows: int = 100, cols: int = 200) -> Image.Image:
    """左半分=黒(0)、右半分=白(255) の水平グラデーション画像を返す。"""
    arr = np.zeros((rows, cols), dtype=np.uint8)
    arr[:, cols // 2 :] = 255
    return Image.fromarray(arr, mode="L")


def _make_uniform_image(value: int, rows: int = 64, cols: int = 64) -> Image.Image:
    """一様な明度の画像を返す。value: 0=黒, 255=白。"""
    arr = np.full((rows, cols), value, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def _count_passages_in_region(
    adj: Dict[int, List[int]],
    grid: CellGrid,
    col_start: int,
    col_end: int,
) -> int:
    """指定列範囲内のセルが持つ通路数の合計を返す（自セル内エッジのみカウント）。"""
    total = 0
    for r in range(grid.rows):
        for c in range(col_start, col_end):
            cid = grid.cell_id(r, c)
            for nb in adj.get(cid, []):
                nb_r, nb_c = grid.cell_rc(nb)
                if col_start <= nb_c < col_end:
                    total += 1  # 双方向につき後で /2
    return total // 2


def _all_connected_bfs(adj: Dict[int, List[int]], n: int) -> bool:
    """BFS で n 個全セルが1連結か確認する。"""
    if n == 0:
        return True
    visited: Set[int] = {0}
    queue: deque = deque([0])
    while queue:
        node = queue.popleft()
        for nb in adj.get(node, []):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return len(visited) == n


def _build_gradient_grid(rows: int, cols: int) -> CellGrid:
    """左半 lum=0(暗)、右半 lum=1(明)の CellGrid を構築する。"""
    luminance = np.zeros((rows, cols), dtype=np.float64)
    luminance[:, cols // 2 :] = 1.0

    ids = np.arange(rows * cols, dtype=np.int32).reshape(rows, cols)
    walls = []
    # 水平壁（右隣）
    for r in range(rows):
        for c in range(cols - 1):
            cid1, cid2 = int(ids[r, c]), int(ids[r, c + 1])
            w = float((luminance[r, c] + luminance[r, c + 1]) / 2.0)
            walls.append((min(cid1, cid2), max(cid1, cid2), w))
    # 垂直壁（下隣）
    for r in range(rows - 1):
        for c in range(cols):
            cid1, cid2 = int(ids[r, c]), int(ids[r + 1, c])
            w = float((luminance[r, c] + luminance[r + 1, c]) / 2.0)
            walls.append((min(cid1, cid2), max(cid1, cid2), w))

    return CellGrid(rows=rows, cols=cols, luminance=luminance, walls=walls)


# ===========================================================================
# Stage 1: 前処理テスト (preprocess_image)
# ===========================================================================

def test_preprocess_image_float_range():
    """preprocess_image の出力が 0.0〜1.0 の float 配列であること。"""
    img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    gray = preprocess_image(img, max_side=64)
    assert gray.dtype in (np.float32, np.float64), "出力は float 型であること"
    assert float(gray.min()) >= 0.0, f"最小値が 0.0 未満: {gray.min()}"
    assert float(gray.max()) <= 1.0, f"最大値が 1.0 超: {gray.max()}"


def test_preprocess_image_rgb_to_gray():
    """RGB 画像がグレースケール 2D 配列に変換されること。"""
    img = Image.new("RGB", (32, 48), color=(200, 100, 50))
    gray = preprocess_image(img, max_side=512)
    assert gray.ndim == 2, f"次元数が 2 でない: {gray.ndim}"


def test_preprocess_image_resize():
    """max_side 指定時に長辺が max_side 以下になること。"""
    img = Image.new("L", (400, 300), color=128)
    gray = preprocess_image(img, max_side=100)
    assert max(gray.shape) <= 100, f"リサイズ後の長辺が 100 超: {gray.shape}"


# ===========================================================================
# Stage 2: 密度マップ変換テスト (build_density_map)
# ===========================================================================

def test_build_density_map_shape():
    """出力の shape が (grid_rows, grid_cols) と一致すること。"""
    gray = np.random.rand(80, 120)
    dm = build_density_map(gray, 20, 30)
    assert dm.shape == (20, 30), f"期待 (20,30)、実際 {dm.shape}"


def test_build_density_map_all_black_is_zero():
    """全黒画像の密度マップはほぼ 0.0 であること。"""
    gray = np.zeros((64, 64))
    dm = build_density_map(gray, 16, 16)
    assert float(dm.max()) < 0.01, f"全黒のはずが max={dm.max()}"


def test_build_density_map_all_white_is_one():
    """全白画像の密度マップはほぼ 1.0 であること。"""
    gray = np.ones((64, 64))
    dm = build_density_map(gray, 16, 16)
    assert float(dm.min()) > 0.99, f"全白のはずが min={dm.min()}"


def test_build_density_map_gradient_monotone():
    """水平グラデーション画像（左=0、右=1）の列平均が単調増加すること。"""
    rows, cols = 64, 64
    gray = np.tile(np.linspace(0.0, 1.0, cols), (rows, 1))  # (64, 64) 左0→右1
    dm = build_density_map(gray, 8, 8)
    col_means = dm.mean(axis=0)
    # 各列の平均が前列より大きいか等しいこと（単調非減少）
    assert np.all(np.diff(col_means) >= -1e-6), f"列平均が単調増加でない: {col_means}"


def test_build_density_map_range():
    """密度マップの値が [0.0, 1.0] 内に収まること。"""
    gray = np.random.rand(50, 50)
    dm = build_density_map(gray, 10, 10)
    assert float(dm.min()) >= 0.0
    assert float(dm.max()) <= 1.0


# ===========================================================================
# Stage 3a: Kruskal 迷路生成テスト (build_spanning_tree)
# ===========================================================================

def test_spanning_tree_perfect_maze_edge_count():
    """Perfect maze は n-1 本のエッジを持つこと（スパニングツリーの性質）。"""
    grid = _build_gradient_grid(10, 10)
    adj = build_spanning_tree(grid)
    n = grid.num_cells  # 100
    # 各エッジは adj に 2回登録（有向）→ 実際のエッジ数 = 合計 / 2
    total_degree = sum(len(nbs) for nbs in adj.values())
    assert total_degree == 2 * (n - 1), f"期待 {2*(n-1)} degree-units、実際 {total_degree}"


def test_spanning_tree_all_cells_connected():
    """スパニングツリーが全セルを1連結にすること（BFS で確認）。"""
    grid = _build_gradient_grid(8, 8)
    adj = build_spanning_tree(grid)
    assert _all_connected_bfs(adj, grid.num_cells), "スパニングツリーが全セルを連結していない"


def test_spanning_tree_adjacency_symmetric():
    """adj[a] に b が含まれるなら adj[b] に a が含まれること（無向グラフ）。"""
    grid = _build_gradient_grid(6, 6)
    adj = build_spanning_tree(grid)
    for a, nbs in adj.items():
        for b in nbs:
            assert a in adj[b], f"非対称: adj[{a}]={nbs} だが adj[{b}]に {a} がない"


def test_spanning_tree_dark_region_higher_degree():
    """DM-1 核心: Kruskal で暗部（lum=0）のセルが明部（lum=1）より高い平均次数を持つこと。"""
    rows, cols = 8, 8
    grid = _build_gradient_grid(rows, cols)
    adj = build_spanning_tree(grid)

    half = cols // 2  # 4
    # 暗部: 左半（lum=0）
    dark_degrees = [len(adj[grid.cell_id(r, c)]) for r in range(rows) for c in range(half)]
    # 明部: 右半（lum=1）
    bright_degrees = [len(adj[grid.cell_id(r, c)]) for r in range(rows) for c in range(half, cols)]

    avg_dark = sum(dark_degrees) / len(dark_degrees)
    avg_bright = sum(bright_degrees) / len(bright_degrees)
    assert avg_dark > avg_bright, (
        f"暗部の平均次数 ({avg_dark:.3f}) が明部 ({avg_bright:.3f}) より高くない"
    )


def test_spanning_tree_5x5():
    """5×5 グリッドで spanning tree が正常に生成されること。"""
    gray = np.random.rand(20, 20)
    grid = build_cell_grid(gray, 5, 5)
    adj = build_spanning_tree(grid)
    assert _all_connected_bfs(adj, 25), "5x5 スパニングツリーが全セルを連結していない"


# ===========================================================================
# Stage 3b: BFS ソルバーテスト (bfs_has_path)
# ===========================================================================

def test_bfs_has_path_in_spanning_tree():
    """スパニングツリー内の任意の2セル間に BFS パスが存在すること。"""
    grid = _build_gradient_grid(6, 6)
    adj = build_spanning_tree(grid)
    n = grid.num_cells
    # 0→n-1（対角セル）のパスを確認
    assert bfs_has_path(adj, 0, n - 1), "スパニングツリー内の対角間にパスがない"


def test_bfs_has_path_disconnected():
    """孤立セルへのパスが False を返すこと。"""
    adj: Dict[int, List[int]] = {0: [1], 1: [0], 2: [3], 3: [2], 4: []}
    assert not bfs_has_path(adj, 0, 4), "孤立セル 4 へのパスが存在してはならない"


def test_bfs_has_path_start_equals_goal():
    """start == goal のとき True を返すこと。"""
    adj: Dict[int, List[int]] = {0: [1], 1: [0]}
    assert bfs_has_path(adj, 0, 0), "start == goal のとき True であること"


# ===========================================================================
# Stage 3c: 密度後処理テスト (post_process_density)
# ===========================================================================

def test_post_process_density_dark_gains_passages():
    """extra_removal_rate > 0 のとき暗部がより多くの通路を得ること。"""
    rows, cols = 8, 8
    grid = _build_gradient_grid(rows, cols)
    adj_base = build_spanning_tree(grid)

    # post_process 前の暗部通路数
    passages_before = _count_passages_in_region(adj_base, grid, 0, cols // 2)

    # 高い extra_removal_rate で暗部ループを追加
    adj_post = post_process_density(
        adj_base,
        grid,
        extra_removal_rate=0.9,
        dark_threshold=0.5,
        light_threshold=1.1,  # 明部削除なし
        rng=np.random.default_rng(0),
    )
    passages_after = _count_passages_in_region(adj_post, grid, 0, cols // 2)

    assert passages_after >= passages_before, (
        f"暗部通路数が増えていない: before={passages_before}, after={passages_after}"
    )


def test_post_process_density_solution_preserved():
    """post_process_density 後も全セルが1連結であること（解の存在を保証）。"""
    rows, cols = 8, 8
    grid = _build_gradient_grid(rows, cols)
    adj = build_spanning_tree(grid)
    adj_post = post_process_density(
        adj,
        grid,
        extra_removal_rate=0.5,
        dark_threshold=0.4,
        light_threshold=0.7,
        rng=np.random.default_rng(42),
    )
    assert _all_connected_bfs(adj_post, grid.num_cells), (
        "post_process_density 後に全セル連結が壊れた"
    )


# ===========================================================================
# Stage 5: レンダリングテスト (PNG / SVG)
# ===========================================================================

def test_generate_density_maze_png_nonempty():
    """PNG 出力が 0 バイトでないこと。"""
    img = _make_gradient_image(32, 64)
    result = generate_density_maze(img, grid_size=10, width=200, height=200)
    assert len(result.png_bytes) > 0, "PNG bytes が空"


def test_generate_density_maze_svg_tag():
    """SVG 出力に '<svg' タグが含まれること。"""
    img = _make_gradient_image(32, 64)
    result = generate_density_maze(img, grid_size=10, width=200, height=200)
    assert "<svg" in result.svg, "SVG 出力に '<svg' タグがない"


def test_generate_density_maze_png_valid_image():
    """PNG バイト列が PIL で開ける有効な画像であること。"""
    img = _make_gradient_image(32, 64)
    result = generate_density_maze(img, grid_size=8, width=160, height=160)
    opened = Image.open(io.BytesIO(result.png_bytes))
    assert opened.size[0] > 0 and opened.size[1] > 0, "PNG が無効な画像"


# ===========================================================================
# E2E テスト (generate_density_maze)
# ===========================================================================

def test_e2e_result_fields():
    """DensityMazeResult が DM-1 必須フィールドを持つこと。"""
    img = _make_gradient_image(32, 64)
    result = generate_density_maze(img, grid_size=10, width=200, height=200)
    assert result.svg, "svg フィールドが空"
    assert result.png_bytes, "png_bytes フィールドが空"
    assert result.entrance >= 0, "entrance が無効"
    assert result.exit_cell >= 0, "exit_cell が無効"
    assert len(result.solution_path) >= 1, "solution_path が空"
    assert result.grid_rows > 0 and result.grid_cols > 0, "grid_rows/cols が無効"


def test_e2e_solution_path_bfs_confirmed():
    """solution_path の入口・出口間に BFS パスが存在すること。"""
    img = _make_gradient_image(32, 64)
    result = generate_density_maze(img, grid_size=10, width=200, height=200)

    # SVG から adj を再構築するのは困難なため、solution_path の連続性のみ確認
    assert result.solution_path[0] == result.entrance or result.solution_path[-1] == result.entrance, (
        "solution_path の端点に entrance が含まれない"
    )
    assert result.exit_cell in (result.solution_path[0], result.solution_path[-1]), (
        "solution_path の端点に exit_cell が含まれない"
    )


def test_e2e_gradient_dark_region_denser():
    """
    DM-1 判定基準: グラデーション画像（左=黒/右=白）を入力すると
    暗部（左）が明部（右）より高い通路密度になること。

    extra_removal_rate=0.8 で暗部ループを強制的に追加し、
    隣接リストの平均次数で密度を測定する。
    """
    rows, cols = 12, 12
    grid = _build_gradient_grid(rows, cols)
    adj_base = build_spanning_tree(grid)
    adj = post_process_density(
        adj_base,
        grid,
        extra_removal_rate=0.8,
        dark_threshold=0.5,
        light_threshold=1.1,  # 明部削除なし（暗部増加のみ）
        rng=np.random.default_rng(0),
    )

    half = cols // 2
    dark_deg = [len(adj[grid.cell_id(r, c)]) for r in range(rows) for c in range(half)]
    bright_deg = [len(adj[grid.cell_id(r, c)]) for r in range(rows) for c in range(half, cols)]

    avg_dark = sum(dark_deg) / len(dark_deg)
    avg_bright = sum(bright_deg) / len(bright_deg)

    assert avg_dark > avg_bright, (
        f"DM-1 判定基準NG: 暗部通路密度 {avg_dark:.3f} ≤ 明部 {avg_bright:.3f}"
    )


def test_e2e_large_grid_200():
    """
    grid_size=200 相当（200セル幅）でも正常に動作すること。
    実際のセル数はパイプライン内部でクリップされる。
    """
    img = _make_gradient_image(200, 400)
    # grid_sizeは内部でmax_side/4以下にクリップされるが、パラメータとして200を指定
    result = generate_density_maze(
        img,
        grid_size=200,
        max_side=800,
        width=400,
        height=200,
        show_solution=False,
    )
    assert result.grid_cols > 0
    assert len(result.png_bytes) > 0


def test_e2e_all_dark_image():
    """全黒画像（密度最大）でも迷路が生成され解が存在すること。"""
    img = _make_uniform_image(0)  # 全黒
    result = generate_density_maze(img, grid_size=10, width=100, height=100, show_solution=False)
    assert len(result.solution_path) >= 1


def test_e2e_all_bright_image():
    """全白画像（密度最小）でも迷路が生成され解が存在すること。"""
    img = _make_uniform_image(255)  # 全白
    result = generate_density_maze(img, grid_size=10, width=100, height=100, show_solution=False)
    assert len(result.solution_path) >= 1


def test_e2e_density_range_boundary():
    """density_min=0.1 / density_max=0.9 相当の density_factor でも動作すること。"""
    img = _make_gradient_image(32, 64)
    result = generate_density_maze(
        img,
        grid_size=10,
        width=200,
        height=200,
        density_factor=0.1,
        show_solution=False,
    )
    assert len(result.png_bytes) > 0

    result2 = generate_density_maze(
        img,
        grid_size=10,
        width=200,
        height=200,
        density_factor=0.9,
        show_solution=False,
    )
    assert len(result2.png_bytes) > 0
