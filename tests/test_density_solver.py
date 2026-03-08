# -*- coding: utf-8 -*-
"""
密度迷路 Phase 2: 解の一意性制御テスト。

対象: backend/core/density/solver.py
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List

import numpy as np
import pytest
from PIL import Image

from backend.core.density.solver import (
    bfs_has_path,
    count_solutions_dfs,
    enforce_unique_solution,
    is_unique_solution,
)


# ============================================================
# テスト用グラフファクトリ
# ============================================================

def _linear(n: int) -> Dict[int, List[int]]:
    """0-1-2-..-(n-1) の直線グラフ（唯一解: start=0, goal=n-1）。"""
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n - 1):
        adj[i].append(i + 1)
        adj[i + 1].append(i)
    return adj


def _triangle() -> Dict[int, List[int]]:
    """
    0-1-2-0 の三角形グラフ（0→2 への解が 2 本: 0→1→2 と 0→2）。
    """
    return {
        0: [1, 2],
        1: [0, 2],
        2: [1, 0],
    }


def _diamond() -> Dict[int, List[int]]:
    """
    ダイアモンド: 0→1→3 と 0→2→3 の 2 経路。
        0
       / \
      1   2
       \ /
        3
    """
    return {
        0: [1, 2],
        1: [0, 3],
        2: [0, 3],
        3: [1, 2],
    }


def _spanning_tree_4x4() -> Dict[int, List[int]]:
    """4×4 グリッドの spanning tree（DFS 生成、唯一解）。"""
    # 単純な DFS spanning tree を手動で構築
    adj: Dict[int, List[int]] = {i: [] for i in range(16)}
    # 右方向の辺: (0,1),(1,2),(2,3),(4,5),(5,6),(6,7),(8,9),(9,10),(10,11),(12,13),(13,14),(14,15)
    right = [(r * 4 + c, r * 4 + c + 1) for r in range(4) for c in range(3)]
    # 下方向の辺の一部（木を構成する最小限）
    down = [(r * 4 + c, (r + 1) * 4 + c) for r in range(3) for c in range(4)]
    # spanning tree: 全右辺 + 下辺の一部（サイクルを避けるため列0のみ）
    tree_edges = right + [(0, 4), (4, 8), (8, 12)]
    seen = set()
    for a, b in tree_edges:
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            adj[a].append(b)
            adj[b].append(a)
    return adj


# ============================================================
# bfs_has_path
# ============================================================

def test_bfs_has_path_linear_start_to_end():
    """直線グラフで start→end への経路が存在する。"""
    adj = _linear(5)
    assert bfs_has_path(adj, 0, 4) is True


def test_bfs_has_path_linear_reverse():
    """直線グラフで end→start への経路も存在する（無向）。"""
    adj = _linear(5)
    assert bfs_has_path(adj, 4, 0) is True


def test_bfs_has_path_self_loop():
    """start == goal のとき True を返す。"""
    adj = _linear(3)
    assert bfs_has_path(adj, 1, 1) is True


def test_bfs_has_path_disconnected():
    """分断グラフでは到達できないノードへ False を返す。"""
    adj: Dict[int, List[int]] = {0: [1], 1: [0], 2: [3], 3: [2]}
    assert bfs_has_path(adj, 0, 3) is False


def test_bfs_has_path_triangle():
    """三角形グラフで任意 2 点間に経路が存在する。"""
    adj = _triangle()
    assert bfs_has_path(adj, 0, 2) is True
    assert bfs_has_path(adj, 2, 0) is True


def test_bfs_has_path_single_node():
    """ノードが 1 つ（start == goal）で True を返す。"""
    adj: Dict[int, List[int]] = {0: []}
    assert bfs_has_path(adj, 0, 0) is True


# ============================================================
# count_solutions_dfs
# ============================================================

def test_count_solutions_linear_is_one():
    """直線グラフ（木）は唯一解。"""
    adj = _linear(5)
    assert count_solutions_dfs(adj, 0, 4) == 1


def test_count_solutions_triangle_has_two():
    """三角形グラフ（0→2 への経路が 2 本: 0→1→2 と 0→2）。"""
    adj = _triangle()
    n = count_solutions_dfs(adj, 0, 2, max_solutions=3)
    assert n == 2


def test_count_solutions_diamond_has_two():
    """ダイアモンドグラフは 0→3 への解が 2 本。"""
    adj = _diamond()
    n = count_solutions_dfs(adj, 0, 3, max_solutions=3)
    assert n == 2


def test_count_solutions_disconnected_is_zero():
    """分断グラフでは解なし（0 を返す）。"""
    adj: Dict[int, List[int]] = {0: [1], 1: [0], 2: [3], 3: [2]}
    assert count_solutions_dfs(adj, 0, 3) == 0


def test_count_solutions_start_equals_goal():
    """start == goal のとき解は 1。"""
    adj = _linear(3)
    assert count_solutions_dfs(adj, 2, 2) == 1


def test_count_solutions_max_solutions_cap():
    """max_solutions 上限で頭打ちになる。"""
    adj = _triangle()
    # 三角形は 2 解だが max_solutions=1 で頭打ち
    n = count_solutions_dfs(adj, 0, 2, max_solutions=1)
    assert n == 1


def test_count_solutions_spanning_tree_is_one():
    """spanning tree（4×4）は任意の 2 点間で唯一解。"""
    adj = _spanning_tree_4x4()
    assert count_solutions_dfs(adj, 0, 15) == 1


def test_count_solutions_2cell_graph_is_one():
    """2 セルの最小グラフは唯一解。"""
    adj: Dict[int, List[int]] = {0: [1], 1: [0]}
    assert count_solutions_dfs(adj, 0, 1) == 1


def test_count_solutions_max_visits_limits_search():
    """max_visits=1 では解を全数列挙できない（0 または未満の結果）。"""
    adj = _diamond()
    # max_visits=1 で探索を制限
    n = count_solutions_dfs(adj, 0, 3, max_solutions=3, max_visits=1)
    # 1 回の訪問では完全探索できないので 0 か 1 になる
    assert n <= 1


# ============================================================
# is_unique_solution
# ============================================================

def test_is_unique_linear_true():
    """直線グラフは唯一解 → True。"""
    adj = _linear(6)
    assert is_unique_solution(adj, 0, 5) is True


def test_is_unique_triangle_false():
    """三角形グラフは複数解 → False。"""
    adj = _triangle()
    assert is_unique_solution(adj, 0, 2) is False


def test_is_unique_diamond_false():
    """ダイアモンドグラフは複数解 → False。"""
    adj = _diamond()
    assert is_unique_solution(adj, 0, 3) is False


def test_is_unique_self():
    """start == goal のとき唯一解 → True。"""
    adj = _linear(4)
    assert is_unique_solution(adj, 2, 2) is True


def test_is_unique_disconnected_false():
    """分断グラフは解なし → False（解数 0 は唯一解でない）。"""
    adj: Dict[int, List[int]] = {0: [1], 1: [0], 2: [3], 3: [2]}
    assert is_unique_solution(adj, 0, 3) is False


def test_is_unique_spanning_tree_true():
    """spanning tree は任意の 2 点間で唯一解 → True。"""
    adj = _spanning_tree_4x4()
    assert is_unique_solution(adj, 0, 15) is True


# ============================================================
# enforce_unique_solution
# ============================================================

def test_enforce_unique_triangle_becomes_unique():
    """三角形グラフを一意解に絞る（エッジ除去後に唯一解）。"""
    adj = _triangle()
    result_adj, is_unique = enforce_unique_solution(adj, 0, 2, num_cells=3)
    assert is_unique is True
    # 唯一解になったことをカウントでも確認
    n = count_solutions_dfs(result_adj, 0, 2)
    assert n == 1


def test_enforce_unique_diamond_becomes_unique():
    """ダイアモンドグラフを一意解に絞る。"""
    adj = _diamond()
    result_adj, is_unique = enforce_unique_solution(adj, 0, 3, num_cells=4)
    assert is_unique is True
    n = count_solutions_dfs(result_adj, 0, 3)
    assert n == 1


def test_enforce_unique_preserves_path():
    """一意化後も start → goal への経路が存在する（経路保護）。"""
    adj = _diamond()
    result_adj, _ = enforce_unique_solution(adj, 0, 3, num_cells=4)
    assert bfs_has_path(result_adj, 0, 3) is True


def test_enforce_unique_already_unique():
    """既に唯一解のグラフを変換しても唯一解のまま。"""
    adj = _linear(5)
    result_adj, is_unique = enforce_unique_solution(adj, 0, 4, num_cells=5)
    assert is_unique is True
    n = count_solutions_dfs(result_adj, 0, 4)
    assert n == 1


def test_enforce_unique_deterministic():
    """同じ入力に対して同じ結果を返す（決定的動作）。"""
    adj = _diamond()
    result1, unique1 = enforce_unique_solution(adj, 0, 3, num_cells=4)
    result2, unique2 = enforce_unique_solution(adj, 0, 3, num_cells=4)
    assert unique1 == unique2
    # 同じエッジセットになること
    from backend.core.density.solver import _adj_to_edge_set
    assert _adj_to_edge_set(result1) == _adj_to_edge_set(result2)


def test_enforce_unique_max_removals_limit():
    """max_removals=0 では何も変更しない。"""
    adj = _diamond()
    result_adj, _ = enforce_unique_solution(adj, 0, 3, num_cells=4, max_removals=0)
    # エッジが変わっていないこと（ダイアモンドのエッジ数は 4）
    from backend.core.density.solver import _adj_to_edge_set
    original_edges = _adj_to_edge_set(adj)
    result_edges = _adj_to_edge_set(result_adj)
    assert original_edges == result_edges


# ============================================================
# density 迷路との統合テスト
# ============================================================

def _make_small_image(w: int = 32, h: int = 32) -> Image.Image:
    arr = np.ones((h, w), dtype=np.uint8) * 128
    return Image.fromarray(arr, mode="L")


def test_density_spanning_tree_is_already_unique():
    """spanning tree 生成後は常に唯一解。enforce_unique_solution を適用しても唯一解のまま。"""
    from backend.core.density import generate_density_maze
    from backend.core.density.preprocess import preprocess_image
    from backend.core.density.grid_builder import build_cell_grid
    from backend.core.density.maze_builder import build_spanning_tree

    img = _make_small_image(32, 32)
    result = generate_density_maze(img, grid_size=5, max_side=32)
    gray = preprocess_image(img, max_side=32)
    grid = build_cell_grid(gray, result.grid_rows, result.grid_cols)
    adj = build_spanning_tree(grid)

    # spanning tree は唯一解
    assert is_unique_solution(adj, result.entrance, result.exit_cell)

    # enforce しても唯一解のまま
    result_adj, is_unique = enforce_unique_solution(
        adj, result.entrance, result.exit_cell, grid.num_cells
    )
    assert is_unique is True


def test_density_spanning_tree_solution_count_is_one():
    """spanning tree の解数が DFS カウントで 1 になる。"""
    from backend.core.density import generate_density_maze
    from backend.core.density.preprocess import preprocess_image
    from backend.core.density.grid_builder import build_cell_grid
    from backend.core.density.maze_builder import build_spanning_tree

    img = _make_small_image(32, 32)
    result = generate_density_maze(img, grid_size=4, max_side=32)
    gray = preprocess_image(img, max_side=32)
    grid = build_cell_grid(gray, result.grid_rows, result.grid_cols)
    adj = build_spanning_tree(grid)

    n = count_solutions_dfs(adj, result.entrance, result.exit_cell)
    assert n == 1, f"spanning tree の解数が {n}（期待値: 1）"


def test_bfs_confirms_spanning_tree_connectivity():
    """BFS でも spanning tree の入口→出口連結を確認できる。"""
    from backend.core.density import generate_density_maze
    from backend.core.density.preprocess import preprocess_image
    from backend.core.density.grid_builder import build_cell_grid
    from backend.core.density.maze_builder import build_spanning_tree

    img = _make_small_image(32, 32)
    result = generate_density_maze(img, grid_size=4, max_side=32)
    gray = preprocess_image(img, max_side=32)
    grid = build_cell_grid(gray, result.grid_rows, result.grid_cols)
    adj = build_spanning_tree(grid)

    assert bfs_has_path(adj, result.entrance, result.exit_cell) is True


def test_enforce_unique_on_graph_with_extra_edge():
    """spanning tree にエッジを 1 本追加するとサイクルが生じ、enforce で一意解に戻る。"""
    from backend.core.density import generate_density_maze
    from backend.core.density.preprocess import preprocess_image
    from backend.core.density.grid_builder import build_cell_grid
    from backend.core.density.maze_builder import build_spanning_tree

    img = _make_small_image(32, 32)
    result = generate_density_maze(img, grid_size=4, max_side=32)
    gray = preprocess_image(img, max_side=32)
    grid = build_cell_grid(gray, result.grid_rows, result.grid_cols)
    adj = build_spanning_tree(grid)

    n_cells = grid.num_cells
    entrance = result.entrance
    exit_cell = result.exit_cell

    # 既存グラフに存在しないエッジを 1 本追加してサイクルを作る
    from backend.core.density.solver import _adj_to_edge_set, _edge_set_to_adj
    existing = _adj_to_edge_set(adj)
    added = False
    for a in range(n_cells):
        for b in range(a + 1, n_cells):
            if (a, b) not in existing:
                # a と b が隣接セルかチェック（グリッド上の隣接）
                ar, ac = a // grid.cols, a % grid.cols
                br, bc = b // grid.cols, b % grid.cols
                if abs(ar - br) + abs(ac - bc) == 1:
                    existing.add((a, b))
                    added = True
                    break
        if added:
            break

    if not added:
        pytest.skip("隣接エッジを追加できなかった（グリッドが 1x1 等）")

    adj_with_extra = _edge_set_to_adj(existing, n_cells)

    # 追加後は複数解の可能性がある（解数 >= 1 であること）
    n_before = count_solutions_dfs(adj_with_extra, entrance, exit_cell, max_solutions=3)
    assert n_before >= 1

    # enforce で唯一解に絞る
    result_adj, is_unique = enforce_unique_solution(
        adj_with_extra, entrance, exit_cell, n_cells
    )
    assert is_unique is True
    assert bfs_has_path(result_adj, entrance, exit_cell) is True
