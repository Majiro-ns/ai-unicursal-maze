# -*- coding: utf-8 -*-
"""
MAZE-Q3: Dijkstra経路の実効化 テスト (cmd_361k_a4)

テスト内容:
  MQ3-1: ループあり（extra_removal_rate>0）でスパニング木より多くのエッジが存在
         → Dijkstraが別経路を選択できる条件の確認
  MQ3-2: 4辺入口拡張 — コーナー以外の辺中央セルが最暗のとき入口として選ばれること
  MQ3-3: 明部優先度の定量検証 — ループあり迷路でDijkstraがBFSより高い平均輝度の経路を返すこと
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List

import numpy as np
import pytest
from PIL import Image

from backend.core.density.entrance_exit import (
    find_entrance_exit_and_path,
    find_image_guided_path,
    _border_cells_by_side,
    _OPPOSITE_SIDE,
)
from backend.core.density.grid_builder import build_cell_grid
from backend.core.density.maze_builder import build_spanning_tree, post_process_density
from backend.core.density.preprocess import preprocess_image


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_gradient_image(width: int = 64, height: int = 64) -> Image.Image:
    """左黒(0) → 右白(255) のグラデーション画像。"""
    arr = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    return Image.fromarray(arr, mode="L")


def _make_dark_image(width: int = 64, height: int = 64) -> Image.Image:
    """全面黒（輝度0）画像 — extra_removal_rate>0 で多くのループが追加される。"""
    arr = np.zeros((height, width), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def _bfs_shortest_path(
    adj: Dict[int, List[int]], start: int, end: int, n: int
) -> List[int]:
    """BFS で start → end の最短経路を返す。到達不可の場合は空リスト。"""
    dist = [-1] * n
    prev = [-1] * n
    dist[start] = 0
    q: deque = deque([start])
    while q:
        u = q.popleft()
        if u == end:
            break
        for v in adj.get(u, []):
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                prev[v] = u
                q.append(v)
    if dist[end] == -1:
        return []
    path: List[int] = []
    cur = end
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


# ---------------------------------------------------------------------------
# MQ3-1: ループあり（extra_removal_rate>0）でスパニング木より多くのエッジが存在
# ---------------------------------------------------------------------------

class TestLoopGraphEnablesDifferentRouting:
    """ループあり迷路でDijkstraが別経路を選べる前提条件を確認する。"""

    def setup_method(self):
        img = _make_dark_image(64, 64)  # 全面暗い → 暗部ループが多く追加される
        gray = preprocess_image(img, max_side=64)
        self.grid = build_cell_grid(gray, 8, 8)
        self.adj_tree = build_spanning_tree(self.grid)
        rng = np.random.default_rng(42)
        self.adj_loop = post_process_density(
            self.adj_tree, self.grid, extra_removal_rate=0.8, rng=rng
        )

    def test_loop_graph_has_more_edges_than_spanning_tree(self):
        """extra_removal_rate=0.8 でループが追加され、スパニング木より多くのエッジを持つ。"""
        tree_edges = sum(len(v) for v in self.adj_tree.values()) // 2
        loop_edges = sum(len(v) for v in self.adj_loop.values()) // 2
        assert loop_edges > tree_edges, (
            f"ループが追加されていない: tree={tree_edges}, loop={loop_edges}"
        )

    def test_loop_graph_adj_is_symmetric(self):
        """ループ追加後のadjが双方向（u→v ならば v→u）であること。"""
        for u, neighbors in self.adj_loop.items():
            for v in neighbors:
                assert u in self.adj_loop.get(v, []), (
                    f"双方向性違反: {u}→{v} は存在するが {v}→{u} は存在しない"
                )

    def test_loop_graph_remains_connected(self):
        """ループ追加後も全セルが連結であること。"""
        n = self.grid.num_cells
        visited = {0}
        q: deque = deque([0])
        while q:
            u = q.popleft()
            for v in self.adj_loop.get(u, []):
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        assert len(visited) == n, (
            f"ループ追加後に連結性が失われた: visited={len(visited)}, total={n}"
        )

    def test_dijkstra_finds_valid_path_on_loop_graph(self):
        """ループあり迷路でDijkstraが有効な入口→出口経路を返すこと。"""
        entrance, exit_cell, path = find_image_guided_path(
            self.adj_loop, self.grid.num_cells,
            self.grid.luminance, self.grid.rows, self.grid.cols,
        )
        assert len(path) >= 1, "ループあり迷路でDijkstra経路が空"
        assert path[0] == entrance, "経路先頭が入口と一致しない"
        assert path[-1] == exit_cell, "経路末尾が出口と一致しない"


# ---------------------------------------------------------------------------
# MQ3-2: 4辺入口拡張 — コーナー以外のセルを入口として選べること
# ---------------------------------------------------------------------------

class TestEntranceFromBorderNonCornerCell:
    """4辺入口拡張: 辺中央のセルが最暗なとき入口として選ばれること。"""

    def test_top_edge_middle_cell_becomes_entrance(self):
        """上辺中央セルが最暗の場合、コーナーではなく辺中央が入口になる。

        設定:
          rows=4, cols=6  → コーナー: {0, 5, 18, 23}
          上辺 col=3 (cell_id=3) のみ luminance=0.0、他は 1.0
          → 新ロジック: 入口=cell 3 (コーナーではない)
          → 旧ロジック（4隅固定）: 入口=コーナーのいずれか（全て lum=1.0）
        """
        rows, cols = 4, 6
        n = rows * cols

        # luminance: 全面明るい(1.0) ただし上辺中央 col=3 だけ暗い(0.0)
        lum = np.ones((rows, cols), dtype=float)
        lum[0, 3] = 0.0  # cell_id = 0*cols + 3 = 3

        # 格子状adj（全エッジ接続）
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        for r in range(rows):
            for c in range(cols):
                cid = r * cols + c
                if c + 1 < cols:
                    adj[cid].append(cid + 1)
                    adj[cid + 1].append(cid)
                if r + 1 < rows:
                    adj[cid].append(cid + cols)
                    adj[cid + cols].append(cid)

        entrance, exit_cell, path = find_image_guided_path(adj, n, lum, rows, cols)

        corners = {0, cols - 1, cols * (rows - 1), n - 1}

        # 4辺拡張: 入口はコーナーではなく辺中央セル(3)
        assert entrance not in corners, (
            f"入口 {entrance} がコーナー {corners} になっている — 4辺拡張が効いていない"
        )
        assert entrance == 3, (
            f"入口 {entrance} が最暗セル 3 (lum=0.0) でない"
        )
        assert path[0] == entrance, "経路先頭が入口と一致しない"

    def test_exit_on_opposite_side_of_entrance(self):
        """入口が上辺にあるとき、出口が下辺のセルであること。"""
        rows, cols = 4, 6
        n = rows * cols

        lum = np.ones((rows, cols), dtype=float)
        lum[0, 3] = 0.0  # 上辺中央が最暗 → entrance_side = "top"

        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        for r in range(rows):
            for c in range(cols):
                cid = r * cols + c
                if c + 1 < cols:
                    adj[cid].append(cid + 1)
                    adj[cid + 1].append(cid)
                if r + 1 < rows:
                    adj[cid].append(cid + cols)
                    adj[cid + cols].append(cid)

        entrance, exit_cell, path = find_image_guided_path(adj, n, lum, rows, cols)

        # 入口が上辺 → 出口は下辺 (row = rows-1)
        exit_row = exit_cell // cols
        assert exit_row == rows - 1, (
            f"出口 {exit_cell} が下辺(row={rows-1})以外 (row={exit_row})"
        )

    def test_border_cells_by_side_no_duplicates(self):
        """_border_cells_by_side が4辺間で重複なくセルを返すこと。"""
        rows, cols = 6, 8
        sides = _border_cells_by_side(rows, cols)
        all_cells: List[int] = []
        for cells in sides.values():
            all_cells.extend(cells)
        assert len(all_cells) == len(set(all_cells)), (
            "4辺間でセルIDの重複がある"
        )

    def test_border_cells_by_side_covers_all_border(self):
        """_border_cells_by_side が全辺セルを網羅すること。"""
        rows, cols = 5, 7
        sides = _border_cells_by_side(rows, cols)
        n = rows * cols

        expected_border = set()
        for r in range(rows):
            for c in range(cols):
                if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                    expected_border.add(r * cols + c)

        actual_border = set()
        for cells in sides.values():
            actual_border.update(cells)

        assert actual_border == expected_border, (
            f"辺セルの不一致: 欠落={expected_border - actual_border}, "
            f"余剰={actual_border - expected_border}"
        )

    def test_opposite_side_mapping_is_consistent(self):
        """_OPPOSITE_SIDE が対称的であること（top↔bottom, left↔right）。"""
        for side, opp in _OPPOSITE_SIDE.items():
            assert _OPPOSITE_SIDE[opp] == side, (
                f"_OPPOSITE_SIDE が非対称: {side}→{opp}→{_OPPOSITE_SIDE[opp]}"
            )


# ---------------------------------------------------------------------------
# MQ3-3: 明部優先度の定量検証
# ---------------------------------------------------------------------------

class TestLuminancePriorityGuidedVsBfs:
    """ループあり迷路でDijkstraがBFSより高い平均輝度の経路を返すことを定量検証する。"""

    def setup_method(self):
        """左黒→右白グラデーション 8x8 グリッド + ループ (rate=0.8)。"""
        img = _make_gradient_image(64, 64)
        gray = preprocess_image(img, max_side=64)
        self.grid = build_cell_grid(gray, 8, 8)
        adj_tree = build_spanning_tree(self.grid)
        rng = np.random.default_rng(42)
        self.adj_loop = post_process_density(
            adj_tree, self.grid, extra_removal_rate=0.8, rng=rng
        )
        self.flat_lum = self.grid.luminance.flatten()

    def test_dijkstra_path_luminance_ge_bfs_on_same_endpoints(self):
        """同一入口・出口でDijkstra経路の平均輝度 >= BFS最短経路の平均輝度。

        Dijkstraはエッジコスト=1-輝度でルーティングするため、
        BFS（最短距離）より明部を通る経路を選ぶはず。
        """
        n = self.grid.num_cells

        # Dijkstraで入口・出口を取得
        entrance, exit_cell, dijkstra_path = find_image_guided_path(
            self.adj_loop, n, self.grid.luminance, self.grid.rows, self.grid.cols
        )

        # 同一入口・出口でBFS最短経路
        bfs_path = _bfs_shortest_path(self.adj_loop, entrance, exit_cell, n)
        assert bfs_path, "BFSで入口→出口に到達できない（テスト前提条件エラー）"

        dijkstra_lum = float(np.mean([self.flat_lum[c] for c in dijkstra_path]))
        bfs_lum = float(np.mean([self.flat_lum[c] for c in bfs_path]))

        assert dijkstra_lum >= bfs_lum, (
            f"Dijkstra平均輝度 {dijkstra_lum:.3f} < BFS平均輝度 {bfs_lum:.3f}\n"
            f"ループあり迷路でのDijkstra明部優先効果が確認できない"
        )

    def test_dijkstra_path_mean_luminance_positive(self):
        """Dijkstra解法経路の平均輝度が正値であること（グラデーション画像での健全性確認）。"""
        n = self.grid.num_cells
        _, _, path = find_image_guided_path(
            self.adj_loop, n, self.grid.luminance, self.grid.rows, self.grid.cols
        )
        mean_lum = float(np.mean([self.flat_lum[c] for c in path]))
        assert mean_lum > 0.0, (
            f"ループあり迷路のDijkstra経路平均輝度 {mean_lum:.3f} が 0 以下"
        )

    def test_loop_graph_dijkstra_path_reachable_by_bfs(self):
        """ループあり迷路のDijkstra経路上の各セルがBFSで前後セルと隣接していること。"""
        n = self.grid.num_cells
        entrance, exit_cell, path = find_image_guided_path(
            self.adj_loop, n, self.grid.luminance, self.grid.rows, self.grid.cols
        )
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            assert v in self.adj_loop.get(u, []), (
                f"Dijkstra経路に無効エッジ: {u} → {v} は adj に存在しない"
            )
