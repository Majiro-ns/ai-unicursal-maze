"""
密度迷路 Phase 1: Kruskal + Union-Find で perfect maze（spanning tree）を生成。
Phase 2b: post_process_density() — ループ許容の密度後処理（暗部追加壁除去・明部通路削除）。
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set

import numpy as np

from .grid_builder import CellGrid


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


def _is_connected(adj_sets: Dict[int, Set[int]], n: int) -> bool:
    """BFS で全 n 個のセルが連結か確認する。O(N)。"""
    if n == 0:
        return True
    visited: Set[int] = {0}
    queue: deque[int] = deque([0])
    while queue:
        cell = queue.popleft()
        for nb in adj_sets[cell]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return len(visited) == n


def post_process_density(
    adj: Dict[int, List[int]],
    grid: CellGrid,
    *,
    extra_removal_rate: float = 0.5,
    dark_threshold: float = 0.3,
    light_threshold: float = 0.7,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, List[int]]:
    """
    Perfect maze にループ許容の密度後処理を適用する。

    Step 1 — 暗部（luminance < dark_threshold）:
        隣接する壁を確率的に除去してループを作る（imperfect maze）。
        除去確率 = (1.0 - avg_luminance(セル対)) * extra_removal_rate。
        extra_removal_rate=0 のときは何もしない（既存動作と同一）。

    Step 2 — 明部（luminance > light_threshold）:
        両端が明るい通路を確率的に削除して壁密度を上げる。
        削除後に BFS で連結チェックを行い、切断するなら元に戻す。

    Args:
        adj: build_spanning_tree() の戻り値（隣接リスト）
        grid: CellGrid（rows / cols / luminance を参照）
        extra_removal_rate: 暗部追加除去率の上限 [0, 1]
        dark_threshold: これ未満の輝度を「暗部」と判定
        light_threshold: これ超の輝度を「明部」と判定
        rng: 乱数生成器（None の場合 seed=42 固定）

    Returns:
        修正済み adj（Dict[int, List[int]]）
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = grid.num_cells
    lum = grid.luminance  # (rows, cols) float 0-1

    # Set ベースのコピー（O(1) ルックアップ）
    adj_sets: Dict[int, Set[int]] = {i: set(nb) for i, nb in adj.items()}

    # spanning tree エッジを記録（全て bridge → Step 2 でスキップ可能）
    spanning_edges: Set[tuple] = set()
    for u, nbs in adj.items():
        for v in nbs:
            spanning_edges.add((min(u, v), max(u, v)))

    # ----- Step 1: 暗部 — 追加壁除去でループ作成 -----
    if extra_removal_rate > 0.0:
        for r in range(grid.rows):
            for c in range(grid.cols):
                cid = grid.cell_id(r, c)
                lum_a = float(lum[r, c])
                if lum_a >= dark_threshold:
                    continue
                # 右・下の隣接セルとの壁を対象（重複処理を避けるため）
                for dr, dc in ((0, 1), (1, 0)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < grid.rows and 0 <= nc < grid.cols):
                        continue
                    cid2 = grid.cell_id(nr, nc)
                    if cid2 in adj_sets[cid]:
                        continue  # 既に通路 — スキップ
                    lum_b = float(lum[nr, nc])
                    avg_lum = (lum_a + lum_b) / 2.0
                    prob = (1.0 - avg_lum) * extra_removal_rate
                    if rng.random() < prob:
                        adj_sets[cid].add(cid2)
                        adj_sets[cid2].add(cid)

    # ----- Step 2: 明部 — 通路削除（連結性保護） -----
    if light_threshold < 1.0:
        # 両端が明るい通路を列挙（cid < cid2 で重複排除）
        bright_passages: List[tuple[int, int]] = []
        for r in range(grid.rows):
            for c in range(grid.cols):
                if float(lum[r, c]) <= light_threshold:
                    continue
                cid = grid.cell_id(r, c)
                for cid2 in list(adj_sets[cid]):
                    if cid2 <= cid:
                        continue
                    nr2, nc2 = grid.cell_rc(cid2)
                    if float(lum[nr2, nc2]) > light_threshold:
                        bright_passages.append((cid, cid2))

        # ランダム順で削除を試みる
        order = np.arange(len(bright_passages))
        rng.shuffle(order)
        for i in order:
            cid, cid2 = bright_passages[i]
            if cid2 not in adj_sets[cid]:
                continue  # 既に削除済み
            # spanning tree エッジは必ず bridge → 削除不可、スキップ（O(N) BFS 不要）
            if (min(cid, cid2), max(cid, cid2)) in spanning_edges:
                continue
            # Step 1 で追加されたエッジのみ連結チェックを行う
            adj_sets[cid].discard(cid2)
            adj_sets[cid2].discard(cid)
            if not _is_connected(adj_sets, n):
                adj_sets[cid].add(cid2)
                adj_sets[cid2].add(cid)

    return {i: list(s) for i, s in adj_sets.items()}


def build_spanning_tree(grid: CellGrid) -> Dict[int, List[int]]:
    """
    Kruskal: 壁を weight 昇順で処理し、Union-Find でサイクルを避けつつ壁を除去。
    戻り値: 各セル id に対する「除去された壁でつながる隣接セル」のリスト（隣接リスト）。
    perfect maze = 全セルが1連結の木。
    """
    n = grid.num_cells
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    uf = UnionFind(n)
    # 壁を weight 昇順でソート（小さい＝暗い＝先に除去＝道を開ける）
    sorted_walls = sorted(grid.walls, key=lambda x: x[2])
    for c1, c2, _ in sorted_walls:
        if uf.union(c1, c2):
            adj[c1].append(c2)
            adj[c2].append(c1)
    return adj
