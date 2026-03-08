"""
密度迷路 Phase 1/2: 解経路の端点を入口・出口に（後から決める）。
Phase 1: 木の直径の両端を BFS で求める。
Phase 2: 複数候補から「美しい」解経路を選択するヒューリスティクスを追加。
         スコア = 経路長 × 平均輝度（明るい領域を通る長い経路が高スコア）。
Phase 2b: Dijkstra による画像適応ルーティング。
          エッジコスト = 1 - avg_luminance → 明部を優先的に通る。
"""
from __future__ import annotations

import heapq
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


def _bfs_farthest(adj: Dict[int, List[int]], start: int, n: int) -> Tuple[int, List[int]]:
    """start から最も遠いノードと、start からの距離リストを返す。"""
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    farthest = start
    for i in range(n):
        if dist[i] >= 0 and dist[i] > dist[farthest]:
            farthest = i
    return farthest, dist


def _path_from_bfs(adj: Dict[int, List[int]], dist: List[int], start: int, end: int) -> List[int]:
    """dist が start からの距離のとき、end から start へ戻るパスを復元。"""
    path = [end]
    while path[-1] != start:
        u = path[-1]
        for v in adj.get(u, []):
            if dist[v] == dist[u] - 1:
                path.append(v)
                break
        else:
            break
    return path[::-1]


def find_entrance_exit_and_path(
    adj: Dict[int, List[int]],
    num_cells: int,
) -> Tuple[int, int, List[int]]:
    """
    木の直径の両端を入口・出口とし、その間の唯一の経路を解経路として返す。
    戻り値: (entrance_cell_id, exit_cell_id, solution_path_cell_ids)
    """
    if num_cells == 0:
        return -1, -1, []
    if num_cells == 1:
        return 0, 0, [0]
    # 0 から最も遠い点 A
    a, dist_a = _bfs_farthest(adj, 0, num_cells)
    # A から最も遠い点 B = 直径のもう一端
    b, dist_b = _bfs_farthest(adj, a, num_cells)
    path = _path_from_bfs(adj, dist_b, a, b)
    return a, b, path


def find_entrance_exit_heuristic(
    adj: Dict[int, List[int]],
    num_cells: int,
    luminance: np.ndarray,
    max_candidates: int = 10,
) -> Tuple[int, int, List[int]]:
    """
    Phase 2: 美しい解経路を選択するヒューリスティクス。

    アルゴリズム:
    1. 葉ノード（隣接1本のみ）を列挙し、輝度の高い順にソート
    2. 上位 max_candidates 個を起点候補として各々 BFS で最遠端を求める
    3. スコア = 経路長 × 経路上の平均輝度 で最良候補を選択
       （明るい領域を通る長い経路 = 視認しやすく美しい解経路）

    戻り値: (entrance_cell_id, exit_cell_id, solution_path_cell_ids)
    """
    if num_cells == 0:
        return -1, -1, []
    if num_cells == 1:
        return 0, 0, [0]

    flat_lum = luminance.flatten()

    # 葉ノード（degree=1）を輝度の降順で取得（明るい端が入口・出口として好ましい）
    leaves = [i for i in range(num_cells) if len(adj.get(i, [])) == 1]
    if not leaves:
        # 全ノードが葉でない場合（1x1グリッド等）はフォールバック
        return find_entrance_exit_and_path(adj, num_cells)

    leaves_sorted = sorted(leaves, key=lambda l: -float(flat_lum[l]))
    candidates = leaves_sorted[:max_candidates]

    best_score = -1.0
    best_entrance, best_exit, best_path = candidates[0], candidates[0], [candidates[0]]

    for start in candidates:
        far, dist = _bfs_farthest(adj, start, num_cells)
        path = _path_from_bfs(adj, dist, start, far)
        if len(path) == 0:
            continue
        path_lum = float(np.mean([flat_lum[c] for c in path]))
        score = float(len(path)) * path_lum
        if score > best_score:
            best_score = score
            best_entrance, best_exit, best_path = start, far, path

    return best_entrance, best_exit, best_path


def find_image_guided_path(
    adj: Dict[int, List[int]],
    num_cells: int,
    luminance: np.ndarray,
    grid_rows: int,
    grid_cols: int,
) -> Tuple[int, int, List[int]]:
    """
    Phase 2b: Dijkstra による画像適応ルーティング。

    解法経路が画像の明るい部分を優先的に通るように Dijkstra でルーティングする。
    masterpieceの「白い道を塗りつぶした完成形」を実現するための柱3実装。

    アルゴリズム:
      1. 各エッジのコスト = 1.0 - avg(luminance[u], luminance[v])
         → 明るいセル間(luminance≈1.0) → コスト≈0.0 → 優先的に通る
         → 暗いセル間(luminance≈0.0) → コスト≈1.0 → 迂回する
      2. 入口: 4隅の中で最も暗い角 (masterpiece の塗り始め = 暗い角から出発)
         出口: 入口の対角線上の角
      3. Dijkstra で入口→出口の最小コスト経路を求める

    Args:
        adj: 隣接リスト (spanning tree)
        num_cells: グリッドのセル総数
        luminance: CellGrid.luminance (rows×cols の float 0-1 配列)
        grid_rows: グリッドの行数
        grid_cols: グリッドの列数

    Returns:
        (entrance_cell_id, exit_cell_id, solution_path)
    """
    if num_cells <= 1:
        return 0, 0, [0]

    flat_lum = luminance.flatten()

    # --- 入口/出口の選択: 4隅から輝度最低を入口, 対角を出口 ---
    corners = {
        "tl": 0,
        "tr": grid_cols - 1,
        "bl": grid_cols * (grid_rows - 1),
        "br": num_cells - 1,
    }
    diag_pairs = [("tl", "br"), ("tr", "bl")]
    opposite = {"tl": "br", "br": "tl", "tr": "bl", "bl": "tr"}

    # 4隅の中で輝度最低の角を入口
    entrance_key = min(corners.keys(), key=lambda k: float(flat_lum[corners[k]]))
    exit_key = opposite[entrance_key]
    entrance = corners[entrance_key]
    exit_cell = corners[exit_key]

    # --- Dijkstra ---
    # コスト = 1.0 - avg(lum_u, lum_v) ∈ [0, 1]
    INF = float("inf")
    dist: List[float] = [INF] * num_cells
    prev: List[int] = [-1] * num_cells
    dist[entrance] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, entrance)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v in adj.get(u, []):
            edge_cost = 1.0 - (float(flat_lum[u]) + float(flat_lum[v])) / 2.0
            edge_cost = max(0.0, edge_cost)  # 数値誤差ガード
            nd = d + edge_cost
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    # --- 経路復元 ---
    if dist[exit_cell] == INF:
        # exit_cell に到達できない場合（グリッドが小さい等）は BFS フォールバック
        return find_entrance_exit_and_path(adj, num_cells)

    path: List[int] = []
    cur = exit_cell
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    if not path or path[0] != entrance:
        return find_entrance_exit_and_path(adj, num_cells)

    return entrance, exit_cell, path
