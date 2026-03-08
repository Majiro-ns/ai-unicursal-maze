"""
密度迷路 Phase 2: 解の一意性制御。

BFS で連結確認、DFS で解数カウント、エッジ除去で一意解に絞る。
隣接リスト形式（Dict[int, List[int]]）で動作する。
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Set, Tuple


def bfs_has_path(adj: Dict[int, List[int]], start: int, goal: int) -> bool:
    """BFS で start → goal への経路が存在するかを確認する。"""
    if start == goal:
        return True
    visited: Set[int] = {start}
    queue: deque = deque([start])
    while queue:
        node = queue.popleft()
        for nb in adj.get(node, []):
            if nb == goal:
                return True
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return False


def count_solutions_dfs(
    adj: Dict[int, List[int]],
    start: int,
    goal: int,
    *,
    max_solutions: int = 2,
    max_visits: int = 100_000,
) -> int:
    """
    DFS で start → goal の単純路（単純パス）を列挙し、max_solutions まで数える。

    Returns:
        0: 解なし
        1: 唯一解
        max_solutions: それ以上の解が存在（頭打ち）
    """
    if start == goal:
        return 1
    if start not in adj and not adj.get(start):
        return 0

    num_solutions = 0
    visits = 0
    visited: Set[int] = set()

    def _dfs(node: int) -> None:
        nonlocal num_solutions, visits
        if num_solutions >= max_solutions or visits >= max_visits:
            return
        visits += 1
        if node == goal:
            num_solutions += 1
            return
        visited.add(node)
        for nb in adj.get(node, []):
            if nb not in visited:
                _dfs(nb)
        visited.remove(node)

    _dfs(start)
    return num_solutions


def is_unique_solution(
    adj: Dict[int, List[int]],
    start: int,
    goal: int,
    *,
    max_visits: int = 100_000,
) -> bool:
    """start → goal の解がちょうど 1 本かどうかを返す。"""
    n = count_solutions_dfs(adj, start, goal, max_solutions=2, max_visits=max_visits)
    return n == 1


def _adj_to_edge_set(adj: Dict[int, List[int]]) -> Set[Tuple[int, int]]:
    """隣接リストから無向エッジセット {(min(a,b), max(a,b))} に変換する。"""
    edges: Set[Tuple[int, int]] = set()
    for a, neighbors in adj.items():
        for b in neighbors:
            edges.add((min(a, b), max(a, b)))
    return edges


def _edge_set_to_adj(edges: Set[Tuple[int, int]], n: int) -> Dict[int, List[int]]:
    """エッジセットから n ノードの隣接リストに変換する。"""
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def enforce_unique_solution(
    adj: Dict[int, List[int]],
    start: int,
    goal: int,
    num_cells: int,
    *,
    max_removals: int = 200,
    max_visits: int = 50_000,
) -> Tuple[Dict[int, List[int]], bool]:
    """
    解の一意性制御: 複数解がある場合、壁（エッジ除去）を追加して一意解に絞る。
    解の経路は保護される（入口 → 出口の連結を維持）。

    アルゴリズム:
      1. 現在の解数を DFS でカウント
      2. 解数 > 1 なら: エッジを 1 本ずつ試し、削除後に解数が減かつ ≥1 なら削除確定
      3. 解数が 1 になるか、削除できるエッジがなくなるまで繰り返す

    Args:
        adj:          隣接リスト（Dict[int, List[int]]）
        start:        入口セル id
        goal:         出口セル id
        num_cells:    総セル数（ノード数）
        max_removals: 最大壁追加回数（無限ループ防止）
        max_visits:   DFS 訪問上限

    Returns:
        (result_adj, is_unique):
          result_adj: 変換後の隣接リスト（一意化済みまたは最大限削減済み）
          is_unique:  変換後に唯一解（解数 == 1）かどうか
    """
    _MAX_SOLS = 10  # 内部カウント上限（3本以上の経路も検出）
    active_edges = _adj_to_edge_set(adj)

    for _ in range(max_removals):
        current_adj = _edge_set_to_adj(active_edges, num_cells)
        n = count_solutions_dfs(
            current_adj, start, goal,
            max_solutions=_MAX_SOLS, max_visits=max_visits,
        )
        if n <= 1:
            break

        removed = False
        for edge in sorted(active_edges):  # ソートで決定的に動作
            trial_edges = active_edges - {edge}
            trial_adj = _edge_set_to_adj(trial_edges, num_cells)
            n_trial = count_solutions_dfs(
                trial_adj, start, goal,
                max_solutions=_MAX_SOLS, max_visits=max_visits,
            )
            if 1 <= n_trial < n:  # 解数が減り、かつ経路が残る
                active_edges = trial_edges
                removed = True
                break

        if not removed:
            break

    result_adj = _edge_set_to_adj(active_edges, num_cells)
    n_final = count_solutions_dfs(result_adj, start, goal, max_solutions=2, max_visits=max_visits)
    return result_adj, (n_final == 1)
