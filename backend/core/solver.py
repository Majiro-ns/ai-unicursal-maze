from __future__ import annotations

import logging
from dataclasses import dataclass
from math import atan2
from typing import Dict, List, Set, Tuple

from .graph_builder import MazeGraph
from .graph_utils import build_adjacency
from .path_finder import PathPoint

logger = logging.getLogger(__name__)


@dataclass
class SolveResult:
    """
    T-8/T-9: 解の有無・解数・難易度指標をまとめた結果。
    - num_solutions: start→goal の解の個数（0, 1, または 2 で頭打ち）
    - difficulty_score: 0.0〜1.0。曲がり角の密度（turn_count / max(1, path_length-2)）で正規化
    - turn_count: 曲がり角の数（パス内の方向転換の総数）
    - path_length: 経路長（パスのノード数）
    """
    has_solution: bool
    num_solutions: int | None
    difficulty_score: float | None
    turn_count: int | None = None    # T-9: 曲がり角の数
    path_length: int | None = None  # T-9: 経路長（ノード数）


def count_solutions_on_graph(
    graph: MazeGraph,
    start: int,
    goal: int,
    *,
    max_solutions: int = 2,
    max_visits: int = 100_000,
) -> int:
    """
    MazeGraph 上で start→goal の単純路を DFS で列挙し、
    最大 max_solutions 個まで数える簡易ソルバ。
    戻り値:
      - 0: 解なし
      - 1: 一意解
      - 2: 複数解（max_solutions=2 のため 2 で頭打ち）
    """
    if start not in graph.nodes or goal not in graph.nodes:
        return 0

    adj = build_adjacency(graph)
    visited: Set[int] = set()
    num_solutions = 0
    visits = 0

    def dfs(nid: int) -> None:
        nonlocal num_solutions, visits
        if num_solutions >= max_solutions or visits >= max_visits:
            return
        visits += 1
        if nid == goal:
            num_solutions += 1
            return
        visited.add(nid)
        for nb in adj.get(nid, []):
            if nb not in visited:
                dfs(nb)
        visited.remove(nid)

    dfs(start)
    return num_solutions


def has_unique_solution(
    graph: MazeGraph,
    start: int,
    goal: int,
    *,
    max_visits: int = 100_000,
) -> bool:
    """
    start→goal の解がちょうど1本であるかを評価する（T-8 一意解評価）。

    戻り値: 解が1本なら True、0本または2本以上なら False。
    """
    n = count_solutions_on_graph(
        graph,
        start,
        goal,
        max_solutions=2,
        max_visits=max_visits,
    )
    return n == 1


def evaluate_uniqueness(
    graph: MazeGraph,
    start: int,
    goal: int,
    *,
    max_solutions: int = 2,
    max_visits: int = 100_000,
) -> Tuple[int, bool]:
    """
    一意解かどうかを評価し、解の数と一意フラグを返す（T-8）。

    戻り値: (num_solutions, is_unique)
      - num_solutions: 0, 1, または max_solutions で頭打ちの 2
      - is_unique: num_solutions == 1 のとき True
    """
    n = count_solutions_on_graph(
        graph,
        start,
        goal,
        max_solutions=max_solutions,
        max_visits=max_visits,
    )
    return n, n == 1


def pick_start_goal_from_path(
    graph: MazeGraph,
    path_points: List[PathPoint],
) -> Tuple[int | None, int | None]:
    """
    現在の仕様に基づき、main_path の両端座標から MazeGraph ノード ID を推定する。
    将来的に FeatureSummary や UI 指定を統合する際の差し替えポイント。
    """
    if not path_points or not graph.nodes:
        return None, None

    coord_to_id: Dict[tuple[int, int], int] = {
        (node.x, node.y): nid for nid, node in graph.nodes.items()
    }
    start_coord = (int(round(path_points[0].x)), int(round(path_points[0].y)))
    goal_coord = (int(round(path_points[-1].x)), int(round(path_points[-1].y)))
    start_id = coord_to_id.get(start_coord)
    goal_id = coord_to_id.get(goal_coord)
    return start_id, goal_id


def _build_subgraph(graph: MazeGraph, active_edge_ids: Set[int]) -> MazeGraph:
    """
    T-13: active_edge_ids に含まれるエッジのみを持つサブグラフを返す内部ヘルパー。
    nodes は元グラフと共有する（読み取り専用で使用）。
    """
    active_edges = [e for e in graph.edges if e.id in active_edge_ids]
    return MazeGraph(nodes=graph.nodes, edges=active_edges)


def prune_edges(
    graph: MazeGraph,
    start: int,
    goal: int,
    *,
    max_removals: int = 100,
    max_visits: int = 50_000,
) -> MazeGraph:
    """
    T-13: 余分なエッジを削減して一意解に近づける（枝削減）。

    エッジを1本ずつ試み、削除後に解数が減少しかつ最低1解が残る場合のみ削除を確定する。
    解数が 1 以下になるまで、または max_removals 回まで繰り返す。

    内部では max_solutions=10 でカウントし、3本以上の経路も正しく検出する。

    Args:
        graph:        対象グラフ
        start:        始点ノード ID
        goal:         終点ノード ID
        max_removals: 最大削除回数（無限ループ防止）
        max_visits:   DFS 訪問上限（大きなグラフ用）

    Returns:
        エッジを削減した新しい MazeGraph
    """
    _MAX_SOLS = 10  # 内部カウント上限（3本以上の経路も検出）
    active_ids: Set[int] = {e.id for e in graph.edges}

    for _ in range(max_removals):
        current = _build_subgraph(graph, active_ids)
        n = count_solutions_on_graph(
            current, start, goal,
            max_solutions=_MAX_SOLS, max_visits=max_visits,
        )
        if n <= 1:
            break

        removed = False
        for eid in sorted(active_ids):  # ソートで決定的に動作
            trial_ids = active_ids - {eid}
            trial = _build_subgraph(graph, trial_ids)
            n_trial = count_solutions_on_graph(
                trial, start, goal,
                max_solutions=_MAX_SOLS, max_visits=max_visits,
            )
            if n_trial < n and n_trial >= 1:  # 解数が減り、かつ経路が残る
                active_ids = trial_ids
                removed = True
                break

        if not removed:
            break

    return _build_subgraph(graph, active_ids)


def force_unique_solution(
    graph: MazeGraph,
    start: int,
    goal: int,
    *,
    max_removals: int = 100,
    max_visits: int = 100_000,
) -> Tuple[MazeGraph, bool]:
    """
    T-13: 一意解強制化アルゴリズム。

    手順:
      1. count_solutions_on_graph で解数を確認する
      2. 解が複数の場合: prune_edges で分岐エッジ（壁追加相当）を削除して一意化
      3. 再検査: 削減後に count_solutions で 1 解になったか確認する

    Args:
        graph:        対象グラフ
        start:        始点ノード ID
        goal:         終点ノード ID
        max_removals: prune_edges に渡す最大削除回数
        max_visits:   DFS 訪問上限

    Returns:
        (result_graph, is_unique):
          result_graph: 変換後の MazeGraph（一意解でない場合も最大限削減済み）
          is_unique:    変換後に一意解（解数 == 1）かどうか
    """
    n_initial = count_solutions_on_graph(
        graph, start, goal, max_solutions=2, max_visits=max_visits,
    )
    if n_initial <= 1:
        return graph, (n_initial == 1)

    # 枝削減で一意化を試みる
    result = prune_edges(
        graph, start, goal,
        max_removals=max_removals,
        max_visits=max_visits,
    )

    # 再検査（公開 API は max_solutions=2 で確認）
    n_final = count_solutions_on_graph(
        result, start, goal, max_solutions=2, max_visits=max_visits,
    )
    return result, (n_final == 1)


def solve_path(
    path_points: List[PathPoint],
    *,
    num_solutions_hint: int | None = None,
) -> SolveResult:
    """
    パス列に対して T-9 難易度指標を計算する。

    - turn_count: 連続3点で方向が変化した回数（閾値 1e-3 ラジアン）
    - path_length: len(path_points)
    - difficulty_score: min(1.0, turn_count / max(1, n-2))（曲がり角の密度）
    - num_solutions: num_solutions_hint をそのまま返す（解数は count_solutions_on_graph で別算出）
    """
    n = len(path_points)
    if n == 0:
        return SolveResult(
            has_solution=False,
            num_solutions=0 if num_solutions_hint is None else num_solutions_hint,
            difficulty_score=None,
            turn_count=0,
            path_length=0,
        )

    if n < 3:
        turns = 0
        difficulty = 0.0
    else:
        turns = 0
        for i in range(1, n - 1):
            p_prev = path_points[i - 1]
            p_cur = path_points[i]
            p_next = path_points[i + 1]
            v1x = p_cur.x - p_prev.x
            v1y = p_cur.y - p_prev.y
            v2x = p_next.x - p_cur.x
            v2y = p_next.y - p_cur.y
            ang1 = atan2(v1y, v1x)
            ang2 = atan2(v2y, v2x)
            if abs(ang2 - ang1) > 1e-3:
                turns += 1
        difficulty = min(1.0, turns / max(1, n - 2))

    if num_solutions_hint is None:
        num_solutions = 1
    else:
        num_solutions = num_solutions_hint

    return SolveResult(
        has_solution=True,
        num_solutions=num_solutions,
        difficulty_score=difficulty,
        turn_count=turns,    # T-9: 曲がり角の数
        path_length=n,       # T-9: 経路長
    )

