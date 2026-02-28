from __future__ import annotations

from dataclasses import dataclass
from math import atan2
from typing import Dict, List, Set, Tuple

from .graph_builder import MazeGraph
from .graph_utils import build_adjacency
from .path_finder import PathPoint


@dataclass
class SolveResult:
    has_solution: bool
    num_solutions: int | None
    difficulty_score: float | None
    # T-9: 難易度指標
    turn_count: int | None = None    # パス内の方向転換の総数
    path_length: int | None = None   # パスのノード数（経路長）


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


def solve_path(
    path_points: List[PathPoint],
    *,
    num_solutions_hint: int | None = None,
) -> SolveResult:
    """
    パス列に対して簡易な難易度スコアを計算し、
    併せて「解の個数のヒント」を結果にまとめる。
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

