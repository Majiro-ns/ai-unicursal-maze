from __future__ import annotations

"""
スケルトングラフ (MazeGraph) から「壁/通路」を持つグリッド表現 (MazeGrid)
への変換と、グリッド上での簡易ソルバのためのユーティリティ。
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .graph_builder import MazeGraph


@dataclass
class MazeGrid:
    width: int
    height: int
    cells: np.ndarray  # shape = (H, W), bool: True が通路、False が壁
    start: Optional[Tuple[int, int]] = None  # (x, y)
    goal: Optional[Tuple[int, int]] = None   # (x, y)


def graph_to_grid(
    graph: MazeGraph,
    *,
    scale: int = 2,
    padding: int = 2,
    start_id: int | None = None,
    goal_id: int | None = None,
) -> MazeGrid:
    """
    MazeGraph を MazeGrid に変換する。

    - Node の (x, y) 座標範囲を取得し、scale 倍 + padding 分だけ大きいグリッドを確保する。
    - 各エッジを直線で補間しながら通路セルとして塗る。
    - start_id / goal_id が指定されていれば、それに対応するセルを start / goal に設定し、
      その周囲も通路として確保する。
    """
    if not graph.nodes:
        cells = np.zeros((1, 1), dtype=bool)
        return MazeGrid(width=1, height=1, cells=cells)

    xs = [n.x for n in graph.nodes.values()]
    ys = [n.y for n in graph.nodes.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    src_width = max_x - min_x + 1
    src_height = max_y - min_y + 1

    width = src_width * scale + 2 * padding
    height = src_height * scale + 2 * padding

    cells = np.zeros((height, width), dtype=bool)

    def to_grid_coord(x: float, y: float) -> tuple[int, int]:
        gx = int(round((x - min_x) * scale)) + padding
        gy = int(round((y - min_y) * scale)) + padding
        gx = max(0, min(width - 1, gx))
        gy = max(0, min(height - 1, gy))
        return gx, gy

    def mark_path_cell(gx: int, gy: int) -> None:
        """通路セルを太めにマークする（中心 + 4 近傍）。"""
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = gx + dx
            ny = gy + dy
            if 0 <= nx < width and 0 <= ny < height:
                cells[ny, nx] = True

    # ノード自身を通路としてマーク
    for node in graph.nodes.values():
        gx, gy = to_grid_coord(node.x, node.y)
        mark_path_cell(gx, gy)

    # エッジを線で補間して通路としてマーク
    for edge in graph.edges:
        n1 = graph.nodes[edge.from_id]
        n2 = graph.nodes[edge.to_id]
        x1, y1 = float(n1.x), float(n1.y)
        x2, y2 = float(n2.x), float(n2.y)
        dx = x2 - x1
        dy = y2 - y1
        steps = int(max(abs(dx), abs(dy)) * scale)
        steps = max(1, steps)
        for i in range(steps + 1):
            t = i / steps
            x = x1 + dx * t
            y = y1 + dy * t
            gx, gy = to_grid_coord(x, y)
            mark_path_cell(gx, gy)

    grid_start: Optional[Tuple[int, int]] = None
    grid_goal: Optional[Tuple[int, int]] = None
    if start_id is not None and start_id in graph.nodes:
        grid_start = to_grid_coord(graph.nodes[start_id].x, graph.nodes[start_id].y)
        mark_path_cell(*grid_start)
    if goal_id is not None and goal_id in graph.nodes:
        grid_goal = to_grid_coord(graph.nodes[goal_id].x, graph.nodes[goal_id].y)
        mark_path_cell(*grid_goal)

    return MazeGrid(width=width, height=height, cells=cells, start=grid_start, goal=grid_goal)


def count_solutions_on_grid(
    grid: MazeGrid,
    *,
    max_solutions: int = 2,
    max_visits: int = 100_000,
) -> int:
    """
    MazeGrid 上で start → goal の単純路を DFS で列挙し、
    最大 max_solutions 個まで数える簡易ソルバ。

    - 戻り値:
      - 0: 解なし
      - 1: 一意解
      - 2: 複数解（max_solutions=2 のため 2 で頭打ち）
    """
    if grid.start is None or grid.goal is None:
        return 0

    sx, sy = grid.start
    gx, gy = grid.goal
    if not grid.cells[sy, sx] or not grid.cells[gy, gx]:
        return 0

    h, w = grid.height, grid.width

    def neighbors(x: int, y: int):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < w and 0 <= ny < h and grid.cells[ny, nx]:
                yield nx, ny

    visited = set()
    num_solutions = 0
    visits = 0

    def dfs(x: int, y: int) -> None:
        nonlocal num_solutions, visits
        if num_solutions >= max_solutions or visits >= max_visits:
            return
        visits += 1
        if (x, y) == (gx, gy):
            num_solutions += 1
            return
        visited.add((x, y))
        for nx, ny in neighbors(x, y):
            if (nx, ny) not in visited:
                dfs(nx, ny)
        visited.remove((x, y))

    dfs(sx, sy)
    return num_solutions

