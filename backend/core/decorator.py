from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from .graph_builder import MazeGraph
from .graph_utils import build_adjacency
from .path_finder import PathPoint


@dataclass
class DummyOptions:
    """
    ダミー枝生成用の簡易オプション。

    V2/V3 初期段階では内部でのみ使用し、MazeOptions には露出しない。
    """

    max_branches: int = 8
    max_branch_length: int = 20


def generate_dummy_branches(
    graph: MazeGraph,
    main_path: List[PathPoint],
    options: DummyOptions | None = None,
) -> List[List[PathPoint]]:
    """
    正解パスから分岐するダミー枝を簡易生成する。

    方針:
      - 正解パス上のノードから、まだ使われていない隣接ノードへ短い DFS を伸ばす。
      - main_path 上のノードは通らない。
      - max_branches / max_branch_length を超えない範囲で複数枝を作る。
    """
    if options is None:
        options = DummyOptions()

    if not graph.nodes or len(main_path) < 2:
        return []

    coord_to_id: Dict[tuple[int, int], int] = {
        (node.x, node.y): nid for nid, node in graph.nodes.items()
    }

    main_ids: Set[int] = set()
    for p in main_path:
        nid = coord_to_id.get((int(round(p.x)), int(round(p.y))))
        if nid is not None:
            main_ids.add(nid)

    adj = build_adjacency(graph)
    used_branch_nodes: Set[int] = set()
    branches: List[List[PathPoint]] = []

    for p in main_path:
        if len(branches) >= options.max_branches:
            break
        nid = coord_to_id.get((int(round(p.x)), int(round(p.y))))
        if nid is None:
            continue

        for nb in adj.get(nid, []):
            if nb in main_ids or nb in used_branch_nodes:
                continue

            branch_ids: List[int] = []
            visited: Set[int] = set(main_ids)

            def dfs(cur: int, depth: int) -> None:
                if (
                    depth >= options.max_branch_length
                    or len(branch_ids) >= options.max_branch_length
                ):
                    return
                visited.add(cur)
                branch_ids.append(cur)
                for nxt in adj.get(cur, []):
                    if nxt in visited or nxt in main_ids:
                        continue
                    dfs(nxt, depth + 1)

            dfs(nb, 0)

            if len(branch_ids) < 2:
                continue

            used_branch_nodes.update(branch_ids)

            branch_points: List[PathPoint] = []
            for bid in branch_ids:
                node = graph.nodes[bid]
                branch_points.append(PathPoint(x=float(node.x), y=float(node.y)))

            branches.append(branch_points)
            if len(branches) >= options.max_branches:
                break

    return branches

