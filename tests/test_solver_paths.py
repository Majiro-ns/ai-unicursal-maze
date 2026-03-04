from __future__ import annotations

from backend.core.graph_builder import MazeGraph, Node, Edge
from backend.core.solver import count_solutions_on_graph


def _make_graph(nodes: list[tuple[int, int]], edges: list[tuple[int, int]]) -> MazeGraph:
    node_map: dict[int, Node] = {}
    for i, (x, y) in enumerate(nodes):
        node_map[i] = Node(id=i, x=x, y=y, degree=0, weight=None)
    edge_list: list[Edge] = []
    for eid, (a, b) in enumerate(edges):
        edge_list.append(Edge(id=eid, from_id=a, to_id=b, length=1.0))
        node_map[a].degree += 1
        node_map[b].degree += 1
    return MazeGraph(nodes=node_map, edges=edge_list)


def test_count_solutions_on_graph_no_solution() -> None:
    graph = _make_graph(nodes=[(0, 0), (1, 0)], edges=[])
    count = count_solutions_on_graph(graph, start=0, goal=1, max_solutions=2)
    assert count == 0


def test_count_solutions_on_graph_single_solution() -> None:
    graph = _make_graph(nodes=[(0, 0), (1, 0), (2, 0)], edges=[(0, 1), (1, 2)])
    count = count_solutions_on_graph(graph, start=0, goal=2, max_solutions=2)
    assert count == 1


def test_count_solutions_on_graph_two_solutions_capped() -> None:
    graph = _make_graph(
        nodes=[(0, 0), (1, 0), (1, 1), (0, 1)],
        edges=[(0, 1), (1, 2), (2, 3), (3, 0)],
    )
    count = count_solutions_on_graph(graph, start=0, goal=2, max_solutions=2)
    assert count == 2
