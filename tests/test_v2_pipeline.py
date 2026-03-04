from __future__ import annotations

import numpy as np

from backend.core.skeleton import edges_to_skeleton
from backend.core.graph_builder import skeleton_to_graph
from backend.core.path_finder import find_unicursal_like_path


def test_edges_to_skeleton_shape_and_dtype() -> None:
    edges = np.zeros((10, 12), dtype=bool)
    edges[3, 4] = True
    edges[3, 5] = True
    edges[4, 5] = True

    skeleton = edges_to_skeleton(edges)
    assert skeleton.shape == edges.shape
    assert skeleton.dtype == bool


def test_skeleton_to_graph_has_nodes_and_edges() -> None:
    skeleton = np.zeros((5, 5), dtype=bool)
    skeleton[2, 1] = True
    skeleton[2, 2] = True
    skeleton[2, 3] = True

    graph = skeleton_to_graph(skeleton)
    assert len(graph.nodes) >= 3
    assert len(graph.edges) >= 2


def test_find_unicursal_like_path_returns_non_empty_for_simple_line() -> None:
    skeleton = np.zeros((5, 5), dtype=bool)
    skeleton[2, 1] = True
    skeleton[2, 2] = True
    skeleton[2, 3] = True

    graph = skeleton_to_graph(skeleton)
    path_points = find_unicursal_like_path(graph)
    assert len(path_points) >= 3

