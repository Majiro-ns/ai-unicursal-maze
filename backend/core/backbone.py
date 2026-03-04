from __future__ import annotations

import collections
from typing import List, Tuple

import numpy as np

from .graph_builder import MazeGraph, skeleton_to_graph
from .skeleton import edges_to_skeleton


def _graph_diameter_endpoints(graph: MazeGraph) -> Tuple[int | None, int | None]:
    """
    Return endpoints of an approximate diameter using double BFS on an unweighted graph.
    """
    if not graph.nodes:
        return None, None

    def bfs_far(start: int) -> Tuple[int, dict[int, int]]:
        dist: dict[int, int] = {start: 0}
        q = collections.deque([start])
        far = start
        while q:
            cur = q.popleft()
            for nb in [e.to_id for e in graph.edges if e.from_id == cur] + [
                e.from_id for e in graph.edges if e.to_id == cur
            ]:
                if nb in dist:
                    continue
                dist[nb] = dist[cur] + 1
                q.append(nb)
                if dist[nb] > dist[far]:
                    far = nb
        return far, dist

    start = next(iter(graph.nodes.keys()))
    a, _ = bfs_far(start)
    b, dist_map = bfs_far(a)
    if dist_map.get(b, 0) == 0 and a != b:
        return None, None
    return a, b


def compute_backbone_endpoints(boundary_mask: np.ndarray | None, landmark_mask: np.ndarray | None) -> Tuple[int | None, int | None]:
    """
    Build a minimal backbone from silhouette boundary + landmarks, then return endpoints of its diameter.
    """
    if boundary_mask is None and landmark_mask is None:
        return None, None

    combined = None
    if boundary_mask is not None:
        combined = boundary_mask.astype(bool)
    if landmark_mask is not None:
        if combined is None:
            combined = landmark_mask.astype(bool)
        else:
            combined = np.logical_or(combined, landmark_mask.astype(bool))

    if combined is None or not combined.any():
        return None, None

    skeleton = edges_to_skeleton(combined)
    graph = skeleton_to_graph(skeleton)
    return _graph_diameter_endpoints(graph)
