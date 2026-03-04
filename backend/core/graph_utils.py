from __future__ import annotations

"""
Graph utility functions.
Common helpers for MazeGraph operations.
"""

from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from .graph_builder import MazeGraph


def build_adjacency(graph: MazeGraph) -> Dict[int, List[int]]:
    """
    Build an adjacency list from a MazeGraph.

    Args:
        graph: MazeGraph with nodes and edges.

    Returns:
        Dictionary mapping node ID to list of neighbor node IDs.
        Each edge creates bidirectional adjacency.
    """
    adj: Dict[int, List[int]] = {nid: [] for nid in graph.nodes.keys()}
    for e in graph.edges:
        adj[e.from_id].append(e.to_id)
        adj[e.to_id].append(e.from_id)
    return adj
