from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, List

import numpy as np

from .features import FeatureSummary
from .graph_utils import build_adjacency


@dataclass
class Node:
    id: int
    x: int
    y: int
    degree: int = 0
    weight: float | None = None


@dataclass
class Edge:
    id: int
    from_id: int
    to_id: int
    length: float


@dataclass
class MazeGraph:
    nodes: Dict[int, Node]
    edges: List[Edge]


def _prune_small_components(graph: MazeGraph, min_component_size: int) -> MazeGraph:
    """
    Remove connected components smaller than min_component_size.
    """
    if min_component_size <= 1 or not graph.nodes:
        return graph

    adj = build_adjacency(graph)
    visited: set[int] = set()
    components: List[List[int]] = []

    for nid in graph.nodes.keys():
        if nid in visited:
            continue
        stack = [nid]
        comp: List[int] = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)
            for nb in adj.get(cur, []):
                if nb not in visited:
                    stack.append(nb)
        components.append(comp)

    keep_nodes: set[int] = set()
    for comp in components:
        if len(comp) >= min_component_size:
            keep_nodes.update(comp)

    if len(keep_nodes) == len(graph.nodes):
        return graph

    new_nodes: Dict[int, Node] = {
        nid: node for nid, node in graph.nodes.items() if nid in keep_nodes
    }
    new_edges: List[Edge] = [
        e for e in graph.edges if e.from_id in keep_nodes and e.to_id in keep_nodes
    ]

    for node in new_nodes.values():
        node.degree = 0
    for e in new_edges:
        new_nodes[e.from_id].degree += 1
        new_nodes[e.to_id].degree += 1

    return MazeGraph(nodes=new_nodes, edges=new_edges)


def apply_feature_weights(graph: MazeGraph, features: FeatureSummary) -> None:
    """
    Attach weights to nodes using FeatureSummary signals (T-11/T-12).

    Priority:
      - silhouette boundary = 1.0（輪郭境界を最優先）
      - landmark lines = 1.0（目鼻口等を最優先）
      - face mask interior: 0.78
      - face_band_mask 内: 0.72（T-12: 髪・顔帯域を優先、ランドマーク未満・顔マスク未満）
      - silhouette interior: 0.55（輪郭内側）
      - outside silhouette: 0.05
      - slight decay by distance from centroid (except boundary)
    """
    if not graph.nodes:
        return

    sil = features.refined_silhouette_mask if features.refined_silhouette_mask is not None else features.silhouette_mask
    face = features.face_mask
    landmark = features.landmark_mask
    face_band = getattr(features, "face_band_mask", None)  # T-12

    face_h: int | None = None
    face_w: int | None = None
    if isinstance(face, np.ndarray) and face.ndim == 2:
        face_h, face_w = face.shape

    lm_h: int | None = None
    lm_w: int | None = None
    if isinstance(landmark, np.ndarray) and landmark.ndim == 2:
        lm_h, lm_w = landmark.shape

    band_h: int | None = None
    band_w: int | None = None
    if isinstance(face_band, np.ndarray) and face_band.ndim == 2:
        band_h, band_w = face_band.shape

    cx: float | None = None
    cy: float | None = None
    if features.centroid is not None:
        cx, cy = features.centroid

    xs = [node.x for node in graph.nodes.values()]
    ys = [node.y for node in graph.nodes.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    max_dx = max(
        abs(max_x - (cx if cx is not None else min_x)),
        abs((cx if cx is not None else min_x) - min_x),
        1.0,
    )
    max_dy = max(
        abs(max_y - (cy if cy is not None else min_y)),
        abs((cy if cy is not None else min_y) - min_y),
        1.0,
    )

    for node in graph.nodes.values():
        base = 1.0

        in_silhouette = False
        is_boundary = False
        in_face = False
        on_landmark = False
        in_face_band = False  # T-12

        if face_band is not None and band_h is not None and band_w is not None:
            if 0 <= node.y < band_h and 0 <= node.x < band_w and face_band[node.y, node.x]:
                in_face_band = True

        if sil is not None:
            h, w = sil.shape
            if 0 <= node.y < h and 0 <= node.x < w and sil[node.y, node.x]:
                in_silhouette = True
                # 4-neighbors: if any is False/outside, treat as boundary
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx = node.x + dx
                    ny = node.y + dy
                    if 0 <= ny < h and 0 <= nx < w:
                        if not sil[ny, nx]:
                            is_boundary = True
                            break
                    else:
                        is_boundary = True
                        break

        if face is not None and face_h is not None and face_w is not None:
            if 0 <= node.y < face_h and 0 <= node.x < face_w and face[node.y, node.x]:
                in_face = True

        if landmark is not None and lm_h is not None and lm_w is not None:
            if 0 <= node.y < lm_h and 0 <= node.x < lm_w and landmark[node.y, node.x]:
                on_landmark = True

        if in_silhouette:
            if is_boundary:
                base = 1.0
            else:
                base = 0.55
        else:
            base = 0.05

        if in_face:
            base = max(base, 0.78)

        if in_face_band:
            base = max(base, 0.72)  # T-12: 髪・顔帯域（face_band）を優先

        if on_landmark:
            base = max(base, 1.0)  # T-11: ランドマークを最優先（0.95→1.0）

        if not is_boundary and cx is not None and cy is not None:
            dx = abs(node.x - cx) / max_dx
            dy = abs(node.y - cy) / max_dy
            dist_norm = min(1.0, (dx + dy) * 0.5)
            base *= (1.0 - 0.2 * dist_norm)

        node.weight = max(0.0, min(1.0, base))


def skeleton_to_graph(skeleton: np.ndarray, *, min_component_size: int = 1) -> MazeGraph:
    """
    Build MazeGraph from a skeleton image (H, W, bool).
    - True pixels become nodes.
    - 4-neighbor connections become edges (length 1.0 for axis neighbors).
    - Optionally prune small connected components.
    """
    if skeleton.ndim != 2:
        raise ValueError("skeleton must be 2D")

    h, w = skeleton.shape
    nodes: Dict[int, Node] = {}
    id_from_coord: Dict[tuple[int, int], int] = {}

    next_node_id = 0
    for y in range(h):
        for x in range(w):
            if not skeleton[y, x]:
                continue
            node_id = next_node_id
            next_node_id += 1
            nodes[node_id] = Node(id=node_id, x=x, y=y, degree=0)
            id_from_coord[(x, y)] = node_id

    edges: List[Edge] = []
    edge_id = 0

    for (x, y), nid in id_from_coord.items():
        for dx, dy in ((1, 0), (0, 1)):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx]:
                nid2 = id_from_coord[(nx, ny)]
                length = sqrt(dx * dx + dy * dy)
                edges.append(Edge(id=edge_id, from_id=nid, to_id=nid2, length=length))
                edge_id += 1
                nodes[nid].degree += 1
                nodes[nid2].degree += 1

    graph = MazeGraph(nodes=nodes, edges=edges)
    graph = _prune_small_components(graph, min_component_size=min_component_size)
    return graph
