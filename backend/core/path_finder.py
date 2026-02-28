from __future__ import annotations

import logging
from dataclasses import dataclass
from math import atan2, pi
from typing import Dict, List, Set

from .features import FeatureSummary
from .graph_builder import MazeGraph, Node
from .graph_utils import build_adjacency

logger = logging.getLogger(__name__)


@dataclass
class PathPoint:
    x: float
    y: float


def _pick_start_candidates(
    graph: MazeGraph,
    features: FeatureSummary | None = None,
    max_candidates: int = 24,
) -> List[int]:
    """
    Prefer nodes near image edge, with high weight, and on silhouette boundary/landmarks.
    """
    if not graph.nodes:
        return []

    xs = [node.x for node in graph.nodes.values()]
    ys = [node.y for node in graph.nodes.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def boundary_hint(node: Node) -> int:
        if features is None or features.silhouette_mask is None:
            return 0
        mask = features.silhouette_mask
        h, w = mask.shape
        if not (0 <= node.y < h and 0 <= node.x < w):
            return 0
        if not mask[node.y, node.x]:
            return 0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = node.x + dx
            ny = node.y + dy
            if not (0 <= ny < h and 0 <= nx < w) or not mask[ny, nx]:
                return 1
        return 0

    def landmark_hint(node: Node) -> int:
        if features is None or features.landmark_mask is None:
            return 0
        lm = features.landmark_mask
        h, w = lm.shape
        if 0 <= node.y < h and 0 <= node.x < w and lm[node.y, node.x]:
            return 1
        return 0

    def node_priority(nid: int) -> tuple[float, int, int, int]:
        node = graph.nodes[nid]
        weight = node.weight if node.weight is not None else 0.0
        edge_dist = min(
            node.x - min_x,
            max_x - node.x,
            node.y - min_y,
            max_y - node.y,
        )
        boundary = boundary_hint(node)
        landmark = landmark_hint(node)
        return (
            -(weight + 0.6 * boundary + 0.6 * landmark),
            edge_dist,
            -node.degree,
            node.x,
            node.y,
        )

    ordered = sorted(graph.nodes.keys(), key=node_priority)
    return ordered[:max_candidates]


def _node_boundary_flag(node: Node, features: FeatureSummary | None) -> int:
    if features is None or features.silhouette_mask is None:
        return 0
    mask = features.silhouette_mask
    h, w = mask.shape
    if not (0 <= node.y < h and 0 <= node.x < w):
        return 0
    if not mask[node.y, node.x]:
        return 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx = node.x + dx
        ny = node.y + dy
        if not (0 <= ny < h and 0 <= nx < w) or not mask[ny, nx]:
            return 1
    return 0


def _node_landmark_flag(node: Node, features: FeatureSummary | None) -> int:
    if features is None or features.landmark_mask is None:
        return 0
    lm = features.landmark_mask
    h, w = lm.shape
    if 0 <= node.y < h and 0 <= node.x < w and lm[node.y, node.x]:
        return 1
    return 0


def _neighbor_priority(
    graph: MazeGraph,
    features: FeatureSummary | None,
    cur: int,
    parent: int | None,
    nb: int,
) -> tuple[float, float, int, int]:
    node = graph.nodes[cur]
    nb_node = graph.nodes[nb]
    w = nb_node.weight if nb_node.weight is not None else 0.0

    boundary = _node_boundary_flag(nb_node, features)
    landmark = _node_landmark_flag(nb_node, features)

    if parent is None:
        turn_penalty = 0.0
    else:
        p_node = graph.nodes[parent]
        vx1 = node.x - p_node.x
        vy1 = node.y - p_node.y
        vx2 = nb_node.x - node.x
        vy2 = nb_node.y - node.y
        turn_penalty = abs(vx1 * vy2 - vy1 * vx2)

    # Prefer high weight + boundary/landmark, then straighter lines.
    return (-(w + 0.3 * boundary + 0.4 * landmark), turn_penalty, -nb_node.degree, nb_node.x, nb_node.y)


def _beam_search_path(
    graph: MazeGraph,
    adj: Dict[int, List[int]],
    start: int,
    features: FeatureSummary | None,
    max_steps: int,
    beam_width: int = 20,
    branch_factor: int = 4,
) -> List[int]:
    """
    Beam search to expand multiple paths and keep the best-scoring candidates.
    """
    if start not in graph.nodes:
        return []

    beams: List[List[int]] = [[start]]
    best: List[int] = [start]

    for _ in range(max_steps):
        candidates: List[List[int]] = []
        for path in beams:
            cur = path[-1]
            parent = path[-2] if len(path) >= 2 else None
            neighbors = adj.get(cur, [])
            neighbors = sorted(neighbors, key=lambda nb: _neighbor_priority(graph, features, cur, parent, nb))
            visited_set = set(path)
            for nb in neighbors[:branch_factor]:
                if nb in visited_set:
                    continue
                candidates.append(path + [nb])

        if not candidates:
            break

        scored = sorted(
            candidates,
            key=lambda p: _score_path(graph, p, features),
            reverse=True,
        )
        beams = scored[:beam_width]
        if _score_path(graph, beams[0], features) >= _score_path(graph, best, features):
            best = beams[0]

    return best


def find_unicursal_like_path(
    graph: MazeGraph,
    features: FeatureSummary | None = None,
    *,
    max_steps: int = 10_000,
    debug: bool = False,
    start_candidates_override: List[int] | None = None,
) -> List[PathPoint]:
    """
    Evaluate multiple starts with beam search, return the best single path.
    """
    if not graph.nodes:
        return []

    adj = build_adjacency(graph)
    start_candidates = start_candidates_override or _pick_start_candidates(graph, features=features)

    best_path_nodes: List[int] = []
    best_score: float = -1.0

    for start in start_candidates:
        path_nodes = _beam_search_path(
            graph,
            adj,
            start,
            features,
            max_steps=max_steps,
        )
        if not path_nodes:
            continue

        score, length_score, weight_score, curvature_score, landmark_score, boundary_score = _score_path(
            graph,
            path_nodes,
            features,
            return_components=True,
        )

        if debug and features is not None:
            logger.debug(
                "path_score start=%s len=%d weight=%.3f length=%.3f curvature=%.3f landmark=%.3f final=%.3f",
                start,
                len(path_nodes),
                weight_score,
                length_score,
                curvature_score,
                boundary_score,
                landmark_score,
                score,
            )

        if score > best_score or (score == best_score and len(path_nodes) > len(best_path_nodes)):
            best_score = score
            best_path_nodes = path_nodes

    path_points: List[PathPoint] = []
    for nid in best_path_nodes:
        node: Node = graph.nodes[nid]
        path_points.append(PathPoint(x=float(node.x), y=float(node.y)))

    return path_points


def _turn_angle_abs(
    prev_node: Node,
    cur_node: Node,
    next_node: Node,
) -> float:
    """
    Absolute turn angle in [0, π].
    """
    vx1 = cur_node.x - prev_node.x
    vy1 = cur_node.y - prev_node.y
    vx2 = next_node.x - cur_node.x
    vy2 = next_node.y - cur_node.y
    ang1 = atan2(vy1, vx1)
    ang2 = atan2(vy2, vx2)
    delta = ang2 - ang1
    delta = (delta + pi) % (2 * pi) - pi
    return abs(delta)


def _score_path(
    graph: MazeGraph,
    path_nodes: List[int],
    features: FeatureSummary | None = None,
    *,
    return_components: bool = False,
) -> tuple[float, float, float, float, float, float] | float:
    """
    Path scoring:
      - length_score    : path node count / total nodes
      - weight_score    : mean node weight
      - curvature_score : smoothness (1=smooth, 0=all sharp)
      - landmark_score  : fraction of nodes on landmark mask
      - boundary_score  : fraction of nodes on silhouette boundary
    """
    if not path_nodes or not graph.nodes:
        empty = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return empty if return_components else 0.0

    n_total = max(1, len(graph.nodes))
    n_path = len(path_nodes)

    length_score = min(1.0, n_path / n_total)

    weight_sum = 0.0
    for nid in path_nodes:
        node = graph.nodes.get(nid)
        if node is None:
            continue
        w = node.weight if node.weight is not None else 0.0
        weight_sum += w
    weight_score = weight_sum / n_total

    if n_path < 3:
        curvature_score = 1.0
    else:
        turns: List[float] = []
        for i in range(1, n_path - 1):
            prev_node = graph.nodes[path_nodes[i - 1]]
            cur_node = graph.nodes[path_nodes[i]]
            next_node = graph.nodes[path_nodes[i + 1]]
            turns.append(_turn_angle_abs(prev_node, cur_node, next_node))
        if not turns:
            curvature_score = 1.0
        else:
            avg_turn = sum(turns) / len(turns)
            curvature_score = max(0.0, 1.0 - avg_turn / pi)

    landmark_score = 0.0
    boundary_score = 0.0
    if features is not None and features.landmark_mask is not None:
        lm = features.landmark_mask
        h, w = lm.shape
        hits = 0
        for nid in path_nodes:
            node = graph.nodes.get(nid)
            if node is None:
                continue
            if 0 <= node.y < h and 0 <= node.x < w and lm[node.y, node.x]:
                hits += 1
        landmark_score = hits / n_total

    if features is not None and features.silhouette_mask is not None:
        sil = features.silhouette_mask
        h, w = sil.shape
        hits = 0
        for nid in path_nodes:
            node = graph.nodes.get(nid)
            if node is None:
                continue
            if not (0 <= node.y < h and 0 <= node.x < w):
                continue
            if not sil[node.y, node.x]:
                continue
            # boundary check
            is_boundary = False
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx = node.x + dx
                ny = node.y + dy
                if not (0 <= ny < h and 0 <= nx < w) or not sil[ny, nx]:
                    is_boundary = True
                    break
            if is_boundary:
                hits += 1
        boundary_score = hits / n_total

    # Favor longer paths while keeping weight/face cues important.
    # Landmark score boosted to 30% to prioritize routes through facial features.
    score = (
        0.3 * length_score
        + 0.25 * weight_score
        + 0.05 * curvature_score
        + 0.3 * landmark_score
        + 0.1 * boundary_score
    )

    if return_components:
        return score, length_score, weight_score, curvature_score, landmark_score, boundary_score
    return score
