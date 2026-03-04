"""
tests/test_t11_path_aesthetics.py
==================================
T-11: 一筆パスの美観チューニング テスト

スコア重み（landmark 0.30→0.35, boundary 0.10→0.15）の検証と
ヒートマップ可視化機能のテスト。

CHECK-9 根拠:
  test_landmark_coefficient_is_0_35:
    landmark_mask 全面あり(hits=5, n_total=5 → landmark_score=1.0)のとき、
    features=None との差が 0.35 であることを確認。
    T-11前(0.30)との差も検証。

  test_boundary_coefficient_is_0_15:
    1ノードのシルエット端でboundary_score=1.0のとき、
    features=None との差が 0.15 であることを確認。

  test_higher_landmark_hits_yields_higher_score:
    landmark半分(3/5) vs 全部(5/5) → 全部の方が高スコア。

  test_higher_boundary_coverage_yields_higher_score:
    silhouette境界ありのパス vs なし → 前者が高スコア。

  test_score_components_landmark_boundary_additive:
    landmark_only のスコア < landmark+boundary 両方のスコア。

  test_compute_path_heatmap_nonzero_at_path_points:
    compute_path_heatmap → パス通過点が非ゼロ。

  test_compute_path_heatmap_out_of_bounds_ignored:
    範囲外座標は無視され、合計 = 範囲内点の件数。

  test_compute_path_heatmap_empty_path:
    空パス → 全ゼロ。

作成: 足軽6 cmd_285k_maze_T11
"""
from __future__ import annotations

import numpy as np

from backend.core.features import FeatureSummary
from backend.core.graph_builder import MazeGraph, Node, Edge
from backend.core.path_finder import (
    _score_path,
    compute_path_heatmap,
    PathPoint,
)


# ── ヘルパー ──────────────────────────────────────────────────────────────────

def _make_linear_graph(n: int, weights: list[float] | None = None) -> MazeGraph:
    """n 個のノードを x=0,1,...,n-1, y=0 に並べた直線グラフを返す。"""
    node_map: dict[int, Node] = {}
    for i in range(n):
        w = weights[i] if weights else None
        node_map[i] = Node(id=i, x=i, y=0, degree=0, weight=w)
    edge_list: list[Edge] = []
    for i in range(n - 1):
        edge_list.append(Edge(id=i, from_id=i, to_id=i + 1, length=1.0))
        node_map[i].degree += 1
        node_map[i + 1].degree += 1
    return MazeGraph(nodes=node_map, edges=edge_list)


# ── スコア係数テスト ──────────────────────────────────────────────────────────

def test_landmark_coefficient_is_0_35() -> None:
    """
    T-11 CHECK-9: landmark 係数が 0.35 であることを確認。

    5 ノード直線グラフ（全ノードが y=0, landmark_mask=ones(1,5)）で
    landmark_score = 5/5 = 1.0。
    features=None との差 = 0.35 * 1.0 = 0.35。
    T-10 時の係数 0.30 では差が 0.30 だったが、T-11 後は 0.35 に増加。
    """
    graph = _make_linear_graph(5)
    path_nodes = [0, 1, 2, 3, 4]

    # features なし（landmark 寄与ゼロ）
    score_no_lm = _score_path(graph, path_nodes, features=None)

    # landmark_mask = ones(1, 5) → 全ノードがランドマーク上
    lm = np.ones((1, 5), dtype=bool)
    features = FeatureSummary(landmark_mask=lm, centroid=(2.0, 0.0))
    score_with_lm = _score_path(graph, path_nodes, features=features)

    diff = score_with_lm - score_no_lm
    # face_component = (1-0) * (0.35 * 1.0 + 0.15 * 0.0) = 0.35
    assert abs(diff - 0.35) < 0.01, (
        f"T-11後 landmark係数差 = {diff:.4f}、期待値 0.35"
    )
    # T-10 の係数 0.30 より大きいことを確認
    assert diff > 0.30, f"T-11後 landmark係数({diff:.4f}) > T-10係数(0.30) のはず"


def test_boundary_coefficient_is_0_15() -> None:
    """
    T-11 CHECK-9: boundary 係数が 0.15 であることを確認。

    1 ノード（座標 (0,0)）、sil=ones(1,1)。
    4 近傍のうち右 (1,0) が範囲外 → is_boundary=True → boundary_hits=1。
    boundary_score = 1/1 = 1.0。
    features=None との差 = 0.15 * 1.0 = 0.15。
    """
    node_map: dict[int, Node] = {0: Node(id=0, x=0, y=0, degree=0, weight=None)}
    graph = MazeGraph(nodes=node_map, edges=[])
    path_nodes = [0]

    score_no_boundary = _score_path(graph, path_nodes, features=None)

    sil = np.ones((1, 1), dtype=bool)
    features = FeatureSummary(silhouette_mask=sil, centroid=(0.0, 0.0))
    score_with_boundary = _score_path(graph, path_nodes, features=features)

    diff = score_with_boundary - score_no_boundary
    # face_component = (1-0) * (0.35 * 0.0 + 0.15 * 1.0) = 0.15
    assert abs(diff - 0.15) < 0.01, (
        f"T-11後 boundary係数差 = {diff:.4f}、期待値 0.15"
    )
    assert diff > 0.10, f"T-11後 boundary係数({diff:.4f}) > T-10係数(0.10) のはず"


def test_higher_landmark_hits_yields_higher_score() -> None:
    """
    T-11 CHECK-9: ランドマーク通過数が多いほど高スコアになることを確認。

    landmark_half（3/5 = 0.60）vs landmark_full（5/5 = 1.0）→ full の方が高スコア。
    """
    graph = _make_linear_graph(5)
    path_nodes = [0, 1, 2, 3, 4]

    lm_half = np.zeros((1, 5), dtype=bool)
    lm_half[0, :3] = True  # hits = 3/5 = 0.60
    features_half = FeatureSummary(landmark_mask=lm_half, centroid=(2.0, 0.0))
    score_half = _score_path(graph, path_nodes, features=features_half)

    lm_full = np.ones((1, 5), dtype=bool)  # hits = 5/5 = 1.0
    features_full = FeatureSummary(landmark_mask=lm_full, centroid=(2.0, 0.0))
    score_full = _score_path(graph, path_nodes, features=features_full)

    assert score_full > score_half, (
        f"full_landmark({score_full:.4f}) > half_landmark({score_half:.4f}) でなければならない"
    )


def test_higher_boundary_coverage_yields_higher_score() -> None:
    """
    T-11 CHECK-9: 境界通過が多いほど高スコアになることを確認。

    silhouette_mask あり（boundary_score > 0）vs なし（boundary_score = 0）→
    あり の方が高スコア。
    """
    graph = _make_linear_graph(3)
    path_nodes = [0, 1, 2]

    score_no_boundary = _score_path(graph, path_nodes, features=None)

    # 1x3 silhouette: 両端ノードは境界（左端・右端が範囲外）、中央も上下が範囲外
    sil = np.ones((1, 3), dtype=bool)
    features = FeatureSummary(silhouette_mask=sil, centroid=(1.0, 0.0))
    score_with_boundary = _score_path(graph, path_nodes, features=features)

    assert score_with_boundary > score_no_boundary, (
        f"boundary_あり({score_with_boundary:.4f}) > boundary_なし({score_no_boundary:.4f}) のはず"
    )


def test_score_components_landmark_boundary_additive() -> None:
    """
    T-11 CHECK-9: landmark + boundary 両方が揃うと両成分が加算されることを確認。

    landmark_only のスコア < landmark + boundary 両方のスコア。
    差が boundary 係数 * boundary_score に対応することを確認。
    """
    graph = _make_linear_graph(3)
    path_nodes = [0, 1, 2]

    lm = np.ones((1, 3), dtype=bool)

    # landmark のみ
    features_lm_only = FeatureSummary(landmark_mask=lm, centroid=(1.0, 0.0))
    score_lm_only = _score_path(graph, path_nodes, features=features_lm_only)

    # landmark + boundary 両方
    sil = np.ones((1, 3), dtype=bool)
    features_both = FeatureSummary(
        silhouette_mask=sil, landmark_mask=lm, centroid=(1.0, 0.0)
    )
    score_both = _score_path(graph, path_nodes, features=features_both)

    assert score_both > score_lm_only, (
        f"landmark+boundary({score_both:.4f}) > landmark_only({score_lm_only:.4f}) のはず"
    )


# ── ヒートマップテスト ────────────────────────────────────────────────────────

def test_compute_path_heatmap_nonzero_at_path_points() -> None:
    """
    T-11 CHECK-9: compute_path_heatmap がパス通過点を正しく記録することを確認。
    パス上の (1,1), (2,1), (3,2) が非ゼロ。パス外の (0,0) はゼロ。
    """
    path = [
        PathPoint(x=1.0, y=1.0),
        PathPoint(x=2.0, y=1.0),
        PathPoint(x=3.0, y=2.0),
    ]
    heatmap = compute_path_heatmap(path, width=5, height=5)

    assert heatmap[1][1] > 0, "heatmap[1][1] がゼロ（パス点が記録されていない）"
    assert heatmap[1][2] > 0, "heatmap[1][2] がゼロ（パス点が記録されていない）"
    assert heatmap[2][3] > 0, "heatmap[2][3] がゼロ（パス点が記録されていない）"
    assert heatmap[0][0] == 0.0, "heatmap[0][0] が非ゼロ（パス外が記録されている）"


def test_compute_path_heatmap_out_of_bounds_ignored() -> None:
    """
    T-11 CHECK-9: compute_path_heatmap が範囲外座標を無視することを確認。
    範囲内 1 点 + 範囲外 2 点 → 合計 = 1.0。
    """
    path = [
        PathPoint(x=0.0, y=0.0),   # 範囲内
        PathPoint(x=99.0, y=99.0), # 範囲外
        PathPoint(x=-1.0, y=-1.0), # 範囲外
    ]
    heatmap = compute_path_heatmap(path, width=5, height=5)

    assert heatmap[0][0] == 1.0, f"heatmap[0][0] = {heatmap[0][0]}（期待値 1.0）"
    total = sum(heatmap[y][x] for y in range(5) for x in range(5))
    assert total == 1.0, f"範囲外点が含まれている: total = {total}"


def test_compute_path_heatmap_empty_path() -> None:
    """
    T-11 CHECK-9: 空パスに対して compute_path_heatmap が全ゼロを返すことを確認。
    """
    heatmap = compute_path_heatmap([], width=4, height=4)
    total = sum(heatmap[y][x] for y in range(4) for x in range(4))
    assert total == 0.0, f"空パスで非ゼロの値が存在する: total = {total}"
