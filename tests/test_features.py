from __future__ import annotations

import numpy as np
import pytest

from backend.core.features import (
    FeatureSummary,
    compute_face_band_from_mask,
    compute_geometric_landmark_mask,
    extract_features_from_edges,
)
from backend.core.graph_builder import MazeGraph, Node, Edge, apply_feature_weights


# ---------------------------------------------------------------------------
# 既存テスト
# ---------------------------------------------------------------------------


def test_extract_features_from_edges_empty() -> None:
    edges = np.zeros((10, 10), dtype=bool)
    summary = extract_features_from_edges(edges)
    assert isinstance(summary, FeatureSummary)
    assert summary.centroid is None
    assert summary.bounding_boxes == []


def test_extract_features_from_edges_basic() -> None:
    edges = np.zeros((10, 10), dtype=bool)
    edges[2, 3] = True
    edges[4, 7] = True

    summary = extract_features_from_edges(edges)
    assert summary.centroid is not None
    cx, cy = summary.centroid
    assert 0.0 <= cx <= 9.0
    assert 0.0 <= cy <= 9.0
    assert len(summary.bounding_boxes) == 1
    x_min, y_min, x_max, y_max = summary.bounding_boxes[0]
    assert 0 <= x_min <= x_max < 10
    assert 0 <= y_min <= y_max < 10


# ---------------------------------------------------------------------------
# T-12: compute_face_band_from_mask
# ---------------------------------------------------------------------------


def test_compute_face_band_from_mask_basic() -> None:
    """face_mask の bounding box 全体が True になる。"""
    face_mask = np.zeros((50, 50), dtype=bool)
    face_mask[10:40, 15:35] = True  # 30x20 の矩形
    band = compute_face_band_from_mask(face_mask)
    assert band.shape == face_mask.shape
    # 矩形内部は全て True
    assert band[10:40, 15:35].all()
    # 矩形外の一部は False（上端）
    assert not band[0:10, :].any()


def test_compute_face_band_from_mask_empty() -> None:
    """face_mask が全 False のとき、全 False を返す。"""
    face_mask = np.zeros((20, 20), dtype=bool)
    band = compute_face_band_from_mask(face_mask)
    assert not band.any()
    assert band.shape == (20, 20)


# ---------------------------------------------------------------------------
# T-12: compute_geometric_landmark_mask
# ---------------------------------------------------------------------------


def test_compute_geometric_landmark_mask_basic() -> None:
    """face_mask から目・鼻・口ゾーンが推定される。"""
    face_mask = np.zeros((100, 100), dtype=bool)
    face_mask[10:90, 20:80] = True  # 顔領域: 高さ80, 幅60
    lm = compute_geometric_landmark_mask(face_mask)
    assert lm.shape == face_mask.shape
    # ランドマークマスクは非空
    assert lm.any()
    # face_mask 外は False
    assert not np.logical_and(lm, ~face_mask).any()


def test_compute_geometric_landmark_mask_empty() -> None:
    """face_mask が全 False のとき、全 False を返す。"""
    face_mask = np.zeros((30, 30), dtype=bool)
    lm = compute_geometric_landmark_mask(face_mask)
    assert not lm.any()


def test_compute_geometric_landmark_mask_covers_face_zones() -> None:
    """目・鼻・口ゾーンがそれぞれ推定結果に含まれている。"""
    H, W = 120, 100
    face_mask = np.zeros((H, W), dtype=bool)
    # 顔領域: y=10..100, x=10..90 (高さ90, 幅80)
    face_mask[10:100, 10:90] = True
    fh = 90
    fw = 80
    y0, x0 = 10, 10
    lm = compute_geometric_landmark_mask(face_mask)

    # 左目の中心付近
    left_eye_cy = y0 + int(fh * 0.325)
    left_eye_cx = x0 + int(fw * 0.25)
    assert lm[left_eye_cy, left_eye_cx], "左目中心がランドマークに含まれない"

    # 口の中心付近
    mouth_cy = y0 + int(fh * 0.72)
    mouth_cx = x0 + int(fw * 0.50)
    assert lm[mouth_cy, mouth_cx], "口中心がランドマークに含まれない"


# ---------------------------------------------------------------------------
# T-12: apply_feature_weights の重み優先順位
# ---------------------------------------------------------------------------


def _make_single_node_graph(x: int, y: int) -> MazeGraph:
    """座標 (x, y) の単一ノードグラフを返す。"""
    node = Node(id=0, x=x, y=y, degree=0)
    return MazeGraph(nodes={0: node}, edges=[])


def test_apply_feature_weights_landmark_gets_max_weight() -> None:
    """ランドマーク上のノードは weight=1.0 を得る。"""
    g = _make_single_node_graph(5, 5)
    lm_mask = np.zeros((20, 20), dtype=bool)
    lm_mask[5, 5] = True
    features = FeatureSummary(landmark_mask=lm_mask)
    apply_feature_weights(g, features)
    assert g.nodes[0].weight == pytest.approx(1.0, abs=1e-6)


def test_apply_feature_weights_face_mask_boost() -> None:
    """顔マスク内のノードは weight >= 0.78 を得る（ランドマークなし）。"""
    g = _make_single_node_graph(5, 5)
    face = np.zeros((20, 20), dtype=bool)
    face[5, 5] = True
    features = FeatureSummary(face_mask=face)
    apply_feature_weights(g, features)
    assert g.nodes[0].weight is not None
    assert g.nodes[0].weight >= 0.78 * 0.8  # centroid decay 20% 許容


def test_apply_feature_weights_priority_ordering() -> None:
    """重み優先順位: landmark ≥ face_mask ≥ face_band > outside。"""
    H, W = 40, 40

    def _weight(x: int, y: int, features: FeatureSummary) -> float:
        g = _make_single_node_graph(x, y)
        apply_feature_weights(g, features)
        return float(g.nodes[0].weight or 0.0)

    lm = np.zeros((H, W), dtype=bool)
    lm[10, 10] = True
    face = np.zeros((H, W), dtype=bool)
    face[20, 20] = True
    band = np.zeros((H, W), dtype=bool)
    band[30, 30] = True

    features = FeatureSummary(landmark_mask=lm, face_mask=face, face_band_mask=band)

    w_lm = _weight(10, 10, features)
    w_face = _weight(20, 20, features)
    w_band = _weight(30, 30, features)
    w_out = _weight(5, 5, features)

    assert w_lm >= w_face, f"landmark({w_lm}) should >= face({w_face})"
    assert w_face >= w_band, f"face({w_face}) should >= band({w_band})"
    assert w_band > w_out, f"band({w_band}) should > outside({w_out})"


# ---------------------------------------------------------------------------
# T-12: extract_features_from_edges の auto-generation
# ---------------------------------------------------------------------------


def test_extract_features_auto_face_band_from_face_mask() -> None:
    """face_mask を渡すと face_band_mask が自動生成される。"""
    edges = np.zeros((50, 50), dtype=bool)
    edges[5, 5] = True
    face = np.zeros((50, 50), dtype=bool)
    face[10:40, 15:35] = True
    summary = extract_features_from_edges(edges, face_mask=face)
    assert summary.face_band_mask is not None
    assert summary.face_band_mask.any()
    # face_mask の bounding box 内が True であること
    assert summary.face_band_mask[10:40, 15:35].all()


def test_extract_features_auto_geometric_landmark_when_no_landmark_given() -> None:
    """face_mask があり landmark_mask を渡さないと、幾何的ランドマークが自動生成される。"""
    edges = np.zeros((60, 60), dtype=bool)
    edges[5, 5] = True
    face = np.zeros((60, 60), dtype=bool)
    face[5:55, 5:55] = True  # 大きな顔領域
    summary = extract_features_from_edges(edges, face_mask=face, landmark_mask=None)
    assert summary.landmark_mask is not None
    assert summary.landmark_mask.any()
    # 渡した landmark_mask が優先されること（実 landmark を渡した場合上書きされない）
    real_lm = np.zeros((60, 60), dtype=bool)
    real_lm[30, 30] = True
    summary2 = extract_features_from_edges(edges, face_mask=face, landmark_mask=real_lm)
    # real_lm はリサイズして保持されるはず
    assert summary2.landmark_mask is not None
    assert summary2.landmark_mask[30, 30]

