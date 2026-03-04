from __future__ import annotations

import numpy as np

from backend.core.features import extract_features_from_edges, FeatureSummary


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

