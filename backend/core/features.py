from __future__ import annotations

"""
フェーズ 1: 特徴抽出フェーズ用の型と簡易実装。

V2.x では、後続フェーズから参照できる FeatureSummary と
エッジベースの簡単な特徴抽出を提供する。
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class FeatureSummary:
    """
    画像やエッジマップから得られた特徴量のサマリ。

    現時点で扱う主な情報:
      - エッジ画素の重心 (centroid)
      - エッジ量が多い領域のバウンディングボックス (bounding_boxes)
      - 前景シルエットのマスク (silhouette_mask)
      - シルエットごとのバウンディングボックス (silhouette_boxes)
      - シルエット境界以外の「内部線」(internal_edges)
      - 単純化された主シルエットマスク (refined_silhouette_mask)
      - 顔領域マスク (face_mask)
      - 顔ランドマーク線マスク (landmark_mask)
      - 顔帯域マスク (face_band_mask) … T-12: UI の face_band に対応。帯域内ノードの重みを上げる。
    """

    centroid: Optional[Tuple[float, float]] = None
    bounding_boxes: List[Tuple[int, int, int, int]] | None = None

    silhouette_mask: Optional[np.ndarray] = None  # shape = (H, W), bool
    silhouette_boxes: List[Tuple[int, int, int, int]] | None = None

    internal_edges: Optional[np.ndarray] = None  # shape = (H, W), bool
    refined_silhouette_mask: Optional[np.ndarray] = None  # shape = (H, W), bool

    face_mask: Optional[np.ndarray] = None  # shape = (H, W), bool
    landmark_mask: Optional[np.ndarray] = None  # shape = (H, W), bool
    face_band_mask: Optional[np.ndarray] = None  # shape = (H, W), bool. T-12: 髪・顔帯域（UI face_band）

    def __post_init__(self) -> None:
        if self.bounding_boxes is None:
            self.bounding_boxes = []
        if self.silhouette_boxes is None:
            self.silhouette_boxes = []


def extract_features_from_edges(
    edges: np.ndarray,
    face_mask: Optional[np.ndarray] = None,
    landmark_mask: Optional[np.ndarray] = None,
) -> FeatureSummary:
    """
    エッジマップ (H, W, bool / 0-1) から簡易な FeatureSummary を生成する。

    現時点では:
      - True ピクセルの重心
      - 1 つの大きなバウンディングボックス
      - 可能なら、前景シルエットのマスクとコンポーネントごとの bbox
      - シルエット境界以外の内部線マスク
      - 単純化された主シルエットマスク
      - 顔領域マスク／ランドマーク線マスク（あれば）
    を計算する。
    """
    if edges.ndim != 2:
        raise ValueError("edges は 2 次元配列である必要があります")

    edges_bool = edges.astype(bool)
    ys, xs = np.where(edges_bool)
    if ys.size == 0:
        return FeatureSummary()

    y_mean = float(ys.mean())
    x_mean = float(xs.mean())

    y_min = int(ys.min())
    y_max = int(ys.max())
    x_min = int(xs.min())
    x_max = int(xs.max())

    silhouette_mask: Optional[np.ndarray] = None
    silhouette_boxes: List[Tuple[int, int, int, int]] = []
    internal_edges: Optional[np.ndarray] = None
    refined_silhouette_mask: Optional[np.ndarray] = None
    face_mask_resized: Optional[np.ndarray] = None
    landmark_mask_resized: Optional[np.ndarray] = None

    try:
        from skimage import morphology, measure, segmentation, draw, transform  # type: ignore[import]

        # エッジを膨張・閉じて、「塊」としてのシルエットを得る。
        dilated = morphology.dilation(edges_bool, morphology.disk(1))
        closed = morphology.closing(dilated, morphology.disk(1))

        labeled = measure.label(closed)
        max_label = int(labeled.max())
        if max_label > 0:
            props = measure.regionprops(labeled)
            silhouette_mask = np.zeros_like(edges_bool)

            largest_area = 0
            largest_label = 0

            for p in props:
                min_row, min_col, max_row, max_col = p.bbox
                silhouette_boxes.append((min_col, min_row, max_col - 1, max_row - 1))
                silhouette_mask[labeled == p.label] = True

                if p.area > largest_area:
                    largest_area = int(p.area)
                    largest_label = int(p.label)

            # 主シルエット（最大コンポーネント）を単純化して refined_silhouette_mask とする。
            if largest_label > 0:
                largest_mask = labeled == largest_label

                contours = measure.find_contours(largest_mask.astype(float), 0.5)
                if contours:
                    contour = max(contours, key=lambda c: c.shape[0])
                    approx = measure.approximate_polygon(contour, tolerance=2.0)
                    if approx.shape[0] >= 3:
                        rr, cc = draw.polygon(approx[:, 0], approx[:, 1], largest_mask.shape)
                        refined = np.zeros_like(largest_mask, dtype=bool)
                        refined[rr, cc] = True
                        refined_silhouette_mask = refined

        # シルエット境界を抽出し、それ以外のエッジを内部線として扱う。
        if silhouette_mask is not None:
            boundary = segmentation.find_boundaries(silhouette_mask, mode="inner")
            internal_edges = np.logical_and(edges_bool, np.logical_not(boundary))

            # 内部線の連結成分のうち、ごく小さい点ノイズだけ除外しておく。
            labeled_internal = measure.label(internal_edges)
            max_label_internal = int(labeled_internal.max())
            if max_label_internal > 0:
                counts = np.bincount(labeled_internal.ravel())
                min_area = 2  # 2 ピクセル未満はノイズ扱い
                filtered = np.zeros_like(internal_edges, dtype=bool)
                for label_id in range(1, max_label_internal + 1):
                    if label_id >= len(counts):
                        break
                    area = int(counts[label_id])
                    if area < min_area:
                        continue
                    filtered |= labeled_internal == label_id
                internal_edges = filtered

        # 顔マスク・ランドマーク線マスクが渡されていれば、edges と同じ形にリサイズして保持する。
        h_e, w_e = edges_bool.shape
        if face_mask is not None:
            fm = face_mask.astype(float)
            if fm.shape != edges_bool.shape:
                fm_resized = transform.resize(
                    fm,
                    edges_bool.shape,
                    order=0,
                    anti_aliasing=False,
                    preserve_range=True,
                )
                face_mask_resized = fm_resized >= 0.5
            else:
                face_mask_resized = face_mask.astype(bool)

        if landmark_mask is not None:
            lm = landmark_mask.astype(float)
            if lm.shape != edges_bool.shape:
                lm_resized = transform.resize(
                    lm,
                    edges_bool.shape,
                    order=0,
                    anti_aliasing=False,
                    preserve_range=True,
                )
                landmark_mask_resized = lm_resized >= 0.5
            else:
                landmark_mask_resized = landmark_mask.astype(bool)

    except Exception:
        silhouette_mask = None
        silhouette_boxes = []
        internal_edges = None
        refined_silhouette_mask = None
        face_mask_resized = None
        landmark_mask_resized = None

    return FeatureSummary(
        centroid=(x_mean, y_mean),
        bounding_boxes=[(x_min, y_min, x_max, y_max)],
        silhouette_mask=silhouette_mask,
        silhouette_boxes=silhouette_boxes,
        internal_edges=internal_edges,
        refined_silhouette_mask=refined_silhouette_mask,
        face_mask=face_mask_resized,
        landmark_mask=landmark_mask_resized,
    )

