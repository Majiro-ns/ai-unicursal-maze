"""
密度迷路 Phase 1/2: 領域分割。
Phase 1: 単一領域（全ピクセルを同一ラベル）。
Phase 2: K-means で輝度クラスタリング（多領域セグメンテーション）。
"""
from __future__ import annotations

import numpy as np


def segment_single_region(gray: np.ndarray) -> np.ndarray:
    """
    Phase 1: 全ピクセルを同一領域とする。戻り値は (H, W) で全要素 1。
    """
    return np.ones(gray.shape, dtype=np.int32)


def segment_by_luminance(
    gray: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
) -> np.ndarray:
    """
    Phase 2: K-means で輝度を n_clusters クラスに分割。
    戻り値: (H, W) int ラベルマップ。0=最暗（高密度）〜 n_clusters-1=最明（低密度）。
    ラベルは平均輝度の昇順に再整理する（0が最も暗いクラスタ）。
    """
    from sklearn.cluster import KMeans

    n_clusters = max(1, n_clusters)
    h, w = gray.shape
    X = gray.reshape(-1, 1).astype(np.float32)

    if n_clusters == 1:
        return np.zeros((h, w), dtype=np.int32)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=5)
    raw_labels = km.fit_predict(X)

    # クラスタ中心の輝度昇順でラベルを再マッピング（0=最暗）
    centers = km.cluster_centers_.flatten()
    order = np.argsort(centers)   # order[i] = i番目に明るいクラスタの元ラベル
    remap = np.empty(n_clusters, dtype=np.int32)
    for new_label, old_label in enumerate(order):
        remap[old_label] = new_label

    return remap[raw_labels].reshape(h, w).astype(np.int32)
