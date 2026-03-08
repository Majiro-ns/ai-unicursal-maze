"""
密度迷路 Phase 2: Stage 4 — Canny エッジ検出 → 壁保持強度マップ。

輪郭上のセルはスパニングツリー構築時に壁の weight を増大させる。
weight が高い壁は Kruskal 法で後から（または除去されずに）処理されるため、
結果として輪郭部分に白い壁の線が残り、元画像の輪郭がはっきり見える。

使い方::

    from backend.core.density.edge_enhancer import detect_edge_map, apply_edge_boost_to_walls

    edge_map = detect_edge_map(gray, grid_rows, grid_cols, edge_weight=0.6)
    boosted_walls = apply_edge_boost_to_walls(walls, edge_map, grid_cols, edge_weight=0.6)
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def detect_edge_map(
    gray: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    *,
    sigma: float = 1.0,
    low_threshold: float = 0.05,
    high_threshold: float = 0.20,
) -> np.ndarray:
    """
    Canny エッジ検出でグレースケール画像の輪郭を抽出し、
    セルグリッドの各セルにおけるエッジ強度を返す。

    Args:
        gray: (H, W) float 0.0〜1.0 のグレースケール画像。
        grid_rows: セルグリッドの行数。
        grid_cols: セルグリッドの列数。
        sigma: ガウシアンぼかしの標準偏差。大きいほどノイズ耐性が上がる。
        low_threshold: Canny 二重閾値の下限（0.0〜1.0 相対値）。
        high_threshold: Canny 二重閾値の上限（0.0〜1.0 相対値）。

    Returns:
        (grid_rows, grid_cols) float 配列。0.0=エッジなし、1.0=強いエッジ。
        セル内のエッジピクセル比率を最大値で正規化した値。
    """
    from skimage.feature import canny
    from skimage.filters import gaussian

    if gray.ndim != 2:
        raise ValueError(f"gray must be 2D, got shape {gray.shape}")
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("grid_rows and grid_cols must be positive")

    # ガウシアンぼかしでノイズ低減（sigma=0 はスキップ）
    if sigma > 0:
        smoothed = gaussian(gray, sigma=sigma)
    else:
        smoothed = gray.copy()
    smoothed = np.clip(smoothed, 0.0, 1.0)

    # Canny エッジ検出（戻り値は bool 配列）
    edge_binary: np.ndarray = canny(
        smoothed,
        sigma=0,  # ぼかしは上で済ませた
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    ).astype(np.float64)

    # セルグリッドへ集約: 各セル内のエッジピクセル比率
    h, w = gray.shape
    edge_map = np.zeros((grid_rows, grid_cols), dtype=np.float64)
    for r in range(grid_rows):
        for c in range(grid_cols):
            y0 = int(r * h / grid_rows)
            y1 = int((r + 1) * h / grid_rows)
            x0 = int(c * w / grid_cols)
            x1 = int((c + 1) * w / grid_cols)
            y1 = min(y1, h)
            x1 = min(x1, w)
            if y1 > y0 and x1 > x0:
                edge_map[r, c] = float(np.mean(edge_binary[y0:y1, x0:x1]))

    # 最大値で正規化（全セルにエッジがない場合は 0 のまま）
    max_val = float(edge_map.max())
    if max_val > 1e-9:
        edge_map = edge_map / max_val

    return np.clip(edge_map, 0.0, 1.0)


def extract_edge_waypoints(
    edge_map: np.ndarray,
    grid_cols: int,
    threshold: float = 0.3,
) -> set:
    """
    Extract cell IDs where edge strength exceeds threshold.

    These waypoints are used by connect_through_bright() in path-first mode
    as preferred transit points (K1).

    Args:
        edge_map: (grid_rows, grid_cols) float edge strength map from detect_edge_map().
        grid_cols: Number of grid columns (for cell_id = r * grid_cols + c).
        threshold: Edge strength threshold (0-1). Cells above this are waypoints.

    Returns:
        Set of cell IDs with strong edges.
    """
    waypoints: set = set()
    rows, cols = edge_map.shape
    for r in range(rows):
        for c in range(cols):
            if edge_map[r, c] > threshold:
                waypoints.add(r * grid_cols + c)
    return waypoints


def apply_edge_boost_to_walls(
    walls: List[Tuple[int, int, float]],
    edge_map: np.ndarray,
    grid_cols: int,
    edge_weight: float = 0.5,
) -> List[Tuple[int, int, float]]:
    """
    エッジマップに基づいて壁の weight を増大させる。

    輪郭上の壁（edge_strength が高い壁）は weight が大きくなるため、
    Kruskal 法で後から処理される（= 壁が残りやすい）。
    結果として輪郭部分に白い壁の線が残り、元画像の輪郭として視認できる。

    weight 更新式::

        boosted = w + edge_weight * edge_strength * (1.0 - w)

    - edge_weight = 0.0: 変化なし（Phase1 相当）
    - edge_weight = 1.0: エッジ上の壁を最大限保持（weight → 1.0 に近づく）

    Args:
        walls: (cell_a, cell_b, weight) のリスト。cell_a < cell_b。
        edge_map: (grid_rows, grid_cols) float エッジ強度マップ。
        grid_cols: セルグリッドの列数。
        edge_weight: エッジ効果の強度 0.0〜1.0。

    Returns:
        weight を更新した新しい壁リスト。
    """
    if edge_weight <= 0.0:
        return list(walls)

    edge_weight = float(np.clip(edge_weight, 0.0, 1.0))
    grid_rows = edge_map.shape[0]

    new_walls: List[Tuple[int, int, float]] = []
    for c1, c2, w in walls:
        r1, col1 = c1 // grid_cols, c1 % grid_cols
        r2, col2 = c2 // grid_cols, c2 % grid_cols

        # 範囲外チェック（堅牢性のため）
        if (0 <= r1 < grid_rows and 0 <= col1 < grid_cols and
                0 <= r2 < grid_rows and 0 <= col2 < grid_cols):
            e1 = edge_map[r1, col1]
            e2 = edge_map[r2, col2]
            edge_strength = (e1 + e2) / 2.0
        else:
            edge_strength = 0.0

        # エッジが強いほど weight を 1.0 方向へ押し上げる
        boosted = float(w) + edge_weight * edge_strength * (1.0 - float(w))
        new_walls.append((c1, c2, float(np.clip(boosted, 0.0, 1.0))))

    return new_walls
