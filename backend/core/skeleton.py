from __future__ import annotations

"""
Skeleton helpers.
V2 系アルゴリズム用の「エッジ → スケルトン」変換と簡易ノイズ除去。
"""

import numpy as np

try:
    from skimage.morphology import skeletonize as _skimage_skeletonize  # type: ignore[import]
except Exception:
    _skimage_skeletonize = None  # type: ignore[assignment]

try:
    from skimage.morphology import remove_small_objects as _remove_small_objects  # type: ignore[import]
except Exception:
    _remove_small_objects = None  # type: ignore[assignment]

try:
    from skimage.morphology import opening as _opening  # type: ignore[import]
    from skimage.morphology import disk as _disk  # type: ignore[import]
except Exception:
    _opening = None  # type: ignore[assignment]
    _disk = None  # type: ignore[assignment]


def _remove_isolated_pixels(mask: np.ndarray) -> np.ndarray:
    """
    8 近傍で True が 1 つだけのピクセルを落とす簡易ノイズ除去。
    """
    h, w = mask.shape
    out = mask.copy()
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            y0 = max(0, y - 1)
            y1 = min(h, y + 2)
            x0 = max(0, x - 1)
            x1 = min(w, x + 2)
            neighborhood = mask[y0:y1, x0:x1]
            neighbors_true = int(neighborhood.sum()) - 1
            if neighbors_true <= 0:
                out[y, x] = False
    return out


def edges_to_skeleton(
    edges: np.ndarray,
    min_edge_size: int = 8,
    opening_size: int = 0,
) -> np.ndarray:
    """
    エッジマップからスケルトン画像を生成する簡易実装。
    - 入力: shape=(H, W) の 2 次元配列。bool または 0/1 数値を想定
    - 出力: shape=(H, W) の 2 次元 bool 配列
      - opening_size > 0 のとき morphological opening で細かい突起を除去（T-5 安定化）
      - skeletonize 前に min_edge_size 未満の小コンポーネントをノイズ除去
      - skeletonize 後に孤立ピクセルを 2 回除去（残り滲み対策）
      - スケルトンが元の 20% 未満しか残らない場合は元のエッジにフォールバック
      - 利用できない場合やエラー時は元の edges を bool に正規化して返す
    """
    if edges.ndim != 2:
        raise ValueError("edges は 2 次元配列である必要があります")

    edges_bool = edges.astype(bool)
    skeleton = edges_bool

    if _skimage_skeletonize is not None:
        try:
            pre_cleaned = edges_bool
            # Optional morphological opening: reduce thin protrusions and small blobs (T-5)
            if opening_size > 0 and _opening is not None and _disk is not None and edges_bool.any():
                try:
                    r = max(1, min(opening_size, 5))
                    opened = np.asarray(_opening(edges_bool, _disk(r)), dtype=bool)
                    pre_cleaned = opened if opened.any() else edges_bool
                except Exception:
                    pre_cleaned = edges_bool

            if _remove_small_objects is not None and pre_cleaned.any():
                try:
                    # max_size=N removes objects with size ≤ N (equivalent to old min_size=N+1 which removed size < N+1)
                    # To preserve old min_size=min_edge_size behaviour (removes size < min_edge_size):
                    # use max_size=min_edge_size-1 (removes size ≤ min_edge_size-1)
                    _max_size = max(0, min_edge_size - 1)
                    cleaned = np.asarray(
                        _remove_small_objects(pre_cleaned, max_size=_max_size, connectivity=2),
                        dtype=bool,
                    )
                    pre_cleaned = cleaned if cleaned.any() else pre_cleaned
                except Exception:
                    pass

            candidate = _skimage_skeletonize(pre_cleaned)
            candidate = np.asarray(candidate, dtype=bool)

            if candidate.sum() >= edges_bool.sum() * 0.2:
                # Post-clean: remove isolated pixels (two passes to catch newly isolated pixels)
                skeleton = _remove_isolated_pixels(candidate)
                skeleton = _remove_isolated_pixels(skeleton)
        except Exception:
            skeleton = edges_bool

    return skeleton


def remove_small_skeleton_components(mask: np.ndarray, min_size: int = 5) -> np.ndarray:
    """
    スケルトン画像から min_size 未満の連結コンポーネントを除去する。
    - 入力: shape=(H, W) の 2 次元配列
    - 出力: shape=(H, W) の 2 次元 bool 配列
    - skimage が利用不可または例外時は元の mask を返す（safe fallback）
    """
    if mask.ndim != 2:
        raise ValueError("mask は 2 次元配列である必要があります")

    mask_bool = mask.astype(bool)
    if _remove_small_objects is None:
        return mask_bool

    try:
        # max_size=N removes objects with size ≤ N (equivalent to old min_size=N+1)
        # To preserve old min_size=min_size behaviour (removes size < min_size):
        # use max_size=min_size-1 (removes size ≤ min_size-1)
        _max_size = max(0, min_size - 1)
        result = np.asarray(
            _remove_small_objects(mask_bool, max_size=_max_size, connectivity=2),
            dtype=bool,
        )
        return result
    except Exception:
        return mask_bool


def stabilize_skeleton(
    edges: np.ndarray,
    *,
    min_edge_size: int = 8,
    spur_length: int = 4,
    min_component_size: int = 5,
    opening_size: int = 0,
) -> np.ndarray:
    """
    スケルトン線を安定化する一括パイプライン。

    パイプライン:
        edges
        → edges_to_skeleton(min_edge_size, opening_size)   # ノイズ除去 + 細線化
        → remove_short_spurs(spur_length)                 # 短い棘除去
        → remove_small_skeleton_components(min_component_size)  # 小コンポーネント除去
        → result (bool 2D array)

    opening_size > 0 のとき、細線化前に morphological opening で細かい突起を除去（T-5）。
    """
    skeleton = edges_to_skeleton(
        edges,
        min_edge_size=min_edge_size,
        opening_size=opening_size,
    )
    skeleton = remove_short_spurs(skeleton, max_length=spur_length)
    skeleton = remove_small_skeleton_components(skeleton, min_size=min_component_size)
    return skeleton


def remove_short_spurs(mask: np.ndarray, max_length: int = 4) -> np.ndarray:
    """
    短いスパー（分岐点から伸びる行き止まりの枝）を削ってスケルトンを滑らかにする。
    分岐点(deg>=3)に接続し、長さが max_length 以下の枝だけを対象にする。
    """
    if mask.ndim != 2:
        raise ValueError("mask は 2 次元配列である必要があります")

    work = mask.astype(bool).copy()
    h, w = work.shape

    def neighbors(y: int, x: int) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                ny = y + dy
                nx = x + dx
                if 0 <= ny < h and 0 <= nx < w and work[ny, nx]:
                    out.append((ny, nx))
        return out

    changed = True
    while changed:
        changed = False
        ys, xs = np.where(work)
        endpoints: list[tuple[int, int]] = []
        for y, x in zip(ys, xs):
            if len(neighbors(y, x)) == 1:
                endpoints.append((y, x))

        for y, x in endpoints:
            path: list[tuple[int, int]] = []
            prev: tuple[int, int] | None = None
            cur = (y, x)
            length = 0
            while True:
                path.append(cur)
                nbs = neighbors(cur[0], cur[1])
                if prev is not None:
                    nbs = [p for p in nbs if p != prev]

                if not nbs:
                    break

                if len(nbs) >= 2:
                    if len(path) - 1 <= max_length:
                        for py, px in path:
                            work[py, px] = False
                        changed = True
                    break

                prev = cur
                cur = nbs[0]
                length += 1
                if length > max_length:
                    break

    return work
