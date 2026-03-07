"""
密度迷路 Phase 2: テクスチャ割り当て。
- TextureType: RANDOM / DIRECTIONAL / SPIRAL
- assign_cell_textures(): セルごとのテクスチャ種別を多数決で決定
- compute_gradient_angles(): セルごとの輝度グラジエント方向（ラジアン）
- PRESET_FACE / PRESET_LANDSCAPE / PRESET_GENERIC: ラベル→テクスチャの標準マッピング
"""
from __future__ import annotations

import enum
from typing import Dict

import numpy as np

from .grid_builder import CellGrid


class TextureType(enum.Enum):
    """セルに適用するテクスチャパターン。"""
    RANDOM = "random"            # Phase1と同じ：輝度ベースのランダムな壁重み
    DIRECTIONAL = "directional"  # グラジエント方向に沿う通路を優先
    SPIRAL = "spiral"            # 直角グリッド上のらせん状パターン（正方形渦巻き）
    #   上辺・下辺で水平通路を優先、左辺・右辺で垂直通路を優先し、
    #   セルの中心からの角度に基づく壁バイアスで同心方形の螺旋を表現する。
    #   曲線は使用しない（01a §1.5 制約遵守）。


# ---- プリセット: ラベル(0=暗〜n-1=明) → テクスチャ ----

PRESET_FACE: Dict[int, TextureType] = {
    0: TextureType.DIRECTIONAL,  # 暗い = 髪 → 方向性（流れのある線）
    1: TextureType.SPIRAL,       # やや暗い = 影・輪郭 → らせん（輪郭を囲む構造）
    2: TextureType.RANDOM,       # やや明るい = 肌 → ランダム（密度で表現）
    3: TextureType.DIRECTIONAL,  # 明るい = 背景・ハイライト → 方向性
}

PRESET_LANDSCAPE: Dict[int, TextureType] = {
    0: TextureType.DIRECTIONAL,  # 暗い = 木・地面 → 方向性
    1: TextureType.DIRECTIONAL,  # やや暗 = 草木 → 方向性
    2: TextureType.SPIRAL,       # やや明 = 建物・岩 → らせん（構造物の輪郭・囲み効果）
    3: TextureType.DIRECTIONAL,  # 明るい = 空 → 方向性（グラデーション）
}

PRESET_GENERIC: Dict[int, TextureType] = {
    0: TextureType.RANDOM,
    1: TextureType.RANDOM,
    2: TextureType.RANDOM,
    3: TextureType.RANDOM,
}


def assign_cell_textures(
    grid: CellGrid,
    label_map: np.ndarray,
    label_to_texture: Dict[int, TextureType],
) -> np.ndarray:
    """
    ピクセルラベルマップからセルごとのテクスチャ種別を決定（多数決）。
    戻り値: (grid_rows, grid_cols) object array of TextureType。
    """
    h, w = label_map.shape
    result = np.empty((grid.rows, grid.cols), dtype=object)

    for r in range(grid.rows):
        for c in range(grid.cols):
            y0 = int(r * h / grid.rows)
            y1 = int((r + 1) * h / grid.rows)
            x0 = int(c * w / grid.cols)
            x1 = int((c + 1) * w / grid.cols)
            y1, x1 = min(y1, h), min(x1, w)

            if y1 > y0 and x1 > x0:
                patch = label_map[y0:y1, x0:x1]
                unique, counts = np.unique(patch, return_counts=True)
                dominant_label = int(unique[np.argmax(counts)])
            else:
                dominant_label = 0

            result[r, c] = label_to_texture.get(dominant_label, TextureType.RANDOM)

    return result


def compute_gradient_angles(
    gray: np.ndarray,
    grid_rows: int,
    grid_cols: int,
) -> np.ndarray:
    """
    各セルの輝度グラジエント方向（ラジアン、-π〜π）を返す。
    Sobel フィルタで水平・垂直勾配を計算し arctan2 で方向を求める。
    戻り値: (grid_rows, grid_cols) float64。
    """
    from scipy.ndimage import sobel

    gx = sobel(gray, axis=1)  # 水平方向の勾配
    gy = sobel(gray, axis=0)  # 垂直方向の勾配

    h, w = gray.shape
    angles = np.zeros((grid_rows, grid_cols), dtype=np.float64)

    for r in range(grid_rows):
        for c in range(grid_cols):
            y0 = int(r * h / grid_rows)
            y1 = int((r + 1) * h / grid_rows)
            x0 = int(c * w / grid_cols)
            x1 = int((c + 1) * w / grid_cols)
            y1, x1 = min(y1, h), min(x1, w)
            if y1 > y0 and x1 > x0:
                mean_gx = float(np.mean(gx[y0:y1, x0:x1]))
                mean_gy = float(np.mean(gy[y0:y1, x0:x1]))
                angles[r, c] = np.arctan2(mean_gy, mean_gx)
            # else: 0.0 のまま

    return angles
