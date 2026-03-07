"""
密度迷路 Phase 1/2: 濃度マップ → セルグリッド（壁リスト・セル輝度）。
Phase 2: テクスチャ情報（TextureType・グラジエント方向）を考慮した壁重み計算を追加。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class CellGrid:
    """矩形グリッド。セル id = row * cols + col。"""
    rows: int
    cols: int
    luminance: np.ndarray  # (rows, cols) float 0〜1。セルごとの平均輝度
    walls: List[Tuple[int, int, float]]  # (cell_a, cell_b, weight)。cell_a < cell_b。weight 小→先に除去

    @property
    def num_cells(self) -> int:
        return self.rows * self.cols

    def cell_id(self, row: int, col: int) -> int:
        return row * self.cols + col

    def cell_rc(self, cid: int) -> Tuple[int, int]:
        return cid // self.cols, cid % self.cols


def build_density_map(gray: np.ndarray, grid_rows: int, grid_cols: int) -> np.ndarray:
    """
    グレースケール画像を grid_rows x grid_cols に分割し、各セルの平均輝度を返す。
    戻り値: (grid_rows, grid_cols) float。暗い＝低い値、明るい＝高い値。
    """
    h, w = gray.shape
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("grid_rows and grid_cols must be positive")
    # 各セルに対応するピクセル領域の平均を取る
    out = np.zeros((grid_rows, grid_cols), dtype=np.float64)
    for r in range(grid_rows):
        for c in range(grid_cols):
            y0 = int(r * h / grid_rows)
            y1 = int((r + 1) * h / grid_rows)
            x0 = int(c * w / grid_cols)
            x1 = int((c + 1) * w / grid_cols)
            y1 = min(y1, h)
            x1 = min(x1, w)
            if y1 > y0 and x1 > x0:
                out[r, c] = float(np.mean(gray[y0:y1, x0:x1]))
            else:
                out[r, c] = 0.5
    return np.clip(out, 0.0, 1.0)


def build_cell_grid(
    gray: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    density_factor: float = 1.0,
) -> CellGrid:
    """
    濃度マップを元にセルグリッドと壁リストを構築。
    壁の weight = 平均輝度（明るいセルほど weight 大＝後から除去＝道が粗い）。
    density_factor は輝度差の強調度（未使用でも可）。
    """
    lum = build_density_map(gray, grid_rows, grid_cols)
    walls: List[Tuple[int, int, float]] = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            cid = r * grid_cols + c
            w = lum[r, c]
            # 右隣
            if c + 1 < grid_cols:
                cid2 = r * grid_cols + (c + 1)
                w2 = lum[r, c + 1]
                weight = (w + w2) / 2.0
                walls.append((min(cid, cid2), max(cid, cid2), weight))
            # 下隣
            if r + 1 < grid_rows:
                cid2 = (r + 1) * grid_cols + c
                w2 = lum[r + 1, c]
                weight = (w + w2) / 2.0
                walls.append((min(cid, cid2), max(cid, cid2), weight))

    return CellGrid(rows=grid_rows, cols=grid_cols, luminance=lum, walls=walls)


def build_cell_grid_with_texture(
    gray: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    cell_textures: np.ndarray,
    gradient_angles: Optional[np.ndarray] = None,
    density_factor: float = 1.0,
    bias_strength: float = 0.5,
) -> CellGrid:
    """
    Phase 2: テクスチャ種別とグラジエント方向を考慮した壁重み計算。

    DIRECTIONAL テクスチャ:
      - グラジエント方向に沿う通路（壁）を優先して除去する
      - 右壁（水平通路）: bias = cos²(gradient_angle) → 水平グラジエントで優先
      - 下壁（垂直通路）: bias = sin²(gradient_angle) → 垂直グラジエントで優先
      - weight = base_weight * (1 - bias_strength * bias)
    RANDOM テクスチャ:
      - Phase1 と同じ（輝度ベース重み）

    bias_strength: 0.0 = バイアスなし（Phase1相当）、1.0 = 最大バイアス
    """
    from .texture import TextureType

    lum = build_density_map(gray, grid_rows, grid_cols)

    if gradient_angles is None:
        gradient_angles = np.zeros((grid_rows, grid_cols), dtype=np.float64)

    walls: List[Tuple[int, int, float]] = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            cid = r * grid_cols + c
            lum_a = lum[r, c]
            tex_a = cell_textures[r, c]
            ang_a = gradient_angles[r, c]

            # 右壁（水平通路 = セル(r,c)と(r,c+1)の間）
            if c + 1 < grid_cols:
                cid2 = r * grid_cols + (c + 1)
                lum_b = lum[r, c + 1]
                tex_b = cell_textures[r, c + 1]
                ang_b = gradient_angles[r, c + 1]
                base = (lum_a + lum_b) / 2.0

                # 両セルとも DIRECTIONAL のときバイアスを適用
                if tex_a == TextureType.DIRECTIONAL or tex_b == TextureType.DIRECTIONAL:
                    avg_ang = (ang_a + ang_b) / 2.0
                    # 水平通路: cos²(angle) が大きい → 水平グラジエントのとき優先
                    bias = float(np.cos(avg_ang) ** 2)
                    weight = base * (1.0 - bias_strength * bias)
                else:
                    weight = base

                walls.append((min(cid, cid2), max(cid, cid2), float(np.clip(weight, 0.0, 1.0))))

            # 下壁（垂直通路 = セル(r,c)と(r+1,c)の間）
            if r + 1 < grid_rows:
                cid2 = (r + 1) * grid_cols + c
                lum_b = lum[r + 1, c]
                tex_b = cell_textures[r + 1, c]
                ang_b = gradient_angles[r + 1, c]
                base = (lum_a + lum_b) / 2.0

                if tex_a == TextureType.DIRECTIONAL or tex_b == TextureType.DIRECTIONAL:
                    avg_ang = (ang_a + ang_b) / 2.0
                    # 垂直通路: sin²(angle) が大きい → 垂直グラジエントのとき優先
                    bias = float(np.sin(avg_ang) ** 2)
                    weight = base * (1.0 - bias_strength * bias)
                else:
                    weight = base

                walls.append((min(cid, cid2), max(cid, cid2), float(np.clip(weight, 0.0, 1.0))))

    return CellGrid(rows=grid_rows, cols=grid_cols, luminance=lum, walls=walls)
