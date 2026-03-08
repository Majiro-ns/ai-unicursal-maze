"""
密度迷路 Phase 1/2: 濃度マップ → セルグリッド（壁リスト・セル輝度）。
Phase 2: テクスチャ情報（TextureType・グラジエント方向）を考慮した壁重み計算を追加。
Phase 2 Stage 4: build_cell_grid_with_edges() でエッジ強調を適用可能。
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

    numpy vectorization: np.add.reduceat で行・列方向を一括集計。
    Pythonループ版より 10〜20x 高速（400x600グリッドで 0.4s → 0.03s）。
    """
    h, w = gray.shape
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("grid_rows and grid_cols must be positive")

    # 境界インデックスを事前計算
    y_bnd = [int(r * h / grid_rows) for r in range(grid_rows + 1)]
    x_bnd = [int(c * w / grid_cols) for c in range(grid_cols + 1)]

    # np.add.reduceat で行方向に集計 → shape: (grid_rows, w)
    row_sums = np.add.reduceat(gray.astype(np.float64), y_bnd[:-1], axis=0)
    # 列方向に集計 → shape: (grid_rows, grid_cols)
    block_sums = np.add.reduceat(row_sums, x_bnd[:-1], axis=1)

    # 各ブロックのピクセル数で割って平均を計算
    row_sizes = np.array([y_bnd[r + 1] - y_bnd[r] for r in range(grid_rows)], dtype=np.float64)
    col_sizes = np.array([x_bnd[c + 1] - x_bnd[c] for c in range(grid_cols)], dtype=np.float64)
    block_areas = row_sizes[:, np.newaxis] * col_sizes[np.newaxis, :]
    block_areas = np.maximum(block_areas, 1.0)  # ゼロ除算ガード

    out = block_sums / block_areas
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

    # numpy vectorization: 右壁・下壁を一括生成してリスト化
    ids = np.arange(grid_rows * grid_cols, dtype=np.int32).reshape(grid_rows, grid_cols)

    # 右壁: cid < cid2 は常に成立（同一行で c < c+1）
    h_cid1 = ids[:, :-1].ravel()                  # (grid_rows * (grid_cols-1),)
    h_cid2 = ids[:, 1:].ravel()
    h_weight = ((lum[:, :-1] + lum[:, 1:]) / 2.0).ravel()

    # 下壁: cid < cid2 は常に成立（同一列で r < r+1）
    v_cid1 = ids[:-1, :].ravel()                  # ((grid_rows-1) * grid_cols,)
    v_cid2 = ids[1:, :].ravel()
    v_weight = ((lum[:-1, :] + lum[1:, :]) / 2.0).ravel()

    c1_all = np.concatenate([h_cid1, v_cid1])
    c2_all = np.concatenate([h_cid2, v_cid2])
    w_all  = np.concatenate([h_weight, v_weight])

    walls: List[Tuple[int, int, float]] = list(zip(c1_all.tolist(), c2_all.tolist(), w_all.tolist()))

    return CellGrid(rows=grid_rows, cols=grid_cols, luminance=lum, walls=walls)


def build_cell_grid_with_edges(
    gray: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    density_factor: float = 1.0,
    edge_weight: float = 0.0,
    edge_sigma: float = 1.0,
    edge_low_threshold: float = 0.05,
    edge_high_threshold: float = 0.20,
) -> CellGrid:
    """
    Phase 2 Stage 4: 輝度ベースの壁リストに Canny エッジ強調を適用したセルグリッドを返す。

    edge_weight=0.0 のとき build_cell_grid() と同じ結果。
    edge_weight>0 のとき、輪郭上の壁は weight が増大し（壁が残りやすくなり）、
    レンダリング後に元画像の輪郭が白い線として視認できる。

    Args:
        gray: (H, W) float グレースケール画像。
        grid_rows, grid_cols: セルグリッドサイズ。
        density_factor: 輝度差の強調度（build_cell_grid に渡す）。
        edge_weight: エッジ強調強度 0.0〜1.0。0=なし、1=最大。
        edge_sigma: Canny 前段ガウシアンの標準偏差。
        edge_low_threshold: Canny 二重閾値の下限（0〜1 相対値）。
        edge_high_threshold: Canny 二重閾値の上限（0〜1 相対値）。

    Returns:
        CellGrid（エッジ強調済み壁リストを含む）。
    """
    from .edge_enhancer import detect_edge_map, apply_edge_boost_to_walls

    base_grid = build_cell_grid(gray, grid_rows, grid_cols, density_factor=density_factor)

    if edge_weight <= 0.0:
        return base_grid

    edge_map = detect_edge_map(
        gray, grid_rows, grid_cols,
        sigma=edge_sigma,
        low_threshold=edge_low_threshold,
        high_threshold=edge_high_threshold,
    )
    boosted_walls = apply_edge_boost_to_walls(
        base_grid.walls, edge_map, grid_cols, edge_weight=edge_weight
    )
    return CellGrid(
        rows=grid_rows,
        cols=grid_cols,
        luminance=base_grid.luminance,
        walls=boosted_walls,
    )


def _spiral_angle(r: int, c: int, rows: int, cols: int) -> float:
    """
    セル (r, c) の中心からの角度（ラジアン、-π〜π）を返す。
    らせんバイアス計算用。中心セルでは 0.0 を返す。

    らせんバイアスの基本式（時計回り螺旋 / 01a §3.1 参照）:
      右壁（水平通路）: bias = sin²(θ) — 上辺・下辺（θ≈±π/2）で最大
      下壁（垂直通路）: bias = cos²(θ) — 左辺・右辺（θ≈0, π）で最大
    これにより、上下辺で水平廊下・左右辺で垂直廊下が優先され、
    同心方形の螺旋状パターン（正方形渦巻き）が形成される。
    """
    cr = (rows - 1) / 2.0
    cc = (cols - 1) / 2.0
    dr = r - cr
    dc = c - cc
    if abs(dr) < 1e-9 and abs(dc) < 1e-9:
        return 0.0  # 中心セル: 角度未定義 → バイアスなし（sin²=cos²=0/1 で均等）
    return float(np.arctan2(dr, dc))


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

                if tex_a == TextureType.DIRECTIONAL or tex_b == TextureType.DIRECTIONAL:
                    # 画像グラジエント方向に沿う通路を優先
                    avg_ang = (ang_a + ang_b) / 2.0
                    # 水平通路: cos²(angle) — 水平グラジエントのとき優先
                    bias = float(np.cos(avg_ang) ** 2)
                    weight = base * (1.0 - bias_strength * bias)
                elif tex_a == TextureType.SPIRAL or tex_b == TextureType.SPIRAL:
                    # らせんバイアス（右壁）: sin²(θ) — θ≈±π/2（上辺・下辺）で最大
                    # 上辺・下辺にある壁を優先的に除去し、水平廊下を形成する
                    ang = _spiral_angle(r, c, grid_rows, grid_cols)
                    bias = float(np.sin(ang) ** 2)
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
                    # 垂直通路: sin²(angle) — 垂直グラジエントのとき優先
                    bias = float(np.sin(avg_ang) ** 2)
                    weight = base * (1.0 - bias_strength * bias)
                elif tex_a == TextureType.SPIRAL or tex_b == TextureType.SPIRAL:
                    # らせんバイアス（下壁）: cos²(θ) — θ≈0,π（左辺・右辺）で最大
                    # 左辺・右辺にある壁を優先的に除去し、垂直廊下を形成する
                    ang = _spiral_angle(r, c, grid_rows, grid_cols)
                    bias = float(np.cos(ang) ** 2)
                    weight = base * (1.0 - bias_strength * bias)
                else:
                    weight = base

                walls.append((min(cid, cid2), max(cid, cid2), float(np.clip(weight, 0.0, 1.0))))

    return CellGrid(rows=grid_rows, cols=grid_cols, luminance=lum, walls=walls)
