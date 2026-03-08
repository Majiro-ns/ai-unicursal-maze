# -*- coding: utf-8 -*-
"""
密度迷路 Phase 2 Stage 4: Canny エッジ強調テスト。

対象モジュール:
  backend/core/density/edge_enhancer.py
  backend/core/density/grid_builder.py (build_cell_grid_with_edges)
  backend/core/density/__init__.py (edge_weight パラメータ)
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from backend.core.density.edge_enhancer import detect_edge_map, apply_edge_boost_to_walls
from backend.core.density.grid_builder import build_cell_grid, build_cell_grid_with_edges
from backend.core.density import generate_density_maze


# ---- ヘルパー ----

def _solid_image(w: int = 64, h: int = 64, value: int = 128) -> Image.Image:
    arr = np.full((h, w), value, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def _gradient_image(w: int = 64, h: int = 64) -> Image.Image:
    """左が暗(0)、右が明(255)の水平グラデーション。"""
    col = np.linspace(0, 255, w, dtype=np.uint8)
    arr = np.tile(col, (h, 1))
    return Image.fromarray(arr, mode="L")


def _checkerboard_image(w: int = 64, h: int = 64, cell: int = 8) -> Image.Image:
    """白黒チェッカーボード（エッジが豊富な画像）。"""
    arr = np.zeros((h, w), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            if ((r // cell) + (c // cell)) % 2 == 0:
                arr[r, c] = 255
    return Image.fromarray(arr, mode="L")


def _rgb_image(w: int = 64, h: int = 64) -> Image.Image:
    arr = np.full((h, w, 3), 128, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _gray_array(w: int = 64, h: int = 64, value: float = 0.5) -> np.ndarray:
    return np.full((h, w), value, dtype=np.float64)


def _checkerboard_array(w: int = 64, h: int = 64, cell: int = 8) -> np.ndarray:
    arr = np.zeros((h, w), dtype=np.float64)
    for r in range(h):
        for c in range(w):
            if ((r // cell) + (c // cell)) % 2 == 0:
                arr[r, c] = 1.0
    return arr


# ========================
# detect_edge_map テスト
# ========================

class TestDetectEdgeMap:

    def test_returns_correct_shape(self):
        gray = _gray_array()
        result = detect_edge_map(gray, grid_rows=8, grid_cols=8)
        assert result.shape == (8, 8)

    def test_returns_float_in_0_1(self):
        gray = _checkerboard_array()
        result = detect_edge_map(gray, grid_rows=8, grid_cols=8)
        assert result.dtype == np.float64
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_solid_image_has_no_edges(self):
        """均一画像はエッジなし → 全セル 0.0。"""
        gray = _gray_array(value=0.5)
        result = detect_edge_map(gray, grid_rows=8, grid_cols=8)
        assert float(result.max()) == pytest.approx(0.0, abs=1e-6)

    def test_checkerboard_has_strong_edges(self):
        """チェッカーボードはエッジが豊富 → 最大値 1.0。"""
        gray = _checkerboard_array()
        result = detect_edge_map(gray, grid_rows=8, grid_cols=8)
        assert float(result.max()) == pytest.approx(1.0, abs=1e-6)

    def test_non_uniform_edge_distribution(self):
        """エッジ強度が不均一（一部セルのみ高い）。"""
        gray = _checkerboard_array()
        result = detect_edge_map(gray, grid_rows=8, grid_cols=8)
        # 全セル同じにはならない
        assert float(result.std()) > 0.0

    def test_different_grid_sizes(self):
        gray = _checkerboard_array()
        for rows, cols in [(4, 4), (8, 8), (16, 16)]:
            result = detect_edge_map(gray, grid_rows=rows, grid_cols=cols)
            assert result.shape == (rows, cols)

    def test_invalid_ndim_raises(self):
        gray_3d = np.zeros((64, 64, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="2D"):
            detect_edge_map(gray_3d, grid_rows=8, grid_cols=8)

    def test_invalid_grid_size_raises(self):
        gray = _gray_array()
        with pytest.raises(ValueError):
            detect_edge_map(gray, grid_rows=0, grid_cols=8)

    def test_different_sigma_values(self):
        """sigma=0 でも動作する（ぼかしなし）。"""
        gray = _checkerboard_array()
        result = detect_edge_map(gray, grid_rows=8, grid_cols=8, sigma=0)
        assert result.shape == (8, 8)
        assert float(result.max()) <= 1.0

    def test_high_sigma_reduces_edges(self):
        """sigma が大きいほどエッジが弱くなる（ぼかしが強い）。"""
        gray = _checkerboard_array(cell=4)
        edge_sharp = detect_edge_map(gray, grid_rows=8, grid_cols=8, sigma=0.5)
        edge_blurred = detect_edge_map(gray, grid_rows=8, grid_cols=8, sigma=5.0)
        # ぼかしが強いほど検出エッジ数が減る（合計値が小さい）
        assert float(edge_sharp.sum()) >= float(edge_blurred.sum())

    def test_rectangular_grid(self):
        gray = _checkerboard_array(w=128, h=64)
        result = detect_edge_map(gray, grid_rows=4, grid_cols=8)
        assert result.shape == (4, 8)

    def test_single_cell_grid(self):
        gray = _checkerboard_array()
        result = detect_edge_map(gray, grid_rows=1, grid_cols=1)
        assert result.shape == (1, 1)


# ==============================
# apply_edge_boost_to_walls テスト
# ==============================

class TestApplyEdgeBoostToWalls:

    def _make_walls(self) -> list:
        """3×3 グリッドの壁サンプル（weight=0.5 均一）。"""
        return [
            (0, 1, 0.5),
            (1, 2, 0.5),
            (3, 4, 0.5),
            (4, 5, 0.5),
            (6, 7, 0.5),
            (7, 8, 0.5),
            (0, 3, 0.5),
            (1, 4, 0.5),
            (2, 5, 0.5),
            (3, 6, 0.5),
            (4, 7, 0.5),
            (5, 8, 0.5),
        ]

    def test_zero_edge_weight_no_change(self):
        """edge_weight=0 → 壁リストが変わらない。"""
        walls = self._make_walls()
        edge_map = np.ones((3, 3), dtype=np.float64)
        result = apply_edge_boost_to_walls(walls, edge_map, grid_cols=3, edge_weight=0.0)
        for (c1, c2, w), (r1, r2, rw) in zip(walls, result):
            assert w == pytest.approx(rw, abs=1e-9)

    def test_positive_edge_weight_increases_weights(self):
        """edge_weight>0 かつ edge_map=1.0 → weight が増大する。"""
        walls = [(0, 1, 0.5), (1, 2, 0.3)]
        edge_map = np.ones((1, 3), dtype=np.float64)
        result = apply_edge_boost_to_walls(walls, edge_map, grid_cols=3, edge_weight=0.8)
        for (_, _, orig), (_, _, new_w) in zip(walls, result):
            assert new_w >= orig

    def test_max_weight_clipped_to_1(self):
        """weight が 1.0 を超えない。"""
        walls = [(0, 1, 0.9)]
        edge_map = np.ones((1, 2), dtype=np.float64)
        result = apply_edge_boost_to_walls(walls, edge_map, grid_cols=2, edge_weight=1.0)
        assert result[0][2] <= 1.0

    def test_zero_edge_map_no_boost(self):
        """edge_map=0 → edge_weight に関係なく weight 変化なし。"""
        walls = [(0, 1, 0.5), (1, 2, 0.3)]
        edge_map = np.zeros((1, 3), dtype=np.float64)
        result = apply_edge_boost_to_walls(walls, edge_map, grid_cols=3, edge_weight=1.0)
        for (_, _, orig), (_, _, new_w) in zip(walls, result):
            assert new_w == pytest.approx(orig, abs=1e-9)

    def test_wall_count_preserved(self):
        """壁の数が変わらない。"""
        walls = self._make_walls()
        edge_map = np.random.rand(3, 3)
        result = apply_edge_boost_to_walls(walls, edge_map, grid_cols=3, edge_weight=0.5)
        assert len(result) == len(walls)

    def test_boost_formula_correct(self):
        """weight 更新式: boosted = w + edge_weight * edge_strength * (1 - w)。"""
        walls = [(0, 1, 0.4)]
        edge_map = np.array([[0.8, 0.6]], dtype=np.float64)  # shape (1, 2)
        edge_weight = 0.5
        edge_strength = (0.8 + 0.6) / 2.0
        expected = 0.4 + edge_weight * edge_strength * (1.0 - 0.4)
        result = apply_edge_boost_to_walls(walls, edge_map, grid_cols=2, edge_weight=edge_weight)
        assert result[0][2] == pytest.approx(expected, abs=1e-9)

    def test_empty_walls(self):
        """空の壁リスト → 空のまま。"""
        edge_map = np.ones((3, 3))
        result = apply_edge_boost_to_walls([], edge_map, grid_cols=3, edge_weight=0.5)
        assert result == []

    def test_output_weights_in_0_1(self):
        """全出力 weight が [0, 1] に収まる。"""
        walls = self._make_walls()
        edge_map = np.random.rand(3, 3)
        result = apply_edge_boost_to_walls(walls, edge_map, grid_cols=3, edge_weight=1.0)
        for _, _, w in result:
            assert 0.0 <= w <= 1.0

    def test_higher_edge_weight_gives_higher_wall_weights(self):
        """edge_weight が高いほど wall weight が高い（同じ edge_map）。"""
        walls = [(0, 1, 0.4), (1, 2, 0.6)]
        edge_map = np.ones((1, 3), dtype=np.float64) * 0.8
        result_low = apply_edge_boost_to_walls(walls, edge_map, grid_cols=3, edge_weight=0.2)
        result_high = apply_edge_boost_to_walls(walls, edge_map, grid_cols=3, edge_weight=0.8)
        for (_, _, wl), (_, _, wh) in zip(result_low, result_high):
            assert wh >= wl


# ==============================
# build_cell_grid_with_edges テスト
# ==============================

class TestBuildCellGridWithEdges:

    def test_zero_edge_weight_matches_base(self):
        """edge_weight=0 → build_cell_grid と同じ壁 weight。"""
        from backend.core.density.preprocess import preprocess_image
        img = _checkerboard_image()
        gray = preprocess_image(img, max_side=64)
        base = build_cell_grid(gray, 8, 8)
        with_edges = build_cell_grid_with_edges(gray, 8, 8, edge_weight=0.0)
        # 壁数が同じ
        assert len(with_edges.walls) == len(base.walls)
        # weight が同じ
        for (c1, c2, w1), (e1, e2, w2) in zip(
            sorted(base.walls), sorted(with_edges.walls)
        ):
            assert w1 == pytest.approx(w2, abs=1e-9)

    def test_positive_edge_weight_changes_weights(self):
        """edge_weight>0 → チェッカーボード画像でエッジ上の壁が増大。"""
        from backend.core.density.preprocess import preprocess_image
        img = _checkerboard_image()
        gray = preprocess_image(img, max_side=64)
        base = build_cell_grid(gray, 8, 8)
        with_edges = build_cell_grid_with_edges(gray, 8, 8, edge_weight=0.8)
        base_sum = sum(w for _, _, w in base.walls)
        edge_sum = sum(w for _, _, w in with_edges.walls)
        # エッジ強調で壁 weight 合計が増大
        assert edge_sum >= base_sum

    def test_wall_count_preserved(self):
        from backend.core.density.preprocess import preprocess_image
        img = _gradient_image()
        gray = preprocess_image(img, max_side=64)
        result = build_cell_grid_with_edges(gray, 8, 8, edge_weight=0.5)
        # 8×8 グリッドの壁数 = (8-1)*8*2 = 112
        base = build_cell_grid(gray, 8, 8)
        assert len(result.walls) == len(base.walls)

    def test_returns_cell_grid(self):
        from backend.core.density.preprocess import preprocess_image
        from backend.core.density.grid_builder import CellGrid
        img = _solid_image()
        gray = preprocess_image(img, max_side=64)
        result = build_cell_grid_with_edges(gray, 8, 8, edge_weight=0.5)
        assert isinstance(result, CellGrid)
        assert result.rows == 8
        assert result.cols == 8

    def test_edge_weight_1_maximizes_wall_retention(self):
        """edge_weight=1.0 は edge_weight=0.5 より壁 weight が高い（同じ画像）。"""
        from backend.core.density.preprocess import preprocess_image
        img = _checkerboard_image()
        gray = preprocess_image(img, max_side=64)
        result_05 = build_cell_grid_with_edges(gray, 8, 8, edge_weight=0.5)
        result_10 = build_cell_grid_with_edges(gray, 8, 8, edge_weight=1.0)
        sum_05 = sum(w for _, _, w in result_05.walls)
        sum_10 = sum(w for _, _, w in result_10.walls)
        assert sum_10 >= sum_05


# ==============================
# generate_density_maze 統合テスト
# ==============================

class TestGenerateDensityMazeWithEdge:

    def test_edge_weight_0_backward_compat(self):
        """edge_weight=0（デフォルト）は Phase1 と同じ結果。"""
        img = _solid_image()
        result = generate_density_maze(img, grid_size=5, max_side=64, edge_weight=0.0)
        assert result.entrance >= 0
        assert result.exit_cell >= 0
        assert len(result.solution_path) >= 1

    def test_edge_weight_positive_produces_valid_maze(self):
        """edge_weight>0 でも有効な迷路（入口・出口・解経路）が生成される。"""
        img = _checkerboard_image()
        result = generate_density_maze(
            img, grid_size=6, max_side=64, edge_weight=0.7
        )
        assert result.entrance >= 0
        assert result.exit_cell >= 0
        path = result.solution_path
        assert len(path) >= 1
        assert path[0] == result.entrance
        assert path[-1] == result.exit_cell

    def test_edge_weight_solution_path_no_duplicates(self):
        """エッジ強調あり迷路の解経路に重複セルなし。"""
        img = _checkerboard_image()
        result = generate_density_maze(
            img, grid_size=5, max_side=64, edge_weight=0.6
        )
        path = result.solution_path
        assert len(path) == len(set(path))

    def test_edge_weight_rgba_image(self):
        """RGBA 入力でも動作する。"""
        arr = np.full((64, 64, 4), 128, dtype=np.uint8)
        arr[:, :, 3] = 255
        img = Image.fromarray(arr, mode="RGBA")
        result = generate_density_maze(img, grid_size=5, max_side=64, edge_weight=0.5)
        assert result.entrance >= 0

    def test_edge_weight_rgb_image(self):
        """RGB 入力でも動作する。"""
        img = _rgb_image()
        result = generate_density_maze(img, grid_size=5, max_side=64, edge_weight=0.5)
        assert result.entrance >= 0

    def test_edge_connectivity_preserved(self):
        """エッジ強調を使っても全セルが連結（spanning tree）。"""
        from collections import deque
        from backend.core.density.preprocess import preprocess_image
        from backend.core.density.maze_builder import build_spanning_tree
        img = _checkerboard_image()
        result = generate_density_maze(img, grid_size=6, max_side=64, edge_weight=0.8)
        gray = preprocess_image(img, max_side=64)
        from backend.core.density.grid_builder import build_cell_grid_with_edges
        grid = build_cell_grid_with_edges(
            gray, result.grid_rows, result.grid_cols, edge_weight=0.8
        )
        adj = build_spanning_tree(grid)
        n = grid.num_cells
        visited = set()
        q = deque([0])
        visited.add(0)
        while q:
            cell = q.popleft()
            for nb in adj[cell]:
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        assert len(visited) == n

    def test_png_bytes_generated(self):
        """PNG バイトが生成されること。"""
        img = _checkerboard_image()
        result = generate_density_maze(img, grid_size=5, max_side=64, edge_weight=0.5)
        assert len(result.png_bytes) > 0

    def test_svg_generated(self):
        """SVG 文字列が生成されること。"""
        img = _checkerboard_image()
        result = generate_density_maze(img, grid_size=5, max_side=64, edge_weight=0.5)
        assert "<svg" in result.svg

    def test_various_edge_weights(self):
        """edge_weight 0.0〜1.0 の範囲で全てクラッシュしない。"""
        img = _checkerboard_image()
        for ew in [0.0, 0.2, 0.5, 0.8, 1.0]:
            result = generate_density_maze(img, grid_size=4, max_side=32, edge_weight=ew)
            assert result.entrance >= 0, f"edge_weight={ew} failed"

    def test_edge_params_custom(self):
        """カスタム edge_sigma / threshold でも動作する。"""
        img = _checkerboard_image()
        result = generate_density_maze(
            img, grid_size=4, max_side=32,
            edge_weight=0.6,
            edge_sigma=2.0,
            edge_low_threshold=0.1,
            edge_high_threshold=0.3,
        )
        assert result.entrance >= 0

    def test_gradient_image_valid_maze(self):
        """グラデーション画像でエッジ強調あり迷路が生成できる。"""
        img = _gradient_image()
        result = generate_density_maze(img, grid_size=5, max_side=64, edge_weight=0.6)
        assert result.entrance >= 0
        assert len(result.solution_path) >= 1
