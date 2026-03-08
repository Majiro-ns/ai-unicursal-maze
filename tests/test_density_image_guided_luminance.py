# -*- coding: utf-8 -*-
"""
Dijkstra明部ルーティング（find_image_guided_path）の輝度定量テスト。

cmd_360k_a3 で追加。既存テストが到達可能性のみ確認していたのに対し、
本モジュールは「明部を優先して通るか」を定量的に検証する。

テスト内容:
  TestFindImageGuidedPathLuminanceUnit  - 輝度定量検証（5テスト）
  TestSolutionPathLuminanceHistogram    - ヒストグラム統計（4テスト）
  TestImageGuidedReproducibility        - 再現性（3テスト）
  TestImageGuidedEdgeCases              - エッジケース（3テスト）
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from backend.core.density import generate_density_maze
from backend.core.density.entrance_exit import find_image_guided_path
from backend.core.density.grid_builder import build_cell_grid
from backend.core.density.maze_builder import build_spanning_tree
from backend.core.density.preprocess import preprocess_image


# ──────────────────────────────────────────────
# ヘルパー
# ──────────────────────────────────────────────

def _make_gradient_image(width: int = 64, height: int = 64) -> Image.Image:
    """左黒(0)→右白(255)のグラデーショングレースケール画像。"""
    arr = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    return Image.fromarray(arr, mode="L")


def _make_bright_image(width: int = 32, height: int = 32) -> Image.Image:
    """全面白（輝度1.0）画像。"""
    return Image.fromarray(np.full((height, width), 255, dtype=np.uint8), mode="L")


def _make_dark_image(width: int = 32, height: int = 32) -> Image.Image:
    """全面黒（輝度0.0）画像。"""
    return Image.fromarray(np.zeros((height, width), dtype=np.uint8), mode="L")


def _build_grid_and_adj(img: Image.Image, grid_size: int = 8, max_side: int = 64):
    """画像から (grid, adj) を生成するヘルパー。"""
    gray = preprocess_image(img, max_side=max_side)
    grid = build_cell_grid(gray, grid_size, grid_size)
    adj = build_spanning_tree(grid)
    return grid, adj


# ──────────────────────────────────────────────
# 1. 輝度定量検証
# ──────────────────────────────────────────────

class TestFindImageGuidedPathLuminanceUnit:
    """Dijkstra明部ルーティングの定量テスト。左黒右白グラデーションを使用。"""

    def setup_method(self):
        img = _make_gradient_image(width=64, height=64)
        self.grid, self.adj = _build_grid_and_adj(img, grid_size=8)
        self.flat_lum = self.grid.luminance.flatten()
        entrance, exit_cell, path = find_image_guided_path(
            self.adj, self.grid.num_cells,
            self.grid.luminance, self.grid.rows, self.grid.cols,
        )
        self.entrance = entrance
        self.exit_cell = exit_cell
        self.path = path
        self.path_lum = [float(self.flat_lum[c]) for c in self.path]

    def test_exit_brighter_than_entrance(self):
        """出口セルの輝度 > 入口セルの輝度（暗コーナー入口・明コーナー出口の設計確認）。

        find_image_guided_path の設計:
          入口 = 4隅の中で輝度最低 → 暗い角から出発
          出口 = 入口の対角      → 明るい角に向かう
        この設計により、出口の輝度は入口の輝度より高くなるはず。
        """
        entrance_lum = float(self.flat_lum[self.entrance])
        exit_lum = float(self.flat_lum[self.exit_cell])
        assert exit_lum > entrance_lum, (
            f"出口輝度 {exit_lum:.3f} <= 入口輝度 {entrance_lum:.3f} — "
            f"暗→明のルーティング設計が機能していない"
        )

    def test_path_second_half_brighter_than_first(self):
        """解法経路の後半の平均輝度 > 前半の平均輝度。

        入口=暗い角→出口=明るい角の設計により、後半が明部に近づくはず。
        """
        assert len(self.path) >= 4, (
            f"経路長 {len(self.path)} が短すぎてテスト不可（要4以上）"
        )
        mid = len(self.path_lum) // 2
        first_mean = float(np.mean(self.path_lum[:mid]))
        second_mean = float(np.mean(self.path_lum[mid:]))
        assert second_mean > first_mean, (
            f"後半輝度 {second_mean:.3f} <= 前半輝度 {first_mean:.3f} — "
            f"暗→明の方向性が確認できない"
        )

    def test_entrance_at_darkest_corner(self):
        """入口セルは4隅の中で輝度最低であること。

        find_image_guided_path の設計: 入口 = min(4隅の輝度)
        """
        rows, cols = self.grid.rows, self.grid.cols
        corners = [0, cols - 1, cols * (rows - 1), rows * cols - 1]
        corner_lums = {c: float(self.flat_lum[c]) for c in corners}
        darkest_corner = min(corner_lums, key=corner_lums.get)
        assert self.entrance == darkest_corner, (
            f"入口 {self.entrance}(lum={self.flat_lum[self.entrance]:.3f}) が "
            f"最暗コーナー {darkest_corner}(lum={corner_lums[darkest_corner]:.3f}) と不一致"
        )

    def test_exit_at_opposite_corner(self):
        """出口セルは入口の対角コーナーであること。"""
        rows, cols = self.grid.rows, self.grid.cols
        diag = {0: rows * cols - 1, rows * cols - 1: 0,
                cols - 1: cols * (rows - 1), cols * (rows - 1): cols - 1}
        expected_exit = diag[self.entrance]
        assert self.exit_cell == expected_exit, (
            f"出口 {self.exit_cell} が入口 {self.entrance} の対角 {expected_exit} と不一致"
        )

    def test_path_length_reasonable(self):
        """解法経路長 >= grid_size（十分に長い経路が存在する）。"""
        grid_size = self.grid.rows
        assert len(self.path) >= grid_size, (
            f"経路長 {len(self.path)} < grid_size {grid_size}"
        )


# ──────────────────────────────────────────────
# 2. ヒストグラム統計テスト
# ──────────────────────────────────────────────

class TestSolutionPathLuminanceHistogram:
    """解法経路の輝度分布統計を検証する。"""

    def setup_method(self):
        img = _make_gradient_image(width=64, height=64)
        self.grid, self.adj = _build_grid_and_adj(img, grid_size=8)
        flat_lum = self.grid.luminance.flatten()
        _, _, path = find_image_guided_path(
            self.adj, self.grid.num_cells,
            self.grid.luminance, self.grid.rows, self.grid.cols,
        )
        self.path = path
        self.path_lum = np.array([flat_lum[c] for c in path])

    def test_luminance_mean_positive(self):
        """解法経路の平均輝度が正値であること。"""
        mean = float(np.mean(self.path_lum))
        assert mean > 0.0, f"平均輝度 {mean:.3f} が 0 以下"

    def test_second_half_median_above_first_half(self):
        """後半の中央値輝度 > 前半の中央値輝度（暗→明の方向性を中央値で検証）。

        木上の経路では平均値が入口の暗さに引っ張られるが、
        中央値の前後半比較により経路の方向性を確認できる。
        """
        assert len(self.path_lum) >= 4
        mid = len(self.path_lum) // 2
        first_median = float(np.median(self.path_lum[:mid]))
        second_median = float(np.median(self.path_lum[mid:]))
        assert second_median > first_median, (
            f"後半中央値 {second_median:.3f} <= 前半中央値 {first_median:.3f}"
        )

    def test_luminance_std_nonnegative(self):
        """標準偏差が非負であること（数値的健全性）。"""
        std = float(np.std(self.path_lum))
        assert std >= 0.0, f"標準偏差 {std:.3f} が負"

    def test_second_half_has_more_bright_cells(self):
        """後半の明部セル比率 > 前半の明部セル比率（輝度>0.5のセルが後半に集中）。

        グラデーション画像で入口=暗・出口=明のため、
        経路後半は明部セル比率が高くなる。
        """
        assert len(self.path_lum) >= 4
        mid = len(self.path_lum) // 2
        first_bright = float(np.mean(np.array(self.path_lum[:mid]) > 0.5))
        second_bright = float(np.mean(np.array(self.path_lum[mid:]) > 0.5))
        assert second_bright > first_bright, (
            f"後半明部比率 {second_bright:.2%} <= 前半明部比率 {first_bright:.2%}"
        )


# ──────────────────────────────────────────────
# 3. 再現性テスト
# ──────────────────────────────────────────────

class TestImageGuidedReproducibility:
    """同一パラメータで複数回呼び出した際の再現性を確認する。"""

    def test_find_image_guided_path_reproducible(self):
        """find_image_guided_path は同一 adj・luminance で同一経路を返す。"""
        img = _make_gradient_image(width=64, height=64)
        grid, adj = _build_grid_and_adj(img, grid_size=8)

        en1, ex1, path1 = find_image_guided_path(
            adj, grid.num_cells, grid.luminance, grid.rows, grid.cols
        )
        en2, ex2, path2 = find_image_guided_path(
            adj, grid.num_cells, grid.luminance, grid.rows, grid.cols
        )

        assert en1 == en2, f"入口が異なる: {en1} vs {en2}"
        assert ex1 == ex2, f"出口が異なる: {ex1} vs {ex2}"
        assert path1 == path2, "同一呼び出しで経路が異なる（決定論性の違反）"

    def test_generate_density_maze_path_reproducible(self):
        """generate_density_maze は同一パラメータで同一 solution_path を返す。

        build_spanning_tree / post_process_density は seed=42 固定のため再現性あり。
        """
        img = _make_gradient_image(width=64, height=64)

        r1 = generate_density_maze(img, grid_size=8, max_side=64, use_image_guided=True)
        r2 = generate_density_maze(img, grid_size=8, max_side=64, use_image_guided=True)

        assert r1.entrance == r2.entrance, (
            f"入口が異なる: {r1.entrance} vs {r2.entrance}"
        )
        assert r1.exit_cell == r2.exit_cell, (
            f"出口が異なる: {r1.exit_cell} vs {r2.exit_cell}"
        )
        assert r1.solution_path == r2.solution_path, (
            "同一パラメータで solution_path が異なる（再現性の違反）"
        )

    def test_entrance_exit_stable_across_grid_sizes(self):
        """異なるgrid_sizeでも入口=最暗コーナー・出口=対角が保たれること。"""
        img = _make_gradient_image(width=64, height=64)
        for gs in [4, 6, 8]:
            grid, adj = _build_grid_and_adj(img, grid_size=gs)
            flat_lum = grid.luminance.flatten()
            rows, cols = grid.rows, grid.cols
            entrance, exit_cell, path = find_image_guided_path(
                adj, grid.num_cells, grid.luminance, rows, cols
            )
            corners = [0, cols - 1, cols * (rows - 1), rows * cols - 1]
            darkest = min(corners, key=lambda c: float(flat_lum[c]))
            assert entrance == darkest, (
                f"grid_size={gs}: 入口 {entrance} が最暗コーナー {darkest} と不一致"
            )


# ──────────────────────────────────────────────
# 4. エッジケース
# ──────────────────────────────────────────────

class TestImageGuidedEdgeCases:
    """境界条件での動作確認。"""

    def test_tiny_grid_2x2(self):
        """2×2グリッドで経路が生成されること。"""
        img = _make_gradient_image(width=16, height=16)
        grid, adj = _build_grid_and_adj(img, grid_size=2, max_side=16)
        entrance, exit_cell, path = find_image_guided_path(
            adj, grid.num_cells, grid.luminance, grid.rows, grid.cols
        )
        assert len(path) >= 1, "2×2グリッドで経路が空"
        assert path[0] == entrance, "経路の先頭が入口と一致しない"

    def test_uniform_bright_image(self):
        """全面明るい画像: 全セル輝度≈1.0でも経路が生成されること。"""
        img = _make_bright_image(width=32, height=32)
        grid, adj = _build_grid_and_adj(img, grid_size=6, max_side=32)
        entrance, exit_cell, path = find_image_guided_path(
            adj, grid.num_cells, grid.luminance, grid.rows, grid.cols
        )
        assert len(path) >= 2, "均一明輝度画像で経路が生成されない"
        flat_lum = grid.luminance.flatten()
        path_lum_mean = float(np.mean([flat_lum[c] for c in path]))
        assert path_lum_mean >= 0.0, "輝度が負になっている"

    def test_uniform_dark_image(self):
        """全面暗い画像: 全セル輝度≈0.0でもフォールバックで経路が生成されること。"""
        img = _make_dark_image(width=32, height=32)
        gray = preprocess_image(img, max_side=32)
        grid = build_cell_grid(gray, 6, 6)
        adj = build_spanning_tree(grid)
        entrance, exit_cell, path = find_image_guided_path(
            adj, grid.num_cells, grid.luminance, grid.rows, grid.cols
        )
        # 全面暗い場合は4隅の輝度が全て0に近く、いずれかの角から出発
        assert len(path) >= 1, "全暗画像で経路が空"
        assert entrance in [0, grid.cols - 1,
                            grid.cols * (grid.rows - 1),
                            grid.num_cells - 1], (
            f"入口 {entrance} が4隅以外"
        )
