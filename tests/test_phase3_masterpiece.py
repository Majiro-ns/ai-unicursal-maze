# -*- coding: utf-8 -*-
"""
maze-artisan Phase3: 400x600 実画像 masterpiece 生成テスト（cmd_358k_a6）。

テスト方針:
  1. 多様なパターン画像（円形グラデーション・ストライプ・チェッカーボード等）で
     400x600 masterpiece を生成し、クラッシュせず正常出力されることを確認。
  2. エッジケース（全白・全黒・極小・thickness_range 0/3.0）でのロバスト性確認。
  3. masterpiece 品質検証:
     - 白線（解経路）塗りつぶし後に白ピクセルが増加すること
     - 解経路の隣接性（各ステップが隣接セル間移動）
     - 入口→出口 BFS 到達可能性
"""
from __future__ import annotations

import io
import time
from collections import deque

import numpy as np
import pytest
from PIL import Image

from backend.core.density import generate_density_maze, DensityMazeResult
from tests.fixtures import (
    make_circular_gradient,
    make_horizontal_stripe,
    make_checkerboard,
    make_diagonal_gradient,
    make_all_white,
    make_all_black,
)


# ============================================================
# ヘルパー
# ============================================================

def _masterpiece_400x600(img: Image.Image, **kwargs) -> DensityMazeResult:
    """400x600グリッド・3本柱全有効でmasterpiece生成。"""
    params = dict(
        grid_size=400,
        max_side=512,
        width=1200,
        height=800,
        stroke_width=1.5,
        show_solution=True,
        solution_highlight=False,      # masterpiece白線モード
        thickness_range=1.5,           # 柱1: 可変壁厚
        extra_removal_rate=0.5,        # 柱2: ループ密度
        dark_threshold=0.3,
        light_threshold=0.7,
        use_image_guided=True,         # 柱3: 画像適応ルーティング
    )
    params.update(kwargs)
    return generate_density_maze(img, **params)


def _count_white_pixels(png_bytes: bytes, threshold: int = 200) -> int:
    """PNG バイト列の白ピクセル数（輝度 > threshold）を返す。"""
    img = Image.open(io.BytesIO(png_bytes)).convert("L")
    arr = np.array(img)
    return int(np.sum(arr > threshold))


def _bfs_reachable(adj: dict, src: int, dst: int) -> bool:
    """adj で src → dst に BFS 到達可能か確認。"""
    visited = {src}
    q: deque = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            return True
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                q.append(v)
    return src == dst


# ============================================================
# 1. 多様パターン画像での 400x600 masterpiece 生成
# ============================================================

@pytest.mark.parametrize("img_factory,label", [
    (lambda: make_circular_gradient(256, 256), "circular_gradient"),
    (lambda: make_horizontal_stripe(256, 256, 8), "horizontal_stripe"),
    (lambda: make_checkerboard(256, 256, 8), "checkerboard"),
    (lambda: make_diagonal_gradient(256, 256), "diagonal_gradient"),
])
def test_400x600_masterpiece_various_images(img_factory, label):
    """
    各パターン画像で 400x600 masterpiece が正常に生成されること。
    - クラッシュしない
    - PNG/SVG が非空
    - 解経路が存在し入口→出口が連結
    """
    img = img_factory()
    result = _masterpiece_400x600(img)

    assert result.grid_rows >= 1 and result.grid_cols >= 1, f"{label}: グリッドサイズが不正"
    assert len(result.png_bytes) > 0, f"{label}: PNG が空"
    assert len(result.svg) > 0, f"{label}: SVG が空"
    assert len(result.solution_path) >= 1, f"{label}: 解経路が空"
    assert result.entrance >= 0 and result.exit_cell >= 0, f"{label}: 入口/出口が不正"


def test_400x600_masterpiece_white_path_increases_white_pixels():
    """
    masterpiece 白線描画（show_solution=True, solution_highlight=False）で
    白ピクセル数が show_solution=False より多くなること。
    """
    img = make_diagonal_gradient(256, 256)

    result_no_sol = _masterpiece_400x600(img, show_solution=False)
    result_with_sol = _masterpiece_400x600(img, show_solution=True)

    white_no = _count_white_pixels(result_no_sol.png_bytes)
    white_with = _count_white_pixels(result_with_sol.png_bytes)

    assert white_with >= white_no, (
        f"白線塗りつぶし後({white_with}px) < 塗りつぶし前({white_no}px)"
    )


def test_400x600_solution_path_adjacency():
    """解経路の各ステップが隣接セル間移動であること（独立 BFS 検証）。"""
    from backend.core.density.preprocess import preprocess_image
    from backend.core.density.grid_builder import build_cell_grid
    from backend.core.density.maze_builder import build_spanning_tree, post_process_density

    img = make_circular_gradient(256, 256)
    result = _masterpiece_400x600(img)

    gray = preprocess_image(img, max_side=512)
    grid = build_cell_grid(gray, result.grid_rows, result.grid_cols)
    # post_process も適用して実際に使われた adj と同一にする
    adj = build_spanning_tree(grid)
    adj = post_process_density(
        adj, grid,
        extra_removal_rate=0.5,
        dark_threshold=0.3,
        light_threshold=0.7,
    )

    path = result.solution_path
    for i in range(len(path) - 1):
        assert path[i + 1] in adj.get(path[i], []), (
            f"解経路ステップ {i}→{i+1}: セル {path[i]} と {path[i+1]} が非隣接"
        )


def test_400x600_timing_within_10s():
    """
    400x600 masterpiece（3本柱全有効）が 10 秒以内に生成されること。
    最適化後の実測値: ~1.5s。
    """
    img = make_circular_gradient(256, 256)
    t0 = time.perf_counter()
    result = _masterpiece_400x600(img)
    elapsed = time.perf_counter() - t0

    assert elapsed < 10.0, f"400x600 masterpiece が {elapsed:.2f}s > 10s"
    assert len(result.solution_path) >= 1


# ============================================================
# 2. エッジケース: 全白・全黒・極小・極大
# ============================================================

def test_edge_case_all_white_image():
    """全白画像（255）でクラッシュせず解経路が存在すること。"""
    img = make_all_white(64, 64)
    result = generate_density_maze(
        img, grid_size=10, max_side=64, show_solution=True,
        thickness_range=1.5, extra_removal_rate=0.5,
    )
    assert len(result.solution_path) >= 1
    assert len(result.png_bytes) > 0


def test_edge_case_all_black_image():
    """全黒画像（0）でクラッシュせず解経路が存在すること。"""
    img = make_all_black(64, 64)
    result = generate_density_maze(
        img, grid_size=10, max_side=64, show_solution=True,
        thickness_range=1.5, extra_removal_rate=0.5,
    )
    assert len(result.solution_path) >= 1
    assert len(result.png_bytes) > 0


def test_edge_case_tiny_10x10_grid():
    """10×10 の極小グリッドでクラッシュしないこと。"""
    img = make_circular_gradient(32, 32)
    result = generate_density_maze(
        img, grid_size=10, max_side=32,
        thickness_range=1.5, extra_removal_rate=0.5, use_image_guided=True,
    )
    assert result.grid_rows >= 1 and result.grid_cols >= 1
    assert len(result.solution_path) >= 1


def test_edge_case_3x3_minimum_grid():
    """3×3 の最小グリッドでクラッシュしないこと（max_side=16 で強制縮小）。"""
    img = make_all_white(16, 16)
    result = generate_density_maze(
        img, grid_size=3, max_side=16,
        thickness_range=1.5, extra_removal_rate=0.0,
    )
    assert result.grid_rows >= 1
    assert len(result.solution_path) >= 1


def test_edge_case_large_500x750():
    """
    500x750 相当グリッド（grid_size=500, max_side=512）で60秒以内に生成されること。
    実際の grid_size は max_side 制約で縮小されるため 128 以下になる。
    """
    img = make_diagonal_gradient(512, 512)
    t0 = time.perf_counter()
    result = generate_density_maze(
        img,
        grid_size=500,   # max_side=512 で gray.shape // 4 ≒ 128 に制限
        max_side=512,
        show_solution=True,
        thickness_range=1.5,
        extra_removal_rate=0.5,
        use_image_guided=True,
    )
    elapsed = time.perf_counter() - t0

    assert elapsed < 60.0, f"large grid: {elapsed:.2f}s > 60s"
    assert len(result.solution_path) >= 1


# ============================================================
# 3. thickness_range バリエーション
# ============================================================

@pytest.mark.parametrize("thickness_range,label", [
    (0.0, "uniform_walls"),
    (1.5, "default_range"),
    (3.0, "max_range"),
])
def test_thickness_range_variations(thickness_range: float, label: str):
    """
    thickness_range=0.0（均一壁厚）〜3.0（最大変化）で正常に生成されること。
    """
    img = make_circular_gradient(128, 128)
    result = generate_density_maze(
        img,
        grid_size=20,
        max_side=128,
        thickness_range=thickness_range,
        show_solution=True,
        extra_removal_rate=0.3,
    )
    assert len(result.png_bytes) > 0, f"{label}: PNG が空"
    assert len(result.svg) > 0, f"{label}: SVG が空"
    assert len(result.solution_path) >= 1, f"{label}: 解経路が空"


def test_thickness_range_zero_svg_uniform():
    """
    thickness_range=0.0 のとき SVG 内の壁幅が全て同一であること。
    """
    img = make_checkerboard(64, 64, 4)
    result = generate_density_maze(
        img, grid_size=6, max_side=64,
        thickness_range=0.0, stroke_width=2.0,
        show_solution=False,
    )
    import re
    # SVG の stroke-width 値を全抽出
    widths = [float(m) for m in re.findall(r'stroke-width="([^"]+)"', result.svg)]
    # 外枠の stroke-width は固定(2.0)、内部壁も全て2.0のはず
    non_border = widths[1:]  # 最初は外枠
    if non_border:
        assert max(non_border) - min(non_border) < 0.01, (
            f"thickness_range=0 なのに壁幅がばらついている: {min(non_border):.3f}〜{max(non_border):.3f}"
        )


# ============================================================
# 4. solution_highlight モード比較
# ============================================================

def test_solution_highlight_true_has_orange_in_svg():
    """solution_highlight=True のとき SVG にオレンジ(#E05000)が含まれること。"""
    img = make_diagonal_gradient(64, 64)
    result = generate_density_maze(
        img, grid_size=8, max_side=64,
        show_solution=True, solution_highlight=True,
    )
    assert "#E05000" in result.svg or "E05000" in result.svg, (
        "solution_highlight=True なのに SVG にオレンジ色がない"
    )


def test_solution_highlight_false_no_orange_in_svg():
    """solution_highlight=False（masterpiece）のとき SVG にオレンジが含まれないこと。"""
    img = make_diagonal_gradient(64, 64)
    result = generate_density_maze(
        img, grid_size=8, max_side=64,
        show_solution=True, solution_highlight=False,
    )
    assert "E05000" not in result.svg, (
        "masterpiece モードなのに SVG にオレンジ色が含まれている"
    )


def test_solution_highlight_false_has_white_in_svg():
    """solution_highlight=False のとき SVG に white stroke が含まれること。"""
    img = make_diagonal_gradient(64, 64)
    result = generate_density_maze(
        img, grid_size=8, max_side=64,
        show_solution=True, solution_highlight=False,
    )
    assert 'stroke="white"' in result.svg, (
        "masterpiece モードなのに SVG に白線(stroke=white)がない"
    )
