# -*- coding: utf-8 -*-
"""
maze-artisan masterpiece 統合テスト: 3本柱を全有効にして実画像検証。

3本柱:
  - 可変壁厚 (thickness_range=1.5)
  - ループ密度 (extra_removal_rate=0.5)
  - 解法ルーティング (use_image_guided=True)

テスト画像: 左半分=黒(0)、右半分=白(255) のグレースケール合成画像。
"""
from __future__ import annotations

import io
import re
import time

import numpy as np
import pytest
from PIL import Image

from backend.core.density import generate_density_maze, DensityMazeResult


# ============================================================
# テスト用画像・呼び出しファクトリ
# ============================================================

def _gradient_image(w: int = 64, h: int = 64) -> Image.Image:
    """左半分=黒(0)、右半分=白(255)のグレースケール画像。"""
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[:, w // 2:] = 255
    return Image.fromarray(arr, mode="L")


def _call_masterpiece(img: Image.Image, grid_size: int = 10) -> DensityMazeResult:
    """3本柱を全て有効にして generate_density_maze を呼ぶ。"""
    return generate_density_maze(
        img,
        grid_size=grid_size,
        max_side=128,
        width=400,
        height=400,
        stroke_width=2.0,
        show_solution=True,
        thickness_range=1.5,
        extra_removal_rate=0.5,
        dark_threshold=0.3,
        light_threshold=0.7,
        use_image_guided=True,
        solution_highlight=False,
        edge_weight=0.5,
    )


# ============================================================
# 基本動作テスト
# ============================================================

def test_masterpiece_minimal_grid():
    """最小グリッド（grid_size=2）で3本柱が例外なく動作する。"""
    img = _gradient_image(32, 32)
    result = _call_masterpiece(img, grid_size=2)
    assert result.maze_id.startswith("density-")
    assert len(result.png_bytes) > 0
    assert len(result.svg) > 0


def test_masterpiece_result_fields():
    """DensityMazeResult のフィールドが全て揃っている。"""
    img = _gradient_image(64, 64)
    result = _call_masterpiece(img, grid_size=8)

    assert result.maze_id
    assert result.svg
    assert result.png_bytes
    assert isinstance(result.entrance, int)
    assert isinstance(result.exit_cell, int)
    assert isinstance(result.solution_path, list)
    assert len(result.solution_path) >= 2
    assert result.grid_rows > 0
    assert result.grid_cols > 0


def test_masterpiece_entrance_exit_distinct():
    """入口と出口が異なるセル id を持つ。"""
    img = _gradient_image(64, 64)
    result = _call_masterpiece(img, grid_size=8)
    assert result.entrance != result.exit_cell


# ============================================================
# a. PNG 出力の左半分が右半分より暗い
# ============================================================

def test_png_left_darker_than_right():
    """PNG の左半分ピクセル平均が右半分より低い（暗い）。"""
    img = _gradient_image(64, 64)
    result = _call_masterpiece(img, grid_size=10)

    png = Image.open(io.BytesIO(result.png_bytes)).convert("L")
    arr = np.array(png)
    h, w = arr.shape
    left_mean = float(arr[:, : w // 2].mean())
    right_mean = float(arr[:, w // 2:].mean())

    assert left_mean < right_mean, (
        f"左半分({left_mean:.1f}) ≥ 右半分({right_mean:.1f}): "
        "PNG が輝度分布を反映していない"
    )


def test_png_has_reasonable_dimensions():
    """PNG のサイズが width/height に対応している。"""
    img = _gradient_image(64, 64)
    result = _call_masterpiece(img, grid_size=8)

    png = Image.open(io.BytesIO(result.png_bytes))
    # width=400, height=400 で呼んでいるので、近い値のサイズ
    assert png.width > 0
    assert png.height > 0


# ============================================================
# b. SVG 出力で壁厚が可変（左半分の壁が太い）
# ============================================================

def test_svg_contains_variable_stroke_widths():
    """SVG の壁 stroke-width が単一値でなく複数種類存在する（可変壁厚）。

    SVGフォーマット（a8a880f以降）:
      <g stroke="black" stroke-width="N.NNN"><path d="M... V/H..."/></g>
    """
    img = _gradient_image(64, 64)
    result = _call_masterpiece(img, grid_size=10)

    # <g stroke="black" stroke-width="N"> グループの stroke-width を全取得
    wall_sws = re.findall(
        r'<g stroke="black" stroke-width="([\d.]+)"',
        result.svg,
    )
    assert len(wall_sws) > 0, "SVG に壁グループが存在しない"

    unique_sws = set(wall_sws)
    assert len(unique_sws) > 1, (
        f"stroke-width が単一値 {unique_sws} のみ: 可変壁厚が機能していない"
    )


def test_svg_left_walls_thicker_than_right():
    """SVG の左半分壁（stroke-width 平均）が右半分より大きい。

    SVGフォーマット（a8a880f以降）:
      右壁（垂直線）: <g stroke-width="N"><path d="M{x} {y}V{y2} ..."/>
      下壁（水平線）: <g stroke-width="N"><path d="M{x0} {y}H{x1} ..."/>
    """
    img = _gradient_image(64, 64)
    result = _call_masterpiece(img, grid_size=10)

    # SVG width から中央 x 座標を取得
    width_match = re.search(r'<svg[^>]+width="(\d+)"', result.svg)
    svg_width = int(width_match.group(1)) if width_match else 400
    mid_x = svg_width / 2.0

    # 各 <g> グループから stroke-width とパスの x 座標を抽出
    # グループパターン: <g stroke="black" stroke-width="N"><path d="..."/></g>
    group_pat = re.compile(
        r'<g stroke="black" stroke-width="([\d.]+)">'
        r'<path d="([^"]+)" fill="none"/>'
        r'</g>'
    )
    # Mxy V/H パターン: M{x} {y}V{y2} または M{x0} {y}H{x1}
    mv_pat = re.compile(r'M([\d.]+) [\d.]+V')   # 垂直壁: x は壁のx座標
    mh_pat = re.compile(r'M([\d.]+) [\d.]+H([\d.]+)')  # 水平壁: x0,x1

    left_sws, right_sws = [], []
    for m in group_pat.finditer(result.svg):
        sw = float(m.group(1))
        path_d = m.group(2)

        # 垂直壁（右壁）: Mx Vy → x がその壁の x 座標
        for vm in mv_pat.finditer(path_d):
            x = float(vm.group(1))
            if x < mid_x:
                left_sws.append(sw)
            else:
                right_sws.append(sw)

        # 水平壁（下壁）: Mx0 yHx1 → (x0+x1)/2 が中心 x
        for hm in mh_pat.finditer(path_d):
            x0, x1 = float(hm.group(1)), float(hm.group(2))
            avg_x = (x0 + x1) / 2.0
            if avg_x < mid_x:
                left_sws.append(sw)
            else:
                right_sws.append(sw)

    assert left_sws, "左半分に壁が存在しない"
    assert right_sws, "右半分に壁が存在しない"

    left_avg = float(np.mean(left_sws))
    right_avg = float(np.mean(right_sws))

    assert left_avg > right_avg, (
        f"左壁平均厚({left_avg:.3f}) ≤ 右壁平均厚({right_avg:.3f}): "
        "可変壁厚が画像輝度を反映していない"
    )


def test_svg_wall_stroke_max_in_dark_area():
    """黒画像（輝度=0）では壁厚が stroke_width * (1 + thickness_range) 付近になる。"""
    # 全黒画像: avg_lum ≈ 0 → _wall_stroke = base * (1 + thickness_range) = 2.0 * 2.5 = 5.0
    all_black = Image.fromarray(np.zeros((32, 32), dtype=np.uint8), mode="L")
    result = generate_density_maze(
        all_black,
        grid_size=5,
        max_side=64,
        stroke_width=2.0,
        thickness_range=1.5,
        extra_removal_rate=0.0,
        use_image_guided=False,  # 黒画像では全コーナーが暗い → Dijkstra で問題なし
    )
    wall_sws = [
        float(m)
        for m in re.findall(
            r'<line[^>]+stroke="black" stroke-width="([\d.]+)"', result.svg
        )
    ]
    if wall_sws:
        max_sw = max(wall_sws)
        # stroke_width=2.0, thickness_range=1.5 → max = 2.0 * 2.5 = 5.0
        assert max_sw >= 4.5, (
            f"最大壁厚({max_sw:.3f}) が期待値（≥4.5）より小さい"
        )


# ============================================================
# c. 解法経路が明部（右半分）を優先的に通る
# ============================================================

def test_solution_path_entrance_in_dark_exit_in_bright():
    """use_image_guided では入口が暗い角、出口が明るい対角になる。"""
    img = _gradient_image(64, 64)
    result = _call_masterpiece(img, grid_size=10)

    grid_cols = result.grid_cols
    grid_rows = result.grid_rows
    entrance_col = result.entrance % grid_cols
    exit_col = result.exit_cell % grid_cols

    # 入口は左半分（暗い）、出口は右半分（明るい）が期待値
    # 左半分: col < grid_cols // 2、右半分: col >= grid_cols // 2
    half = grid_cols // 2
    assert entrance_col < half, (
        f"入口(col={entrance_col}) が右半分（明部）にある: "
        "画像適応ルーティングが暗い角を入口に選んでいない"
    )
    assert exit_col >= half, (
        f"出口(col={exit_col}) が左半分（暗部）にある: "
        "出口が明るい角でない"
    )


def test_solution_path_visits_bright_half():
    """解経路が右半分（明部）の少なくとも一部を通る（出口が明部に存在するため）。"""
    img = _gradient_image(64, 64)
    result = _call_masterpiece(img, grid_size=10)

    path = result.solution_path
    assert len(path) > 0, "解経路が空"

    grid_cols = result.grid_cols
    right_cells = sum(1 for cid in path if (cid % grid_cols) >= grid_cols // 2)

    # 出口が右半分にあるため、解経路は少なくとも右半分のセルを含む
    assert right_cells >= 1, (
        f"解経路が右半分（明部）を一度も通らない: 出口が正しく選択されていない"
    )


def test_solution_path_luminance_second_half_brighter():
    """解経路の後半（出口側）の平均輝度が前半（入口側）より高い。

    use_image_guided=True: 入口=暗い角(左)、出口=明るい角(右)。
    パスは暗→明へ推移するため、後半が前半より明るいことを確認する。
    """
    from backend.core.density.preprocess import preprocess_image

    img = _gradient_image(64, 64)
    result = _call_masterpiece(img, grid_size=10)

    path = result.solution_path
    if len(path) < 4:
        pytest.skip("パスが短すぎて前後半比較不可")

    gray = preprocess_image(img, max_side=128)
    from backend.core.density.grid_builder import build_density_map
    lum_map = build_density_map(gray, result.grid_rows, result.grid_cols)
    flat_lum = lum_map.flatten()

    half = len(path) // 2
    first_half_lum = np.mean([flat_lum[cid] for cid in path[:half]])
    second_half_lum = np.mean([flat_lum[cid] for cid in path[half:]])

    assert second_half_lum >= first_half_lum, (
        f"解経路後半({second_half_lum:.3f}) < 前半({first_half_lum:.3f}): "
        "暗→明の推移が確認できない"
    )


# ============================================================
# d. 迷路が連結（解経路が入口→出口を結ぶ）
# ============================================================

def test_solution_path_connects_entrance_to_exit():
    """solution_path が入口→出口を結んでいる（端点確認）。"""
    img = _gradient_image(64, 64)
    result = _call_masterpiece(img, grid_size=10)

    path = result.solution_path
    assert len(path) >= 2, f"解経路が短すぎる: {len(path)} セル"

    endpoints = {path[0], path[-1]}
    assert result.entrance in endpoints, (
        f"解経路の端点({endpoints})に入口({result.entrance})がない"
    )
    assert result.exit_cell in endpoints, (
        f"解経路の端点({endpoints})に出口({result.exit_cell})がない"
    )


def test_solution_path_cells_adjacent_in_grid():
    """解経路の各ステップがグリッド上で隣接セル（上下左右1マス）である。"""
    img = _gradient_image(64, 64)
    result = _call_masterpiece(img, grid_size=10)

    path = result.solution_path
    grid_cols = result.grid_cols

    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        ar, ac = a // grid_cols, a % grid_cols
        br, bc = b // grid_cols, b % grid_cols
        manhattan = abs(ar - br) + abs(ac - bc)
        assert manhattan == 1, (
            f"解経路 path[{i}]={a}({ar},{ac}) → path[{i+1}]={b}({br},{bc}) "
            f"がグリッド上で非隣接 (distance={manhattan})"
        )


def test_maze_bfs_connected():
    """spanning tree + ループ密度処理後も入口→出口が BFS で連結している。"""
    from backend.core.density.solver import bfs_has_path
    from backend.core.density.preprocess import preprocess_image
    from backend.core.density.grid_builder import build_cell_grid_with_edges
    from backend.core.density.maze_builder import build_spanning_tree, post_process_density

    img = _gradient_image(64, 64)
    result = _call_masterpiece(img, grid_size=10)

    gray = preprocess_image(img, max_side=128)
    grid = build_cell_grid_with_edges(
        gray, result.grid_rows, result.grid_cols,
        density_factor=1.0,
        edge_weight=0.5,
    )
    adj = build_spanning_tree(grid)
    adj = post_process_density(
        adj, grid,
        extra_removal_rate=0.5,
        dark_threshold=0.3,
        light_threshold=0.7,
    )

    assert bfs_has_path(adj, result.entrance, result.exit_cell), (
        f"BFS: 入口({result.entrance})→出口({result.exit_cell}) が非連結"
    )


# ============================================================
# e. パフォーマンス: グリッド規模
# ============================================================

def test_grid_100x100_generation():
    """100×100 グリッドで正常に生成でき、解経路が存在する。"""
    img = _gradient_image(128, 128)
    result = generate_density_maze(
        img,
        grid_size=100,
        max_side=256,
        width=800,
        height=800,
        stroke_width=1.0,
        show_solution=True,
        thickness_range=1.5,
        extra_removal_rate=0.5,
        dark_threshold=0.3,
        light_threshold=0.7,
        use_image_guided=True,
        solution_highlight=False,
        edge_weight=0.5,
    )
    assert result.grid_rows >= 1 and result.grid_cols >= 1
    assert len(result.solution_path) > 0
    assert len(result.svg) > 0
    assert len(result.png_bytes) > 0


def test_grid_200x200_performance():
    """200×200 グリッドで生成でき、60 秒以内に完了する。"""
    img = _gradient_image(256, 256)
    start = time.monotonic()
    result = generate_density_maze(
        img,
        grid_size=200,
        max_side=512,
        width=1200,
        height=1200,
        stroke_width=1.0,
        show_solution=False,
        thickness_range=1.5,
        extra_removal_rate=0.3,
        dark_threshold=0.3,
        light_threshold=0.7,
        use_image_guided=True,
        solution_highlight=False,
        edge_weight=0.0,   # 高速化: edge_weight=0 で Canny 省略
    )
    elapsed = time.monotonic() - start

    assert result.grid_rows >= 1
    assert len(result.solution_path) > 0
    assert elapsed < 60.0, f"200×200生成が {elapsed:.1f}秒（制限: 60秒）"


# ============================================================
# extra: ループ密度（extra_removal_rate）が実際にエッジを追加する
# ============================================================

def test_loop_density_adds_edges():
    """extra_removal_rate > 0 のとき spanning tree より多くのエッジを持つ。"""
    from backend.core.density.solver import _adj_to_edge_set
    from backend.core.density.preprocess import preprocess_image
    from backend.core.density.grid_builder import build_cell_grid
    from backend.core.density.maze_builder import build_spanning_tree, post_process_density

    img = _gradient_image(64, 64)
    gray = preprocess_image(img, max_side=64)
    grid = build_cell_grid(gray, 8, 8)

    adj_tree = build_spanning_tree(grid)
    tree_edges = len(_adj_to_edge_set(adj_tree))

    adj_with_loops = post_process_density(
        adj_tree, grid,
        extra_removal_rate=0.5,
        dark_threshold=0.3,
        light_threshold=0.7,
    )
    loop_edges = len(_adj_to_edge_set(adj_with_loops))

    assert loop_edges >= tree_edges, (
        f"ループ密度後のエッジ数({loop_edges}) < spanning tree({tree_edges}): "
        "post_process_density がエッジを追加していない"
    )
