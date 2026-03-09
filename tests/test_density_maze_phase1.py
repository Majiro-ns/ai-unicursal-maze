# -*- coding: utf-8 -*-
"""
密度迷路 Phase 1/2 テスト。
Phase 1（M-12）: 01a §6.1 受け入れ基準: 入口1・出口1・解経路1本、連結性。
Phase 2: 多領域セグメンテーション・テクスチャ割り当て・解ヒューリスティクス・可視化改善。
"""
from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image

from backend.core.density import generate_density_maze
from backend.core.density.grid_builder import build_cell_grid
from backend.core.density.maze_builder import build_spanning_tree
from backend.core.density.entrance_exit import find_entrance_exit_and_path


def _make_small_image(w: int = 64, h: int = 64) -> Image.Image:
    arr = np.ones((h, w), dtype=np.uint8) * 128
    return Image.fromarray(arr, mode="L")


def test_density_maze_returns_one_entrance_one_exit():
    """入口が1つ、出口が1つであること。"""
    img = _make_small_image(64, 64)
    result = generate_density_maze(img, grid_size=5, max_side=64)
    assert result.entrance >= 0
    assert result.exit_cell >= 0
    assert result.entrance < result.grid_rows * result.grid_cols
    assert result.exit_cell < result.grid_rows * result.grid_cols


def test_density_maze_solution_path_is_single_path():
    """解経路が1本（入口から出口までのセル列）であること。"""
    img = _make_small_image(64, 64)
    result = generate_density_maze(img, grid_size=6, max_side=64)
    path = result.solution_path
    assert len(path) >= 1
    assert path[0] == result.entrance
    assert path[-1] == result.exit_cell
    # 経路は重複なし（木の単純路）
    assert len(path) == len(set(path))


def test_density_maze_connectivity():
    """全セルが連結であること（spanning tree）。"""
    img = _make_small_image(32, 32)
    result = generate_density_maze(img, grid_size=4, max_side=32)
    from backend.core.density.preprocess import preprocess_image
    from backend.core.density.grid_builder import build_cell_grid
    gray = preprocess_image(img, max_side=32)
    grid = build_cell_grid(gray, result.grid_rows, result.grid_cols)
    adj = build_spanning_tree(grid)
    n = grid.num_cells
    # BFS で 0 から到達可能なセル数 = n なら連結
    visited = set()
    q = deque([0])
    visited.add(0)
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                q.append(v)
    assert len(visited) == n, "グラフが連結でない"


def test_density_maze_svg_and_png_non_empty():
    """SVG と PNG が空でないこと。"""
    img = _make_small_image(48, 48)
    result = generate_density_maze(img, grid_size=4, max_side=48)
    assert len(result.svg) > 100
    assert "svg" in result.svg.lower()
    assert len(result.png_bytes) > 100


def test_density_maze_1x1_grid():
    """1x1 グリッドでも落ちずに入口=出口=0、解経路=[0] で返ること。"""
    img = _make_small_image(8, 8)
    result = generate_density_maze(img, grid_size=1, max_side=8)
    assert result.grid_rows == 1 and result.grid_cols == 1
    assert result.entrance == 0 and result.exit_cell == 0
    assert result.solution_path == [0]
    assert len(result.svg) > 50
    assert len(result.png_bytes) > 50


def test_density_maze_unique_path_between_entrance_exit():
    """入口から出口までの経路がグラフ上で唯一であること（perfect maze）。"""
    from backend.core.density.preprocess import preprocess_image
    img = _make_small_image(32, 32)
    result = generate_density_maze(img, grid_size=3, max_side=32)
    gray = preprocess_image(img, max_side=32)
    grid = build_cell_grid(gray, result.grid_rows, result.grid_cols)
    adj = build_spanning_tree(grid)
    # 木なので任意2点間の経路は唯一。solution_path が入口→出口の経路になっているか確認
    entrance, exit_cell, path = find_entrance_exit_and_path(adj, grid.num_cells)
    assert path[0] == entrance
    assert path[-1] == exit_cell
    # 隣接性
    for i in range(len(path) - 1):
        assert path[i + 1] in adj.get(path[i], []), f"path[{i}] and path[{i+1}] are not adjacent"


# ============================================================
# Phase 2 テスト
# ============================================================

from backend.core.density.segment import segment_by_luminance, segment_single_region
from backend.core.density.texture import (
    TextureType,
    PRESET_FACE,
    PRESET_GENERIC,
    PRESET_LANDSCAPE,
    assign_cell_textures,
    compute_gradient_angles,
)
from backend.core.density.grid_builder import _spiral_angle
from backend.core.density.grid_builder import (
    CellGrid,
    build_cell_grid_with_texture,
)
from backend.core.density.entrance_exit import find_entrance_exit_heuristic


def _make_gradient_image(w: int = 64, h: int = 64) -> Image.Image:
    """左が暗く右が明るいグラデーション画像。"""
    arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return Image.fromarray(arr, mode="L")


def _make_checkerboard(w: int = 64, h: int = 64) -> Image.Image:
    """暗い・明るいチェッカーボード画像（セグメンテーション検証用）。"""
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[::2, ::2] = 200
    arr[1::2, 1::2] = 200
    arr[::2, 1::2] = 50
    arr[1::2, ::2] = 50
    return Image.fromarray(arr, mode="L")


# ---------- セグメンテーション ----------

def test_segment_single_region_shape():
    """segment_single_region は入力と同形の全1配列を返す。"""
    img = _make_small_image(32, 32)
    from backend.core.density.preprocess import preprocess_image
    gray = preprocess_image(img, max_side=32)
    labels = segment_single_region(gray)
    assert labels.shape == gray.shape
    assert np.all(labels == 1)


def test_segment_multi_region_n_labels():
    """segment_by_luminance は n_clusters 種類のラベルを返す。"""
    img = _make_checkerboard(32, 32)
    from backend.core.density.preprocess import preprocess_image
    gray = preprocess_image(img, max_side=32)
    labels = segment_by_luminance(gray, n_clusters=4)
    assert labels.shape == gray.shape
    unique_labels = np.unique(labels)
    # チェッカーボードは2値なので2〜4ラベルになる
    assert len(unique_labels) >= 2
    assert len(unique_labels) <= 4


def test_segment_label_order_dark_is_zero():
    """ラベル 0 が最暗クラスタ、ラベル n-1 が最明クラスタであること。"""
    img = _make_gradient_image(64, 64)
    from backend.core.density.preprocess import preprocess_image
    gray = preprocess_image(img, max_side=64)
    labels = segment_by_luminance(gray, n_clusters=4)
    # 左列（暗い）の代表ラベル < 右列（明るい）の代表ラベル
    left_label = int(np.median(labels[:, :8]))
    right_label = int(np.median(labels[:, -8:]))
    assert left_label <= right_label, (
        f"暗い領域のラベル({left_label}) > 明るい領域のラベル({right_label})"
    )


def test_segment_n_clusters_1():
    """n_clusters=1 のとき全ラベルが 0。"""
    img = _make_small_image(32, 32)
    from backend.core.density.preprocess import preprocess_image
    gray = preprocess_image(img, max_side=32)
    labels = segment_by_luminance(gray, n_clusters=1)
    assert np.all(labels == 0)


# ---------- テクスチャ割り当て ----------

def test_texture_assign_cells_shape():
    """assign_cell_textures の出力形状が grid と一致する。"""
    img = _make_checkerboard(32, 32)
    from backend.core.density.preprocess import preprocess_image
    gray = preprocess_image(img, max_side=32)
    label_map = segment_by_luminance(gray, n_clusters=4)
    dummy_grid = CellGrid(rows=8, cols=8, luminance=np.zeros((8, 8)), walls=[])
    tex = assign_cell_textures(dummy_grid, label_map, PRESET_FACE)
    assert tex.shape == (8, 8)
    assert all(isinstance(tex[r, c], TextureType) for r in range(8) for c in range(8))


def test_texture_preset_face_contains_directional():
    """PRESET_FACE には DIRECTIONAL テクスチャが含まれる（髪・背景）。"""
    assert TextureType.DIRECTIONAL in PRESET_FACE.values()


def test_texture_preset_generic_all_random():
    """PRESET_GENERIC は全て RANDOM。"""
    assert all(v == TextureType.RANDOM for v in PRESET_GENERIC.values())


# ---------- グラジエント方向 ----------

def test_gradient_angles_shape():
    """compute_gradient_angles の出力形状が grid サイズと一致する。"""
    img = _make_gradient_image(64, 64)
    from backend.core.density.preprocess import preprocess_image
    gray = preprocess_image(img, max_side=64)
    angles = compute_gradient_angles(gray, grid_rows=8, grid_cols=8)
    assert angles.shape == (8, 8)
    assert angles.dtype == np.float64


def test_gradient_angles_horizontal_image():
    """水平グラジエント画像では cos(angle) が大きい（水平方向のグラジエント）。"""
    img = _make_gradient_image(64, 64)
    from backend.core.density.preprocess import preprocess_image
    gray = preprocess_image(img, max_side=64)
    angles = compute_gradient_angles(gray, grid_rows=4, grid_cols=8)
    # 水平グラジエントなら angle ≈ 0、cos(angle) ≈ 1 が多数を占める
    cos_vals = np.abs(np.cos(angles))
    assert float(np.mean(cos_vals)) > 0.5


# ---------- テクスチャ付きグリッド ----------

def test_build_with_texture_directional_wall_count():
    """build_cell_grid_with_texture が正しい数の壁リストを返す。"""
    img = _make_small_image(64, 64)
    from backend.core.density.preprocess import preprocess_image
    gray = preprocess_image(img, max_side=64)
    rows, cols = 4, 4
    all_directional = np.full((rows, cols), TextureType.DIRECTIONAL, dtype=object)
    angles = np.zeros((rows, cols), dtype=np.float64)
    grid = build_cell_grid_with_texture(gray, rows, cols, all_directional, angles)
    # 期待壁数: rows*(cols-1) + cols*(rows-1)
    expected = rows * (cols - 1) + cols * (rows - 1)
    assert len(grid.walls) == expected


def test_build_with_texture_directional_weights_differ():
    """DIRECTIONAL テクスチャでは壁の重みにバリエーションが生まれる。"""
    img = _make_gradient_image(64, 64)
    from backend.core.density.preprocess import preprocess_image
    gray = preprocess_image(img, max_side=64)
    rows, cols = 6, 6
    all_directional = np.full((rows, cols), TextureType.DIRECTIONAL, dtype=object)
    angles = compute_gradient_angles(gray, rows, cols)
    grid = build_cell_grid_with_texture(gray, rows, cols, all_directional, angles, bias_strength=0.5)
    weights = [w for _, _, w in grid.walls]
    # bias ありなのでランダムより重みのばらつきがある（std > 0）
    assert float(np.std(weights)) > 0.0


# ---------- 解ヒューリスティクス ----------

def test_entrance_exit_heuristic_valid_path():
    """find_entrance_exit_heuristic が有効な解経路を返す。"""
    img = _make_small_image(32, 32)
    result = generate_density_maze(img, grid_size=4, max_side=32, use_heuristic=True)
    path = result.solution_path
    assert len(path) >= 1
    assert path[0] == result.entrance
    assert path[-1] == result.exit_cell
    assert len(path) == len(set(path)), "解経路に重複セルがある"


def test_entrance_exit_heuristic_gradient_image():
    """グラデーション画像で use_heuristic=True が明るい領域の端を優先する。"""
    img = _make_gradient_image(64, 64)
    from backend.core.density.preprocess import preprocess_image
    from backend.core.density.grid_builder import build_cell_grid
    gray = preprocess_image(img, max_side=64)
    rows = cols = 6
    grid = build_cell_grid(gray, rows, cols)
    adj = build_spanning_tree(grid)
    entrance, exit_cell, path = find_entrance_exit_heuristic(adj, grid.num_cells, grid.luminance)
    assert len(path) >= 1
    assert path[0] == entrance
    assert path[-1] == exit_cell


# ---------- 解経路可視化改善 ----------

def test_svg_solution_corridor_style():
    """Phase 2 SVG 解経路が廊下風（stroke-linecap=round、太い線幅）になっている。"""
    img = _make_small_image(48, 48)
    result = generate_density_maze(img, grid_size=4, max_side=48, show_solution=True)
    assert "stroke-linecap" in result.svg
    assert "round" in result.svg


def test_svg_solution_entrance_exit_markers():
    """Phase 2 SVG に入口・出口マーカー（circle）が含まれる（solution_highlight=True モード）。"""
    img = _make_small_image(48, 48)
    result = generate_density_maze(img, grid_size=4, max_side=48, show_solution=True, solution_highlight=True)
    assert "circle" in result.svg.lower()


def test_png_solution_non_empty_with_corridor():
    """Phase 2 PNG が有効なバイト列を返す。"""
    img = _make_small_image(48, 48)
    result = generate_density_maze(img, grid_size=4, max_side=48, show_solution=True)
    assert len(result.png_bytes) > 200


# ---------- Phase 2 フルパイプライン ----------

def test_phase2_full_pipeline_generic():
    """Phase 2 フルパイプライン（GENERIC プリセット）が正常に動作する。"""
    img = _make_small_image(64, 64)
    result = generate_density_maze(
        img, grid_size=5, max_side=64,
        use_texture=True, use_heuristic=True,
        preset="generic", n_segments=4,
    )
    assert result.entrance >= 0
    assert result.exit_cell >= 0
    assert len(result.solution_path) >= 1
    assert result.segment_map is not None
    assert result.texture_map is not None
    assert len(result.svg) > 100
    assert len(result.png_bytes) > 100


def test_phase2_full_pipeline_face_preset():
    """Phase 2 フルパイプライン（FACE プリセット）が正常に動作する。"""
    img = _make_gradient_image(64, 64)
    result = generate_density_maze(
        img, grid_size=6, max_side=64,
        use_texture=True, use_heuristic=True,
        preset="face", n_segments=4,
    )
    assert result.entrance >= 0
    assert result.segment_map is not None
    assert result.segment_map.shape[0] > 0


def test_phase2_full_pipeline_landscape_preset():
    """Phase 2 フルパイプライン（LANDSCAPE プリセット）が正常に動作する。"""
    img = _make_gradient_image(64, 64)
    result = generate_density_maze(
        img, grid_size=6, max_side=64,
        use_texture=True,
        preset="landscape", n_segments=4,
    )
    assert result.entrance >= 0
    assert result.texture_map is not None


def test_phase2_segment_map_in_result():
    """use_texture=True のとき result.segment_map が正しい形状を持つ。"""
    img = _make_checkerboard(64, 64)
    result = generate_density_maze(
        img, grid_size=4, max_side=64,
        use_texture=True, n_segments=4,
    )
    assert result.segment_map is not None
    assert result.segment_map.ndim == 2
    assert result.segment_map.shape[0] > 0


def test_phase2_texture_map_in_result():
    """use_texture=True のとき result.texture_map が grid 形状で返る。"""
    img = _make_small_image(64, 64)
    result = generate_density_maze(
        img, grid_size=5, max_side=64,
        use_texture=True,
    )
    assert result.texture_map is not None
    assert result.texture_map.shape == (result.grid_rows, result.grid_cols)


def test_phase2_perfect_maze_preserved():
    """Phase 2 でも perfect maze（入口1・出口1・解経路1本）が維持される。"""
    img = _make_gradient_image(48, 48)
    result = generate_density_maze(
        img, grid_size=5, max_side=48,
        use_texture=True, use_heuristic=True,
        preset="face",
    )
    path = result.solution_path
    assert path[0] == result.entrance
    assert path[-1] == result.exit_cell
    assert len(path) == len(set(path)), "Phase 2 で解経路に重複が生じた"


# ============================================================
# Phase 2 続行: SPIRAL テクスチャ（01a §3.1 / §7.3）
# ============================================================

def test_texture_spiral_enum_exists():
    """TextureType.SPIRAL が定義されていること（01a §3.1 らせん要件）。"""
    assert hasattr(TextureType, "SPIRAL")
    assert TextureType.SPIRAL.value == "spiral"


def test_preset_face_contains_spiral():
    """PRESET_FACE に SPIRAL テクスチャが含まれること（影・輪郭領域）。"""
    assert TextureType.SPIRAL in PRESET_FACE.values()


def test_preset_landscape_contains_spiral():
    """PRESET_LANDSCAPE に SPIRAL テクスチャが含まれること（建物・岩領域）。"""
    assert TextureType.SPIRAL in PRESET_LANDSCAPE.values()


def test_spiral_angle_center_returns_zero():
    """中心セルでは _spiral_angle が 0.0 を返す。"""
    angle = _spiral_angle(2, 2, 5, 5)  # 中心 = (2, 2)
    assert abs(angle) < 1e-6


def test_spiral_angle_top_vs_right_differ():
    """上辺セルと右辺セルで _spiral_angle の値が異なる（方向分化の確認）。"""
    rows, cols = 7, 7
    # 上辺中央: (0, 3) → θ ≈ -π/2
    angle_top = _spiral_angle(0, 3, rows, cols)
    # 右辺中央: (3, 6) → θ ≈ 0
    angle_right = _spiral_angle(3, 6, rows, cols)
    # 上辺は sin²(θ)=1（右壁バイアス大）、右辺は sin²(θ)≈0（右壁バイアス小）
    assert abs(float(np.sin(angle_top) ** 2) - 1.0) < 0.1, "上辺の sin²(θ) が 1 に近くない"
    assert float(np.sin(angle_right) ** 2) < 0.1, "右辺の sin²(θ) が 0 に近くない"


def test_build_with_spiral_texture_wall_count():
    """SPIRAL テクスチャで正しい壁数が生成される。"""
    img = _make_small_image(64, 64)
    from backend.core.density.preprocess import preprocess_image
    gray = preprocess_image(img, max_side=64)
    rows, cols = 4, 4
    all_spiral = np.full((rows, cols), TextureType.SPIRAL, dtype=object)
    angles = np.zeros((rows, cols), dtype=np.float64)
    grid = build_cell_grid_with_texture(gray, rows, cols, all_spiral, angles)
    expected = rows * (cols - 1) + cols * (rows - 1)
    assert len(grid.walls) == expected


def test_build_with_spiral_texture_weights_differ_from_random():
    """均一輝度画像で SPIRAL の壁重みに分散が生じること（RANDOM は分散ゼロ）。

    均一輝度（全セル同一）では RANDOM はすべて同一重みになるが、
    SPIRAL は角度バイアスにより重みが位置によって異なる。
    """
    # 均一輝度画像（全ピクセル=128, 輝度≈0.5 で統一）
    uniform_img = _make_small_image(64, 64)   # _make_small_image は全128の均一画像
    from backend.core.density.preprocess import preprocess_image
    gray = preprocess_image(uniform_img, max_side=64)
    rows, cols = 6, 6
    angles = np.zeros((rows, cols), dtype=np.float64)
    # RANDOM: バイアスなし → 均一輝度なので全壁同一重み → std ≈ 0
    all_random = np.full((rows, cols), TextureType.RANDOM, dtype=object)
    grid_rand = build_cell_grid_with_texture(gray, rows, cols, all_random, angles, bias_strength=0.5)
    std_rand = float(np.std([w for _, _, w in grid_rand.walls]))
    # SPIRAL: 角度バイアスにより壁ごとに重みが異なる → std > 0
    all_spiral = np.full((rows, cols), TextureType.SPIRAL, dtype=object)
    grid_spiral = build_cell_grid_with_texture(gray, rows, cols, all_spiral, angles, bias_strength=0.5)
    std_spiral = float(np.std([w for _, _, w in grid_spiral.walls]))
    assert std_spiral > std_rand, (
        f"SPIRAL std({std_spiral:.4f}) は RANDOM std({std_rand:.4f}) より大きくない。"
        "均一輝度画像で SPIRAL バイアスが壁の重みを変化させていない可能性あり。"
    )


def test_phase2_full_pipeline_face_preset_with_spiral():
    """FACE プリセット（SPIRAL含む）のフルパイプラインが正常動作する。"""
    img = _make_gradient_image(64, 64)
    result = generate_density_maze(
        img, grid_size=6, max_side=64,
        use_texture=True, use_heuristic=True,
        preset="face", n_segments=4,
    )
    assert result.entrance >= 0
    assert result.exit_cell >= 0
    assert len(result.solution_path) >= 1
    assert result.segment_map is not None
    assert result.texture_map is not None
    assert len(result.svg) > 100


def test_phase2_perfect_maze_preserved_with_spiral():
    """SPIRAL テクスチャでも perfect maze（解経路1本・重複なし）が維持される。"""
    img = _make_gradient_image(48, 48)
    result = generate_density_maze(
        img, grid_size=5, max_side=48,
        use_texture=True, use_heuristic=True,
        preset="landscape",  # SPIRAL を含む
    )
    path = result.solution_path
    assert len(path) >= 1
    assert path[0] == result.entrance
    assert path[-1] == result.exit_cell
    assert len(path) == len(set(path)), "SPIRAL テクスチャで解経路に重複が生じた"


# ============================================================
# Phase 2 続行: density_map 正確性・密度制御・エッジケース・BFSソルバ
# ============================================================

from backend.core.density.preprocess import preprocess_image
from backend.core.density.grid_builder import build_cell_grid


def _make_all_black_image(w: int = 32, h: int = 32) -> Image.Image:
    """全黒画像（輝度 0）。"""
    arr = np.zeros((h, w), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def _make_all_white_image(w: int = 32, h: int = 32) -> Image.Image:
    """全白画像（輝度 255）。"""
    arr = np.full((h, w), 255, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


# ---------- density_map 正確性 ----------

def test_preprocess_output_range_gradient():
    """preprocess_image の出力値が 0.0〜1.0 の範囲内に収まること。"""
    img = _make_gradient_image(64, 64)
    gray = preprocess_image(img, max_side=64)
    assert float(gray.min()) >= 0.0
    assert float(gray.max()) <= 1.0


def test_preprocess_all_black_output_near_zero():
    """全黒画像の前処理出力は 0.0 に近いこと（density_map の暗部再現性）。"""
    img = _make_all_black_image(32, 32)
    gray = preprocess_image(img, max_side=32)
    assert float(np.mean(gray)) < 0.05, f"全黒画像の平均輝度が高すぎる: {np.mean(gray):.4f}"


def test_preprocess_all_white_output_near_one():
    """全白画像の前処理出力は 1.0 に近いこと（density_map の明部再現性）。"""
    img = _make_all_white_image(32, 32)
    gray = preprocess_image(img, max_side=32)
    assert float(np.mean(gray)) > 0.95, f"全白画像の平均輝度が低すぎる: {np.mean(gray):.4f}"


# ---------- Kruskal 密度制御（高/中/低） ----------

def test_kruskal_high_density_factor_connected():
    """density_factor=2.0（高密度）でも全セル連結の perfect maze が生成される。"""
    img = _make_small_image(32, 32)
    result = generate_density_maze(img, grid_size=4, max_side=32, density_factor=2.0)
    gray = preprocess_image(img, max_side=32)
    grid = build_cell_grid(gray, result.grid_rows, result.grid_cols, density_factor=2.0)
    adj = build_spanning_tree(grid)
    visited: set = set()
    q: deque = deque([0])
    visited.add(0)
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                q.append(v)
    assert len(visited) == grid.num_cells, "density_factor=2.0 で連結性が失われた"


def test_kruskal_low_density_factor_connected():
    """density_factor=0.1（低密度）でも全セル連結の perfect maze が生成される。"""
    img = _make_small_image(32, 32)
    result = generate_density_maze(img, grid_size=4, max_side=32, density_factor=0.1)
    gray = preprocess_image(img, max_side=32)
    grid = build_cell_grid(gray, result.grid_rows, result.grid_cols, density_factor=0.1)
    adj = build_spanning_tree(grid)
    visited: set = set()
    q: deque = deque([0])
    visited.add(0)
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                q.append(v)
    assert len(visited) == grid.num_cells, "density_factor=0.1 で連結性が失われた"


# ---------- エッジケース（全黒/全白画像） ----------

def test_density_maze_all_black_image_valid_path():
    """全黒画像でも迷路生成が正常完了し、解経路が1本存在すること。"""
    img = _make_all_black_image(32, 32)
    result = generate_density_maze(img, grid_size=4, max_side=32)
    assert result.entrance >= 0
    assert result.exit_cell >= 0
    path = result.solution_path
    assert len(path) >= 1
    assert path[0] == result.entrance
    assert path[-1] == result.exit_cell


def test_density_maze_all_white_image_valid_path():
    """全白画像でも迷路生成が正常完了し、解経路が1本存在すること。"""
    img = _make_all_white_image(32, 32)
    result = generate_density_maze(img, grid_size=4, max_side=32)
    assert result.entrance >= 0
    path = result.solution_path
    assert len(path) >= 1
    assert path[0] == result.entrance
    assert path[-1] == result.exit_cell


# ---------- BFS ソルバ正確性（独立検証） ----------

def test_solution_path_reachable_via_bfs():
    """BFS で entrance → exit_cell への到達を独立検証する。"""
    img = _make_small_image(32, 32)
    result = generate_density_maze(img, grid_size=4, max_side=32)
    gray = preprocess_image(img, max_side=32)
    grid = build_cell_grid(gray, result.grid_rows, result.grid_cols)
    adj = build_spanning_tree(grid)
    entrance = result.entrance
    exit_cell = result.exit_cell
    visited: set = {entrance}
    q: deque = deque([entrance])
    found = False
    while q:
        u = q.popleft()
        if u == exit_cell:
            found = True
            break
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                q.append(v)
    assert found, f"BFS で entrance({entrance}) から exit_cell({exit_cell}) に到達できない"


def test_solution_path_all_steps_adjacent():
    """解経路の各ステップが隣接セル間移動であること（BFS 隣接リストで独立検証）。"""
    img = _make_gradient_image(48, 48)
    result = generate_density_maze(img, grid_size=5, max_side=48)
    gray = preprocess_image(img, max_side=48)
    grid = build_cell_grid(gray, result.grid_rows, result.grid_cols)
    adj = build_spanning_tree(grid)
    path = result.solution_path
    for i in range(len(path) - 1):
        assert path[i + 1] in adj.get(path[i], []), (
            f"解経路ステップ {i}→{i+1}: セル {path[i]} と {path[i+1]} は非隣接"
        )


# --- Phase 2b: 画像適応ルーティング (find_image_guided_path) テスト ---

from backend.core.density.entrance_exit import find_image_guided_path
from backend.core.density.preprocess import preprocess_image


def _make_bright_dark_image(rows: int = 8, cols: int = 8) -> tuple:
    """
    左半分が暗(0)、右半分が明(255)の画像を返す。
    find_image_guided_path は明部を通る経路を選ぶはず。
    """
    arr = np.zeros((rows, cols), dtype=np.uint8)
    arr[:, cols // 2:] = 255
    img = Image.fromarray(arr, mode="L")
    return img, arr


def test_image_guided_path_bright_region_lower_cost():
    """
    明るいセル間エッジのコストが暗いセル間より低いことを検証。
    edge_cost = 1.0 - avg(lum_u, lum_v)
    → 明(1.0): cost ≈ 0.0、暗(0.0): cost ≈ 1.0
    """
    from backend.core.density.grid_builder import build_cell_grid
    from backend.core.density.maze_builder import build_spanning_tree

    img, _ = _make_bright_dark_image(8, 8)
    gray = preprocess_image(img, max_side=64)
    grid = build_cell_grid(gray, 4, 4)
    adj = build_spanning_tree(grid)
    flat_lum = grid.luminance.flatten()

    # 明るいエッジ: luminance ≈ 1.0 → cost ≈ 0.0
    bright_costs = []
    dark_costs = []
    for u, neighbors in adj.items():
        for v in neighbors:
            lum_u = float(flat_lum[u])
            lum_v = float(flat_lum[v])
            cost = 1.0 - (lum_u + lum_v) / 2.0
            avg = (lum_u + lum_v) / 2.0
            if avg > 0.7:
                bright_costs.append(cost)
            elif avg < 0.3:
                dark_costs.append(cost)

    # 明部と暗部のエッジが存在し、明部のコストが低い
    if bright_costs and dark_costs:
        assert min(bright_costs) < max(dark_costs), (
            f"明部コスト最小値({min(bright_costs):.3f}) >= 暗部コスト最大値({max(dark_costs):.3f})"
        )


def test_image_guided_path_entrance_exit_reachable():
    """use_image_guided=True で生成した迷路の入口→出口が BFS で到達可能。"""
    img = _make_small_image(32, 32)
    result = generate_density_maze(img, grid_size=4, max_side=32, use_image_guided=True)
    entrance = result.entrance
    exit_cell = result.exit_cell

    gray = preprocess_image(img, max_side=32)
    grid = build_cell_grid(gray, result.grid_rows, result.grid_cols)
    adj = build_spanning_tree(grid)

    visited: set = {entrance}
    q: deque = deque([entrance])
    found = False
    while q:
        u = q.popleft()
        if u == exit_cell:
            found = True
            break
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                q.append(v)
    assert found, f"use_image_guided: BFS で {entrance} → {exit_cell} に到達できない"


def test_image_guided_path_white_paint_increases_white_area():
    """
    show_solution=True (masterpiece白線モード) で PNG 出力した場合、
    show_solution=False より白ピクセル数が多くなる（経路が白く塗られるため）。
    """
    from PIL import Image as PilImage
    import io

    img = _make_small_image(64, 64)
    # show_solution=False（解経路なし）
    result_no_sol = generate_density_maze(
        img, grid_size=6, max_side=64, use_image_guided=True, show_solution=False
    )
    # show_solution=True（masterpiece白線）
    result_with_sol = generate_density_maze(
        img, grid_size=6, max_side=64, use_image_guided=True, show_solution=True
    )

    def count_white(png_bytes: bytes) -> int:
        im = PilImage.open(io.BytesIO(png_bytes)).convert("L")
        arr = np.array(im)
        return int(np.sum(arr > 200))

    white_no_sol = count_white(result_no_sol.png_bytes)
    white_with_sol = count_white(result_with_sol.png_bytes)
    assert white_with_sol >= white_no_sol, (
        f"白線塗りつぶし後({white_with_sol}px) が塗りつぶし前({white_no_sol}px) より少ない"
    )
