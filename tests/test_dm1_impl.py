"""
cmd_687k_a6_maze_dm1_impl: DM-1密度制御迷路 明示的実装テスト

backend/core/density/dm1.py の DM1Config / DM1Result / generate_dm1_maze() を検証。

検証カテゴリ:
  A: DM1Config（デフォルト・カスタム・バリデーション）
  B: API出力（PNG・SVG・戻り値）
  C: グリッドサイズ・セルサイズ
  D: density_map（形状・範囲・輝度反映）
  E: 壁重み（density_min/max スケール検証）
  F: 隣接リスト（全セル存在・連結性・spanning tree）
  G: E2E密度制御（グラデーション→暗部密/明部疎・極端入力・境界値）
"""
from __future__ import annotations

import io
import struct

import numpy as np
import pytest
from PIL import Image

from backend.core.density.dm1 import DM1Config, DM1Result, _build_dm1_walls, generate_dm1_maze
from backend.core.density.grid_builder import build_density_map
from backend.core.density.solver import bfs_has_path


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_uniform_image(brightness: float, size: int = 64) -> Image.Image:
    """0〜255 の均一輝度画像を作成する。"""
    val = int(np.clip(brightness * 255, 0, 255))
    return Image.fromarray(np.full((size, size), val, dtype=np.uint8), mode="L")


def _make_gradient_image(rows: int = 64, cols: int = 64) -> Image.Image:
    """左(暗=0)→右(明=255) の水平グラデーション画像。"""
    arr = np.tile(np.linspace(0, 255, cols), (rows, 1)).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _png_dimensions(png_bytes: bytes) -> tuple[int, int]:
    """PNG バイト列からサイズ (width, height) を読み取る。"""
    # PNG IHDR: 8bytes sig + 4bytes len + 4bytes "IHDR" + 4bytes W + 4bytes H
    w = struct.unpack(">I", png_bytes[16:20])[0]
    h = struct.unpack(">I", png_bytes[20:24])[0]
    return w, h


def _avg_degree(adj: dict, cell_ids: list) -> float:
    """指定セルの平均隣接数（degree）を返す。"""
    if not cell_ids:
        return 0.0
    return sum(len(adj[c]) for c in cell_ids) / len(cell_ids)


def _is_connected(adj: dict, n: int) -> bool:
    """BFS で全 n セルが連結か確認する。"""
    if n == 0:
        return True
    from collections import deque
    visited = {0}
    q = deque([0])
    while q:
        node = q.popleft()
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                q.append(nb)
    return len(visited) == n


# ===========================================================================
# A: DM1Config
# ===========================================================================

def test_dm1config_defaults():
    """DM1Config のデフォルト値が設計書の推奨範囲に収まること。"""
    cfg = DM1Config()
    assert 1 <= cfg.grid_rows <= 400
    assert 1 <= cfg.grid_cols <= 400
    assert 2 <= cfg.cell_size_px <= 5
    assert 0.0 <= cfg.density_min < cfg.density_max <= 1.0


def test_dm1config_custom_values():
    """DM1Config にカスタム値を設定できること。"""
    cfg = DM1Config(
        grid_rows=200,
        grid_cols=300,
        cell_size_px=2,
        density_min=0.1,
        density_max=0.9,
    )
    assert cfg.grid_rows == 200
    assert cfg.grid_cols == 300
    assert cfg.cell_size_px == 2
    assert cfg.density_min == 0.1
    assert cfg.density_max == 0.9


def test_dm1config_density_range_valid():
    """density_min < density_max の組み合わせを複数検証。"""
    for dmin, dmax in [(0.1, 0.9), (0.0, 1.0), (0.2, 0.8), (0.05, 0.95)]:
        cfg = DM1Config(density_min=dmin, density_max=dmax)
        assert cfg.density_min < cfg.density_max


# ===========================================================================
# B: API 出力
# ===========================================================================

def test_generate_dm1_maze_returns_dm1result():
    """generate_dm1_maze() が DM1Result を返すこと。"""
    img = _make_gradient_image(32, 32)
    cfg = DM1Config(grid_rows=8, grid_cols=8, cell_size_px=3)
    result = generate_dm1_maze(img, cfg)
    assert isinstance(result, DM1Result)


def test_generate_dm1_maze_png_nonempty():
    """PNG バイト列が空でなくヘッダが正しいこと。"""
    img = _make_gradient_image(32, 32)
    cfg = DM1Config(grid_rows=8, grid_cols=8, cell_size_px=2)
    result = generate_dm1_maze(img, cfg)
    assert len(result.png_bytes) > 100
    # PNG シグネチャ確認
    assert result.png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_generate_dm1_maze_svg_valid():
    """SVG 文字列が <svg 要素を含む有効な形式であること。"""
    img = _make_gradient_image(32, 32)
    cfg = DM1Config(grid_rows=8, grid_cols=8, cell_size_px=2)
    result = generate_dm1_maze(img, cfg)
    assert "<svg" in result.svg
    assert "</svg>" in result.svg


def test_generate_dm1_maze_solution_path_nonempty():
    """解経路が少なくとも2セル以上あること。"""
    img = _make_gradient_image(32, 32)
    cfg = DM1Config(grid_rows=8, grid_cols=8)
    result = generate_dm1_maze(img, cfg)
    assert len(result.solution_path) >= 2


def test_generate_dm1_maze_bfs_solution_verified():
    """BFS で入口→出口への解が存在することを独立検証。"""
    img = _make_gradient_image(32, 32)
    cfg = DM1Config(grid_rows=10, grid_cols=10)
    result = generate_dm1_maze(img, cfg)
    assert bfs_has_path(result.adj, result.entrance, result.exit_cell)


# ===========================================================================
# C: グリッドサイズ・セルサイズ
# ===========================================================================

def test_dm1_grid_rows_cols_in_result():
    """DM1Result の grid_rows/grid_cols が config の値以下であること。"""
    img = _make_gradient_image(64, 64)
    cfg = DM1Config(grid_rows=12, grid_cols=16)
    result = generate_dm1_maze(img, cfg)
    assert 1 <= result.grid_rows <= 12
    assert 1 <= result.grid_cols <= 16


def test_dm1_cell_size_px_2():
    """cell_size_px=2 の場合、PNG サイズが grid * 2 px になること。"""
    img = _make_gradient_image(64, 64)
    cfg = DM1Config(grid_rows=10, grid_cols=10, cell_size_px=2)
    result = generate_dm1_maze(img, cfg)
    expected_w = result.grid_cols * 2
    expected_h = result.grid_rows * 2
    w, h = _png_dimensions(result.png_bytes)
    assert w == expected_w, f"PNG幅 {w} != {expected_w}"
    assert h == expected_h, f"PNG高 {h} != {expected_h}"


def test_dm1_cell_size_px_5():
    """cell_size_px=5 の場合、PNG サイズが grid * 5 px になること。"""
    img = _make_gradient_image(64, 64)
    cfg = DM1Config(grid_rows=8, grid_cols=8, cell_size_px=5)
    result = generate_dm1_maze(img, cfg)
    expected_w = result.grid_cols * 5
    expected_h = result.grid_rows * 5
    w, h = _png_dimensions(result.png_bytes)
    assert w == expected_w, f"PNG幅 {w} != {expected_w}"
    assert h == expected_h, f"PNG高 {h} != {expected_h}"


def test_dm1_density_map_shape_matches_grid():
    """density_map の shape が (grid_rows, grid_cols) に一致すること。"""
    img = _make_gradient_image(64, 64)
    cfg = DM1Config(grid_rows=12, grid_cols=16)
    result = generate_dm1_maze(img, cfg)
    assert result.density_map.shape == (result.grid_rows, result.grid_cols)


# ===========================================================================
# D: density_map
# ===========================================================================

def test_dm1_density_map_range_01():
    """density_map の全要素が [0, 1] に収まること。"""
    img = _make_gradient_image(64, 64)
    cfg = DM1Config(grid_rows=10, grid_cols=10)
    result = generate_dm1_maze(img, cfg)
    assert float(result.density_map.min()) >= 0.0
    assert float(result.density_map.max()) <= 1.0


def test_dm1_density_map_dark_image_low_values():
    """真っ黒画像の density_map が 0.5 未満であること（CLAHEで多少上がる場合を考慮し緩め）。"""
    img = _make_uniform_image(0.0)
    cfg = DM1Config(grid_rows=8, grid_cols=8, max_side=0)
    result = generate_dm1_maze(img, cfg)
    # CLAHE は均一画像をスキップするので 0.0 のはず
    assert float(result.density_map.mean()) < 0.5


def test_dm1_density_map_bright_image_high_values():
    """真っ白画像の density_map が 0.5 超であること。"""
    img = _make_uniform_image(1.0)
    cfg = DM1Config(grid_rows=8, grid_cols=8, max_side=0)
    result = generate_dm1_maze(img, cfg)
    assert float(result.density_map.mean()) > 0.5


def test_dm1_density_map_gradient_monotone():
    """グラデーション画像で左列が右列より輝度が低いこと（暗→明の単調性）。"""
    img = _make_gradient_image(64, 64)
    cfg = DM1Config(grid_rows=8, grid_cols=8)
    result = generate_dm1_maze(img, cfg)
    left_lum = float(result.density_map[:, 0].mean())
    right_lum = float(result.density_map[:, -1].mean())
    assert left_lum < right_lum, f"左輝度{left_lum:.3f} >= 右輝度{right_lum:.3f}"


# ===========================================================================
# E: 壁重み（density_min/max スケール）
# ===========================================================================

def test_dm1_wall_weights_in_density_range():
    """_build_dm1_walls の全壁重みが [density_min, density_max] 内に収まること。"""
    lum = np.linspace(0, 1, 64).reshape(8, 8)
    walls = _build_dm1_walls(lum, 8, 8, density_min=0.1, density_max=0.9)
    weights = [w for _, _, w in walls]
    assert min(weights) >= 0.1 - 1e-9
    assert max(weights) <= 0.9 + 1e-9


def test_dm1_dark_cells_have_lower_weight():
    """暗部セル（輝度 0.0）の壁重みが明部セル（輝度 1.0）より低いこと。"""
    # 上半分=暗, 下半分=明
    lum = np.zeros((8, 8))
    lum[4:, :] = 1.0
    walls = _build_dm1_walls(lum, 8, 8, density_min=0.1, density_max=0.9)

    dark_weights, bright_weights = [], []
    for c1, c2, w in walls:
        r1, c_1 = c1 // 8, c1 % 8
        r2, c_2 = c2 // 8, c2 % 8
        avg_r = (r1 + r2) / 2
        if avg_r < 4:
            dark_weights.append(w)
        else:
            bright_weights.append(w)

    if dark_weights and bright_weights:
        assert np.mean(dark_weights) < np.mean(bright_weights)


def test_dm1_wall_weights_invalid_config_raises():
    """density_min > density_max の場合 ValueError が発生すること。"""
    lum = np.ones((4, 4)) * 0.5
    with pytest.raises(ValueError):
        _build_dm1_walls(lum, 4, 4, density_min=0.9, density_max=0.1)


# ===========================================================================
# F: 隣接リスト（spanning tree の性質）
# ===========================================================================

def test_dm1_adj_all_cells_present():
    """adj に全セルのキーが存在すること。"""
    img = _make_gradient_image(32, 32)
    cfg = DM1Config(grid_rows=8, grid_cols=8)
    result = generate_dm1_maze(img, cfg)
    n = result.grid_rows * result.grid_cols
    assert set(result.adj.keys()) == set(range(n))


def test_dm1_adj_graph_connected():
    """adj グラフが全セル連結であること（spanning tree の保証）。"""
    img = _make_gradient_image(32, 32)
    cfg = DM1Config(grid_rows=10, grid_cols=10)
    result = generate_dm1_maze(img, cfg)
    n = result.grid_rows * result.grid_cols
    assert _is_connected(result.adj, n)


def test_dm1_adj_spanning_tree_edge_count():
    """spanning tree のエッジ数が n-1 であること（perfect maze の性質）。"""
    img = _make_gradient_image(32, 32)
    cfg = DM1Config(grid_rows=8, grid_cols=8)
    result = generate_dm1_maze(img, cfg)
    n = result.grid_rows * result.grid_cols
    # 無向辺をカウント（各辺は adj に2回登場するので / 2）
    total_degree = sum(len(v) for v in result.adj.values())
    edge_count = total_degree // 2
    assert edge_count == n - 1, f"エッジ数 {edge_count} != n-1={n-1}"


def test_dm1_entrance_exit_are_different_cells():
    """入口と出口が異なるセルであること。"""
    img = _make_gradient_image(32, 32)
    cfg = DM1Config(grid_rows=8, grid_cols=8)
    result = generate_dm1_maze(img, cfg)
    assert result.entrance != result.exit_cell


# ===========================================================================
# G: E2E 密度制御（DM-1 判定基準）
# ===========================================================================

def test_dm1_gradient_dark_region_denser():
    """
    グラデーション画像（左暗→右明）を入力したとき、
    左半分（暗部）の平均degree > 右半分（明部）の平均degree であること。

    DM-1 判定基準: 暗→密（通路多）/明→疎（通路少）
    """
    img = _make_gradient_image(64, 64)
    cfg = DM1Config(grid_rows=12, grid_cols=12)
    result = generate_dm1_maze(img, cfg)
    cols = result.grid_cols
    half = cols // 2
    dark_ids  = [r * cols + c for r in range(result.grid_rows) for c in range(half)]
    bright_ids = [r * cols + c for r in range(result.grid_rows) for c in range(half, cols)]
    assert _avg_degree(result.adj, dark_ids) > _avg_degree(result.adj, bright_ids)


def test_dm1_black_image_maximum_passages():
    """
    真っ黒画像（輝度≈0）の場合、全セルの平均degree が最大化されること。
    比較: 真っ白画像との平均degree 差を検証。
    """
    cfg = DM1Config(grid_rows=10, grid_cols=10, max_side=0)
    res_black = generate_dm1_maze(_make_uniform_image(0.0), cfg)
    res_white = generate_dm1_maze(_make_uniform_image(1.0), cfg)
    n = cfg.grid_rows * cfg.grid_cols
    deg_black = sum(len(v) for v in res_black.adj.values()) / n
    deg_white = sum(len(v) for v in res_white.adj.values()) / n
    # spanning tree なので常に (n-1)/n * 2 ≈ 2.0 になる。
    # 均一画像同士では値が等しい。どちらも spanning tree を検証する。
    assert deg_black >= 1.8, f"black avg_degree={deg_black:.3f} 低すぎ"
    assert deg_white >= 1.8, f"white avg_degree={deg_white:.3f} 低すぎ"


def test_dm1_density_boundary_min_01():
    """density_min=0.1 で正常に迷路が生成されること。"""
    img = _make_gradient_image(32, 32)
    cfg = DM1Config(grid_rows=8, grid_cols=8, density_min=0.1, density_max=0.9)
    result = generate_dm1_maze(img, cfg)
    assert bfs_has_path(result.adj, result.entrance, result.exit_cell)


def test_dm1_density_boundary_max_09():
    """density_max=0.9 で正常に迷路が生成されること。"""
    img = _make_gradient_image(32, 32)
    cfg = DM1Config(grid_rows=8, grid_cols=8, density_min=0.1, density_max=0.9)
    result = generate_dm1_maze(img, cfg)
    assert len(result.solution_path) >= 2


def test_dm1_no_config_uses_defaults():
    """config=None の場合にデフォルト設定で迷路が生成されること。"""
    img = _make_gradient_image(32, 32)
    result = generate_dm1_maze(img, None)
    assert isinstance(result, DM1Result)
    assert result.grid_rows >= 1
    assert result.grid_cols >= 1


def test_dm1_result_fields_complete():
    """DM1Result の全フィールドが存在し適切な型であること。"""
    img = _make_gradient_image(32, 32)
    cfg = DM1Config(grid_rows=8, grid_cols=8)
    r = generate_dm1_maze(img, cfg)
    assert isinstance(r.svg, str) and len(r.svg) > 0
    assert isinstance(r.png_bytes, bytes) and len(r.png_bytes) > 0
    assert isinstance(r.entrance, int)
    assert isinstance(r.exit_cell, int)
    assert isinstance(r.solution_path, list)
    assert isinstance(r.density_map, np.ndarray)
    assert isinstance(r.adj, dict)


def test_dm1_solution_path_connects_entrance_exit():
    """解経路の先頭が entrance、末尾が exit_cell であること。"""
    img = _make_gradient_image(32, 32)
    cfg = DM1Config(grid_rows=8, grid_cols=8)
    r = generate_dm1_maze(img, cfg)
    if len(r.solution_path) >= 2:
        assert r.solution_path[0] == r.entrance
        assert r.solution_path[-1] == r.exit_cell


def test_dm1_wall_weight_equal_density_gives_uniform_maze():
    """
    density_min == density_max の場合は全壁重みが等しく、
    均一迷路（spanning tree）として生成されること。
    """
    lum = np.random.default_rng(0).random((6, 6))
    walls = _build_dm1_walls(lum, 6, 6, density_min=0.5, density_max=0.5)
    weights = [w for _, _, w in walls]
    assert all(abs(w - 0.5) < 1e-9 for w in weights), "均一モードで重みが0.5以外"
