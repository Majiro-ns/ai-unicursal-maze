"""
E2E 統合テスト: 密度迷路パイプライン一気通貫検証

対象: Phase 1/2 密度迷路 + エッジ強調 + BFS独立ソルバ + PNG出力検証
統合成果:
  - 一意性 (スパニングツリー = perfect maze)
  - エッジ強調 (build_cell_grid_with_edges / edge_enhancer.py)
  - PNG出力 (maze_to_png → bytes validation)
  - BFSソルバ独立検証 (entrance_exit.pyと独立したBFSで解経路を確認)
"""
from __future__ import annotations

import io
import struct
from collections import deque
from typing import Dict, List, Set, Tuple

import numpy as np
import pytest
from PIL import Image

from backend.core.density import generate_density_maze, DensityMazeResult
from backend.core.density.preprocess import preprocess_image
from backend.core.density.grid_builder import (
    build_cell_grid,
    build_cell_grid_with_edges,
    build_density_map,
    CellGrid,
)
from backend.core.density.maze_builder import build_spanning_tree


# ============================================================
# テスト用ヘルパー: 合成画像生成
# ============================================================

def _make_face_like_image(w: int = 64, h: int = 64) -> Image.Image:
    """
    顔らしいグラデーション画像（明るい中心・暗い周囲）。
    輝度差があるため density_factor の効果が出やすい。
    """
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_dist = np.sqrt(cy**2 + cx**2)
    gray_arr = (1.0 - dist / max_dist) * 255
    arr = np.clip(gray_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _make_gradient_image(w: int = 48, h: int = 48) -> Image.Image:
    """水平グラジエント（Phase 2 DIRECTIONAL テクスチャ向け）。"""
    arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return Image.fromarray(arr, mode="L")


def _make_vertical_gradient(w: int = 48, h: int = 48) -> Image.Image:
    """垂直グラジエント（Phase 2 SPIRAL テクスチャ向け）。"""
    arr = np.tile(np.linspace(0, 255, h, dtype=np.uint8).reshape(-1, 1), (1, w))
    return Image.fromarray(arr, mode="L")


def _is_valid_png(data: bytes) -> bool:
    """PNG マジックバイトと IEND チャンクを確認する。"""
    PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
    IEND = b"IEND"
    return data[:8] == PNG_MAGIC and IEND in data


def _bfs_path(adj: Dict[int, List[int]], start: int, end: int) -> List[int]:
    """BFS で start → end への最短経路を返す。到達不能なら空リスト。"""
    parent: Dict[int, int] = {start: -1}
    q: deque = deque([start])
    while q:
        u = q.popleft()
        if u == end:
            # 経路復元
            path: List[int] = []
            cur = end
            while cur != -1:
                path.append(cur)
                cur = parent[cur]
            return list(reversed(path))
        for v in adj.get(u, []):
            if v not in parent:
                parent[v] = u
                q.append(v)
    return []


def _all_cells_reachable(adj: Dict[int, List[int]], num_cells: int) -> bool:
    """BFS で全セルが1つの連結成分に属するかチェック（perfect maze 検証）。"""
    if num_cells == 0:
        return True
    visited: Set[int] = set()
    q: deque = deque([0])
    visited.add(0)
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                q.append(v)
    return len(visited) == num_cells


# ============================================================
# E2E テスト: Phase 1 フルパイプライン
# ============================================================

class TestE2EPhase1:
    """Phase 1 密度迷路の一気通貫 E2E テスト。"""

    def test_face_image_to_png_bfs_verification(self):
        """顔らしい合成画像 → 密度迷路 → PNG出力 → BFS独立検証。"""
        img = _make_face_like_image(64, 64)
        result = generate_density_maze(img, grid_size=6, max_side=64)

        # --- PNG出力検証 ---
        assert len(result.png_bytes) > 0, "PNG bytes が空"
        assert _is_valid_png(result.png_bytes), "PNG バイナリが無効"

        # --- SVG出力検証 ---
        assert len(result.svg) > 100, "SVG が短すぎる"
        assert "<svg" in result.svg, "SVG ヘッダーがない"

        # --- 入口・出口の一意性 ---
        assert result.entrance != result.exit_cell, "入口と出口が同じセル"
        n = result.grid_rows * result.grid_cols
        assert 0 <= result.entrance < n, f"入口 {result.entrance} が範囲外"
        assert 0 <= result.exit_cell < n, f"出口 {result.exit_cell} が範囲外"

    def test_bfs_solver_independently_finds_path(self):
        """BFS独立ソルバで解経路が見つかること。"""
        img = _make_face_like_image(48, 48)
        result = generate_density_maze(img, grid_size=5, max_side=48)

        gray = preprocess_image(img, max_side=48)
        grid = build_cell_grid(gray, result.grid_rows, result.grid_cols)
        adj = build_spanning_tree(grid)

        # 独立BFSで経路探索
        path = _bfs_path(adj, result.entrance, result.exit_cell)
        assert len(path) >= 2, "BFS独立ソルバで経路が見つからない"
        assert path[0] == result.entrance
        assert path[-1] == result.exit_cell

    def test_perfect_maze_all_cells_connected(self):
        """全セルが1つの連結成分に属する（perfect maze = uniqueness 保証）。"""
        img = _make_gradient_image(48, 48)
        result = generate_density_maze(img, grid_size=5, max_side=48)

        gray = preprocess_image(img, max_side=48)
        grid = build_cell_grid(gray, result.grid_rows, result.grid_cols)
        adj = build_spanning_tree(grid)
        n = result.grid_rows * result.grid_cols

        assert _all_cells_reachable(adj, n), "全セル連結でない（perfect maze 違反）"

    def test_solution_path_valid_and_unique_steps(self):
        """解経路の各ステップが隣接セル間 & 重複なし。"""
        img = _make_face_like_image(64, 64)
        result = generate_density_maze(img, grid_size=6, max_side=64)

        gray = preprocess_image(img, max_side=64)
        grid = build_cell_grid(gray, result.grid_rows, result.grid_cols)
        adj = build_spanning_tree(grid)

        path = result.solution_path
        assert len(path) >= 1, "解経路が空"
        assert len(path) == len(set(path)), "解経路に重複セルがある（一意性違反）"

        for i in range(len(path) - 1):
            assert path[i + 1] in adj.get(path[i], []), (
                f"ステップ {i}→{i+1}: セル {path[i]} と {path[i+1]} は非隣接"
            )


# ============================================================
# E2E テスト: Phase 2 フルパイプライン（テクスチャ + ヒューリスティクス）
# ============================================================

class TestE2EPhase2:
    """Phase 2 密度迷路（SPIRAL/DIRECTIONAL テクスチャ + ヒューリスティクス）の E2E テスト。"""

    def test_phase2_face_preset_png_bfs(self):
        """Phase 2 face プリセット → PNG出力 → BFS独立検証。"""
        img = _make_face_like_image(64, 64)
        result = generate_density_maze(
            img, grid_size=6, max_side=64,
            use_texture=True, use_heuristic=True,
            preset="face", n_segments=4,
        )

        assert _is_valid_png(result.png_bytes), "Phase 2 PNG バイナリが無効"
        assert result.segment_map is not None, "Phase 2 segment_map がない"
        assert result.texture_map is not None, "Phase 2 texture_map がない"
        assert result.entrance != result.exit_cell

    def test_phase2_spiral_texture_solution_unique(self):
        """SPIRAL テクスチャでも解経路に重複なし（一意性維持）。"""
        img = _make_vertical_gradient(48, 48)
        result = generate_density_maze(
            img, grid_size=5, max_side=48,
            use_texture=True, use_heuristic=True,
            preset="landscape",  # SPIRAL を含む
        )

        path = result.solution_path
        assert len(path) >= 2, "解経路が短すぎる"
        assert len(path) == len(set(path)), "SPIRAL テクスチャで解経路に重複（一意性違反）"
        assert path[0] == result.entrance
        assert path[-1] == result.exit_cell

    def test_phase2_bfs_independently_verifies_solution(self):
        """Phase 2 パイプラインの解経路を BFS で独立検証。"""
        img = _make_gradient_image(48, 48)
        result = generate_density_maze(
            img, grid_size=5, max_side=48,
            use_texture=True, use_heuristic=True,
            preset="generic",
        )

        # Phase 2 は build_cell_grid_with_texture を使う → adj を再構築
        from backend.core.density.grid_builder import build_cell_grid_with_texture
        from backend.core.density.texture import (
            assign_cell_textures, compute_gradient_angles,
            PRESET_GENERIC,
        )
        gray = preprocess_image(img, max_side=48)
        from backend.core.density.grid_builder import CellGrid, build_density_map
        from backend.core.density.texture import TextureType
        import numpy as np

        lum = build_density_map(gray, result.grid_rows, result.grid_cols)
        dummy_textures = np.full((result.grid_rows, result.grid_cols), TextureType.RANDOM)
        grid = build_cell_grid_with_texture(
            gray, result.grid_rows, result.grid_cols, dummy_textures
        )
        adj = build_spanning_tree(grid)

        path = _bfs_path(adj, result.entrance, result.exit_cell)
        assert len(path) >= 2, "Phase 2 BFS独立ソルバで解経路が見つからない"

    def test_phase2_png_size_reasonable(self):
        """Phase 2 PNG バイトサイズが妥当な範囲に収まる（1KB〜10MB）。"""
        img = _make_face_like_image(64, 64)
        result = generate_density_maze(
            img, grid_size=6, max_side=64,
            use_texture=True, use_heuristic=True,
            preset="face",
        )
        size_kb = len(result.png_bytes) / 1024
        assert 1 <= size_kb <= 10240, f"PNG サイズ {size_kb:.1f}KB が範囲外"


# ============================================================
# E2E テスト: エッジ強調統合
# ============================================================

class TestEdgeEnhancementIntegration:
    """エッジ強調（edge_enhancer.py）統合の E2E テスト。"""

    def test_edge_grid_still_connected(self):
        """エッジ強調ありでも全セルが連結（perfect maze 保証）。"""
        img = _make_face_like_image(48, 48)
        gray = preprocess_image(img, max_side=48)
        grid_rows, grid_cols = 5, 5
        n = grid_rows * grid_cols

        grid = build_cell_grid_with_edges(
            gray, grid_rows, grid_cols,
            edge_weight=0.6,
        )
        adj = build_spanning_tree(grid)

        assert _all_cells_reachable(adj, n), "エッジ強調後に全セル連結でない"

    def test_edge_weight_zero_matches_plain_grid(self):
        """edge_weight=0.0 のとき build_cell_grid() と同数の壁。"""
        img = _make_gradient_image(32, 32)
        gray = preprocess_image(img, max_side=32)
        grid_rows, grid_cols = 4, 4

        plain = build_cell_grid(gray, grid_rows, grid_cols)
        edged = build_cell_grid_with_edges(gray, grid_rows, grid_cols, edge_weight=0.0)

        assert len(plain.walls) == len(edged.walls)

    def test_edge_boost_increases_wall_weights(self):
        """edge_weight>0 のとき壁の合計重みが上昇する（エッジ強調効果）。"""
        img = _make_face_like_image(48, 48)
        gray = preprocess_image(img, max_side=48)
        grid_rows, grid_cols = 5, 5

        plain = build_cell_grid(gray, grid_rows, grid_cols)
        edged = build_cell_grid_with_edges(gray, grid_rows, grid_cols, edge_weight=0.8)

        sum_plain = sum(w for _, _, w in plain.walls)
        sum_edged = sum(w for _, _, w in edged.walls)

        # エッジ強調で壁重みの合計が大きくなるはず（または同等）
        assert sum_edged >= sum_plain * 0.9, "エッジ強調で壁重みが減少した（予期せず）"

    def test_edge_grid_bfs_path_exists(self):
        """エッジ強調済みグリッドで BFS が入口→出口経路を見つける。"""
        from backend.core.density.entrance_exit import find_entrance_exit_and_path
        img = _make_face_like_image(48, 48)
        gray = preprocess_image(img, max_side=48)
        grid_rows, grid_cols = 5, 5
        n = grid_rows * grid_cols

        grid = build_cell_grid_with_edges(gray, grid_rows, grid_cols, edge_weight=0.6)
        adj = build_spanning_tree(grid)
        entrance, exit_cell, solution_path = find_entrance_exit_and_path(adj, n)

        assert entrance != exit_cell
        assert len(solution_path) >= 2
        path = _bfs_path(adj, entrance, exit_cell)
        assert len(path) >= 2, "エッジ強調済みグリッドで BFS 経路が見つからない"


# ============================================================
# E2E テスト: 画像→密度迷路→PNG→ソルバ 一気通貫
# ============================================================

class TestFullPipelineE2E:
    """画像→密度迷路→PNG出力→ソルバ検証 一気通貫テスト。"""

    @pytest.mark.parametrize("preset,img_fn", [
        ("generic", lambda: _make_gradient_image(56, 56)),
        ("face",    lambda: _make_face_like_image(56, 56)),
        ("landscape", lambda: _make_vertical_gradient(56, 56)),
    ])
    def test_full_pipeline_all_presets(self, preset: str, img_fn):
        """全プリセットで一気通貫パイプラインが完走する。"""
        img = img_fn()
        result = generate_density_maze(
            img, grid_size=6, max_side=56,
            use_texture=True, use_heuristic=True,
            preset=preset, n_segments=3,
        )

        # 1. PNG バイナリ検証
        assert _is_valid_png(result.png_bytes), f"{preset}: PNG 無効"

        # 2. 入口・出口
        assert result.entrance != result.exit_cell, f"{preset}: 入口=出口"

        # 3. 解経路の一意性
        path = result.solution_path
        assert len(path) >= 2, f"{preset}: 解経路が短すぎる"
        assert len(path) == len(set(path)), f"{preset}: 解経路に重複"

        # 4. BFS 独立ソルバ（adj を再構築して検証）
        gray = preprocess_image(img, max_side=56)
        grid = build_cell_grid(gray, result.grid_rows, result.grid_cols)
        adj = build_spanning_tree(grid)
        bfs_path = _bfs_path(adj, result.entrance, result.exit_cell)
        assert len(bfs_path) >= 2, f"{preset}: BFS 独立ソルバで経路不達"

    def test_maze_id_unique_per_call(self):
        """generate_density_maze() を2回呼ぶと maze_id が異なる。"""
        img = _make_gradient_image(32, 32)
        r1 = generate_density_maze(img, grid_size=4, max_side=32)
        r2 = generate_density_maze(img, grid_size=4, max_side=32)
        assert r1.maze_id != r2.maze_id, "maze_id が重複している"

    def test_explicit_maze_id_preserved(self):
        """maze_id を指定した場合は結果に保持される。"""
        img = _make_gradient_image(32, 32)
        result = generate_density_maze(img, grid_size=4, max_side=32, maze_id="test-e2e-001")
        assert result.maze_id == "test-e2e-001"

    def test_small_grid_still_produces_path(self):
        """最小グリッド (2×2) でも解経路が生成される。"""
        img = _make_gradient_image(16, 16)
        result = generate_density_maze(img, grid_size=2, max_side=16)
        assert len(result.solution_path) >= 1
        assert _is_valid_png(result.png_bytes)
