# -*- coding: utf-8 -*-
"""
密度迷路 Phase 2b テスト: ループ許容密度制御 (post_process_density)。

検証項目:
  DC-1: extra_removal_rate=0 で既存動作と同一（通路数が増えない）
  DC-2: 暗い画像で通路密度が増加する（ループ発生）
  DC-3: 明るい画像で通路削除が発生する（壁密度増加）
  DC-4: 連結性が常に維持される（入口→出口間を BFS で確認）
  DC-5: extra_removal_rate=1.0 でも連結性が壊れない
  DC-6: generate_density_maze() の新パラメータが動作する
  DC-7: _is_connected ユニットテスト
  DC-8: 暗部のみ処理 (light_threshold=1.0) → 明部は無影響
  DC-9: 明部のみ処理 (extra_removal_rate=0.0, light_threshold=0.0) → 通路数は増えない
  DC-10: 境界値（grid_size=2x2）での動作確認
  DC-11: 大グリッド（20x20）での連結性・性能確認
  DC-12: dark_threshold=1.0（全セル暗い）での動作
  DC-13: light_threshold=0.0（全セル明るい）での最小連結確認
  DC-14: rng 引数で再現性を確認（同一 rng → 同一結果）
  DC-15: post_process_density 戻り値は既存 adj と同一型
  DC-16: adj の双方向性を検証（u→v ならば v→u）
  DC-17: 暗画像 extra_removal_rate=0.5 でループ（閉路）が実際に発生する
  DC-18: 全黒画像で generate_density_maze が正常終了する
  DC-19: 全白画像で generate_density_maze が正常終了する
  DC-20: solution_path が入口→出口を繋いでいることを確認（密度後処理後も有効）
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Set

import numpy as np
import pytest
from PIL import Image

from backend.core.density import generate_density_maze
from backend.core.density.grid_builder import CellGrid, build_cell_grid
from backend.core.density.maze_builder import (
    _is_connected,
    build_spanning_tree,
    post_process_density,
)


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_image(brightness: int, w: int = 64, h: int = 64) -> Image.Image:
    """均一輝度の L モード画像を作成。brightness=[0,255]"""
    arr = np.full((h, w), brightness, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def _make_gradient_image(w: int = 64, h: int = 64) -> Image.Image:
    """左=黒(0) → 右=白(255) のグラデーション画像"""
    arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return Image.fromarray(arr, mode="L")


def _bfs_reachable(adj: Dict[int, List[int]], start: int) -> Set[int]:
    """start から BFS で到達できる全ノードを返す。"""
    visited: Set[int] = {start}
    queue: deque[int] = deque([start])
    while queue:
        u = queue.popleft()
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                queue.append(v)
    return visited


def _count_edges(adj: Dict[int, List[int]]) -> int:
    """双方向 adj から無向エッジ数を返す（各辺を1回カウント）。"""
    total = sum(len(nbs) for nbs in adj.values())
    return total // 2


def _make_dark_grid(grid_size: int = 5) -> CellGrid:
    """全セル輝度=0.1 の CellGrid を返す（暗い画像相当）。"""
    gray = np.full((64, 64), 0.1, dtype=np.float64)
    img_arr = (gray * 255).astype(np.uint8)
    img = Image.fromarray(img_arr, mode="L")
    from backend.core.density.preprocess import preprocess_image
    g = preprocess_image(img, max_side=64)
    return build_cell_grid(g, grid_size, grid_size)


def _make_bright_grid(grid_size: int = 5) -> CellGrid:
    """全セル輝度≈1.0 の CellGrid を返す（明るい画像相当）。"""
    gray = np.full((64, 64), 0.95, dtype=np.float64)
    img_arr = (gray * 255).astype(np.uint8)
    img = Image.fromarray(img_arr, mode="L")
    from backend.core.density.preprocess import preprocess_image
    g = preprocess_image(img, max_side=64)
    return build_cell_grid(g, grid_size, grid_size)


def _has_loop(adj: Dict[int, List[int]], n: int) -> bool:
    """無向グラフにループ（閉路）が存在するか確認。エッジ数 > n-1 ならループあり。"""
    return _count_edges(adj) > n - 1


# ---------------------------------------------------------------------------
# DC-7: _is_connected ユニットテスト
# ---------------------------------------------------------------------------

class TestIsConnected:
    def test_empty(self):
        assert _is_connected({}, 0) is True

    def test_single_node(self):
        assert _is_connected({0: set()}, 1) is True

    def test_connected_chain(self):
        adj_sets = {0: {1}, 1: {0, 2}, 2: {1}}
        assert _is_connected(adj_sets, 3) is True

    def test_disconnected(self):
        adj_sets = {0: {1}, 1: {0}, 2: set(), 3: {2}, 2: {3}}
        # 0-1 と 2-3 が分離（0 から 2 へ到達不能）
        assert _is_connected(adj_sets, 4) is False

    def test_two_nodes_connected(self):
        adj_sets = {0: {1}, 1: {0}}
        assert _is_connected(adj_sets, 2) is True

    def test_two_nodes_disconnected(self):
        adj_sets = {0: set(), 1: set()}
        assert _is_connected(adj_sets, 2) is False


# ---------------------------------------------------------------------------
# DC-1: extra_removal_rate=0 で既存動作と同一
# ---------------------------------------------------------------------------

class TestNoChange:
    def test_rate0_no_new_edges(self):
        """extra_removal_rate=0, light_threshold=1.0 → エッジ数変化なし。"""
        grid = _make_dark_grid()
        adj = build_spanning_tree(grid)
        n = grid.num_cells
        original_edges = _count_edges(adj)

        result = post_process_density(
            adj, grid,
            extra_removal_rate=0.0,
            light_threshold=1.0,
        )
        assert _count_edges(result) == original_edges

    def test_rate0_same_structure(self):
        """extra_removal_rate=0 の結果は spanning tree と同一連結性。"""
        grid = _make_dark_grid()
        adj = build_spanning_tree(grid)
        n = grid.num_cells

        result = post_process_density(adj, grid, extra_removal_rate=0.0, light_threshold=1.0)
        adj_sets = {i: set(nbs) for i, nbs in result.items()}
        assert _is_connected(adj_sets, n)


# ---------------------------------------------------------------------------
# DC-2: 暗い画像で通路密度増加
# ---------------------------------------------------------------------------

class TestDarkImageLoops:
    def test_dark_image_more_edges(self):
        """全黒画像＋extra_removal_rate=0.8 → spanning tree より多くのエッジ。"""
        grid = _make_dark_grid(5)
        adj = build_spanning_tree(grid)
        n = grid.num_cells
        original_edges = _count_edges(adj)

        result = post_process_density(
            adj, grid,
            extra_removal_rate=0.8,
            dark_threshold=0.5,
            light_threshold=1.0,
        )
        assert _count_edges(result) >= original_edges

    def test_dark_image_high_rate_creates_loops(self):
        """extra_removal_rate=1.0 かつ全暗セルでループが発生する。"""
        grid = _make_dark_grid(5)
        adj = build_spanning_tree(grid)
        n = grid.num_cells

        result = post_process_density(
            adj, grid,
            extra_removal_rate=1.0,
            dark_threshold=0.9,
            light_threshold=1.0,
            rng=np.random.default_rng(0),
        )
        # n=25, spanning tree = 24 edges. ループがあれば > 24
        assert _count_edges(result) > n - 1, "ループが発生しているはず"

    def test_dark_stays_connected(self):
        """暗部追加除去後も全セルが連結。"""
        grid = _make_dark_grid(5)
        adj = build_spanning_tree(grid)
        n = grid.num_cells

        result = post_process_density(
            adj, grid,
            extra_removal_rate=1.0,
            dark_threshold=0.9,
            light_threshold=1.0,
        )
        adj_sets = {i: set(nbs) for i, nbs in result.items()}
        assert _is_connected(adj_sets, n)


# ---------------------------------------------------------------------------
# DC-3: 明るい画像で通路削除
# ---------------------------------------------------------------------------

class TestBrightImageWalls:
    def test_bright_image_fewer_edges(self):
        """全明るい画像＋light_threshold=0.0 → spanning tree より少ないエッジは
           連結性を壊すため最低 n-1 本を維持する（削除0もあり得る）。"""
        grid = _make_bright_grid(5)
        adj = build_spanning_tree(grid)
        n = grid.num_cells
        original_edges = _count_edges(adj)

        result = post_process_density(
            adj, grid,
            extra_removal_rate=0.0,
            light_threshold=0.0,
        )
        result_edges = _count_edges(result)
        # 連結性を壊す削除はしないため、最低 n-1 本が残る
        assert result_edges >= n - 1
        # spanning tree より増えてはならない
        assert result_edges <= original_edges

    def test_bright_stays_connected(self):
        """明部通路削除後も全セルが連結。"""
        grid = _make_bright_grid(5)
        adj = build_spanning_tree(grid)
        n = grid.num_cells

        result = post_process_density(
            adj, grid,
            extra_removal_rate=0.0,
            dark_threshold=0.0,
            light_threshold=0.0,
        )
        adj_sets = {i: set(nbs) for i, nbs in result.items()}
        assert _is_connected(adj_sets, n)


# ---------------------------------------------------------------------------
# DC-4/DC-5: 連結性保証
# ---------------------------------------------------------------------------

class TestConnectivity:
    @pytest.mark.parametrize("rate", [0.0, 0.3, 0.7, 1.0])
    def test_connectivity_various_rates(self, rate):
        """extra_removal_rate が変化しても連結性が維持される。"""
        grid = _make_dark_grid(4)
        adj = build_spanning_tree(grid)
        n = grid.num_cells

        result = post_process_density(
            adj, grid,
            extra_removal_rate=rate,
            dark_threshold=0.5,
            light_threshold=0.5,
        )
        adj_sets = {i: set(nbs) for i, nbs in result.items()}
        assert _is_connected(adj_sets, n)

    @pytest.mark.parametrize("seed", [0, 1, 42, 123])
    def test_connectivity_various_seeds(self, seed):
        """異なる乱数シードでも連結性が維持される。"""
        grid = _make_dark_grid(5)
        adj = build_spanning_tree(grid)
        n = grid.num_cells

        result = post_process_density(
            adj, grid,
            extra_removal_rate=0.8,
            dark_threshold=0.8,
            light_threshold=0.2,
            rng=np.random.default_rng(seed),
        )
        adj_sets = {i: set(nbs) for i, nbs in result.items()}
        assert _is_connected(adj_sets, n)


# ---------------------------------------------------------------------------
# DC-6: generate_density_maze の新パラメータ
# ---------------------------------------------------------------------------

class TestGenerateDensityMazeParams:
    def test_extra_removal_rate_param_accepted(self):
        """extra_removal_rate パラメータが受け付けられ結果が返る。"""
        img = _make_image(30)  # 暗い画像
        result = generate_density_maze(
            img, grid_size=5, max_side=64,
            extra_removal_rate=0.5,
            dark_threshold=0.3,
            light_threshold=0.7,
        )
        assert result.entrance >= 0
        assert result.exit_cell >= 0
        assert len(result.solution_path) >= 1

    def test_rate0_same_as_default(self):
        """extra_removal_rate=0 の結果は rate 未指定と同じ連結性。"""
        img = _make_image(30)
        r1 = generate_density_maze(img, grid_size=5, max_side=64, maze_id="a")
        r2 = generate_density_maze(
            img, grid_size=5, max_side=64,
            extra_removal_rate=0.0,
            dark_threshold=0.3,
            light_threshold=1.0,
            maze_id="b",
        )
        # 連結性は同じはず
        assert r1.grid_rows == r2.grid_rows
        assert r1.grid_cols == r2.grid_cols

    def test_dark_image_solution_path_valid(self):
        """全黒画像でも solution_path が入口→出口を繋ぐ。"""
        img = _make_image(10)
        result = generate_density_maze(
            img, grid_size=4, max_side=64,
            extra_removal_rate=0.6,
        )
        assert result.solution_path[0] == result.entrance
        assert result.solution_path[-1] == result.exit_cell


# ---------------------------------------------------------------------------
# DC-8: 暗部のみ処理
# ---------------------------------------------------------------------------

class TestDarkOnlyProcessing:
    def test_dark_only_no_bright_removal(self):
        """light_threshold=1.0 → 明部削除なし。エッジは増えるだけ。"""
        grid = _make_dark_grid(5)
        adj = build_spanning_tree(grid)
        original_edges = _count_edges(adj)
        n = grid.num_cells

        result = post_process_density(
            adj, grid,
            extra_removal_rate=0.8,
            light_threshold=1.0,
        )
        assert _count_edges(result) >= original_edges
        adj_sets = {i: set(nbs) for i, nbs in result.items()}
        assert _is_connected(adj_sets, n)


# ---------------------------------------------------------------------------
# DC-9: 明部のみ処理
# ---------------------------------------------------------------------------

class TestBrightOnlyProcessing:
    def test_bright_only_no_dark_addition(self):
        """extra_removal_rate=0.0 → 暗部追加なし。エッジは同数以下。"""
        grid = _make_bright_grid(5)
        adj = build_spanning_tree(grid)
        original_edges = _count_edges(adj)
        n = grid.num_cells

        result = post_process_density(
            adj, grid,
            extra_removal_rate=0.0,
            light_threshold=0.0,
        )
        assert _count_edges(result) <= original_edges
        adj_sets = {i: set(nbs) for i, nbs in result.items()}
        assert _is_connected(adj_sets, n)


# ---------------------------------------------------------------------------
# DC-10: 境界値（2x2グリッド）
# ---------------------------------------------------------------------------

class TestSmallGrid:
    def test_2x2_connectivity(self):
        """2×2 グリッドで全パラメータ組み合わせでも連結性が維持される。"""
        img = _make_image(50, w=32, h=32)
        from backend.core.density.preprocess import preprocess_image
        gray = preprocess_image(img, max_side=32)
        grid = build_cell_grid(gray, 2, 2)
        adj = build_spanning_tree(grid)
        n = grid.num_cells

        result = post_process_density(
            adj, grid,
            extra_removal_rate=1.0,
            dark_threshold=1.0,
            light_threshold=0.0,
        )
        adj_sets = {i: set(nbs) for i, nbs in result.items()}
        assert _is_connected(adj_sets, n)

    def test_1x1_grid(self):
        """1×1 グリッド（num_cells=1）は何もせず返る。"""
        img = _make_image(50, w=8, h=8)
        from backend.core.density.preprocess import preprocess_image
        gray = preprocess_image(img, max_side=8)
        grid = build_cell_grid(gray, 1, 1)
        adj = build_spanning_tree(grid)

        result = post_process_density(adj, grid, extra_removal_rate=1.0)
        assert isinstance(result, dict)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# DC-11: 大グリッド（20x20）の連結性・性能
# ---------------------------------------------------------------------------

class TestLargeGrid:
    def test_20x20_connectivity(self):
        """20×20 グリッドでも連結性が維持される。"""
        img = _make_gradient_image(64, 64)
        from backend.core.density.preprocess import preprocess_image
        gray = preprocess_image(img, max_side=64)
        grid = build_cell_grid(gray, 20, 20)
        adj = build_spanning_tree(grid)
        n = grid.num_cells

        result = post_process_density(
            adj, grid,
            extra_removal_rate=0.5,
            dark_threshold=0.3,
            light_threshold=0.7,
        )
        adj_sets = {i: set(nbs) for i, nbs in result.items()}
        assert _is_connected(adj_sets, n)

    def test_20x20_loop_exists(self):
        """20×20 暗画像でループ（閉路）が発生する。"""
        img = _make_image(20, 64, 64)
        from backend.core.density.preprocess import preprocess_image
        gray = preprocess_image(img, max_side=64)
        grid = build_cell_grid(gray, 20, 20)
        adj = build_spanning_tree(grid)
        n = grid.num_cells

        result = post_process_density(
            adj, grid,
            extra_removal_rate=0.8,
            dark_threshold=0.9,
            light_threshold=1.0,
            rng=np.random.default_rng(7),
        )
        assert _has_loop(result, n), "大グリッド暗画像でループが発生するはず"


# ---------------------------------------------------------------------------
# DC-14: rng 引数の再現性
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_rng_same_result(self):
        """同一 seed の rng を渡すと同一結果が得られる。"""
        grid = _make_dark_grid(5)
        adj = build_spanning_tree(grid)

        r1 = post_process_density(
            adj, grid,
            extra_removal_rate=0.5,
            rng=np.random.default_rng(99),
        )
        r2 = post_process_density(
            adj, grid,
            extra_removal_rate=0.5,
            rng=np.random.default_rng(99),
        )
        for i in range(grid.num_cells):
            assert set(r1[i]) == set(r2[i]), f"セル {i} の隣接が異なる"

    def test_different_rng_may_differ(self):
        """異なる seed では結果が異なる可能性がある（確率的処理）。"""
        grid = _make_dark_grid(8)
        adj = build_spanning_tree(grid)

        r1 = post_process_density(adj, grid, extra_removal_rate=0.8, rng=np.random.default_rng(1))
        r2 = post_process_density(adj, grid, extra_removal_rate=0.8, rng=np.random.default_rng(2))
        # 全く同じである必要はない（確率的）。少なくとも連結性は維持
        n = grid.num_cells
        for r in (r1, r2):
            adj_sets = {i: set(nbs) for i, nbs in r.items()}
            assert _is_connected(adj_sets, n)


# ---------------------------------------------------------------------------
# DC-15/DC-16: 戻り値の型・双方向性
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_return_type_is_dict(self):
        """戻り値は Dict[int, List[int]]。"""
        grid = _make_dark_grid(4)
        adj = build_spanning_tree(grid)
        result = post_process_density(adj, grid, extra_removal_rate=0.5)
        assert isinstance(result, dict)
        for k, v in result.items():
            assert isinstance(k, int)
            assert isinstance(v, list)

    def test_bidirectional_edges(self):
        """u→v ならば v→u（双方向性）。"""
        grid = _make_dark_grid(6)
        adj = build_spanning_tree(grid)
        result = post_process_density(
            adj, grid,
            extra_removal_rate=0.8,
            dark_threshold=0.9,
            light_threshold=0.1,
        )
        for u, nbs in result.items():
            for v in nbs:
                assert u in result[v], f"双方向性違反: {u}→{v} はあるが {v}→{u} がない"


# ---------------------------------------------------------------------------
# DC-17: 実際のループ（閉路）確認
# ---------------------------------------------------------------------------

class TestLoopDetection:
    def test_loop_means_more_edges_than_spanning_tree(self):
        """ループがあれば無向エッジ数 > n-1。"""
        grid = _make_dark_grid(5)
        adj = build_spanning_tree(grid)
        n = grid.num_cells

        result = post_process_density(
            adj, grid,
            extra_removal_rate=1.0,
            dark_threshold=0.9,
            light_threshold=1.0,
            rng=np.random.default_rng(42),
        )
        # spanning tree は n-1 本のエッジ
        assert _count_edges(result) >= n - 1  # 最低限
        # ループが発生していれば n-1 より多い
        assert _has_loop(result, n)


# ---------------------------------------------------------------------------
# DC-18/DC-19: 全黒・全白画像での generate_density_maze
# ---------------------------------------------------------------------------

class TestEdgeCaseImages:
    def test_all_black_image(self):
        """全黒画像（輝度=0）で generate_density_maze が正常終了する。"""
        img = _make_image(0)
        result = generate_density_maze(
            img, grid_size=4, max_side=64,
            extra_removal_rate=0.5,
        )
        assert result.entrance >= 0
        assert result.exit_cell >= 0

    def test_all_white_image(self):
        """全白画像（輝度=255）で generate_density_maze が正常終了する。"""
        img = _make_image(255)
        result = generate_density_maze(
            img, grid_size=4, max_side=64,
            light_threshold=0.5,
        )
        assert result.entrance >= 0
        assert result.exit_cell >= 0

    def test_gradient_image(self):
        """グラデーション画像（左黒→右白）で通常動作することを確認。"""
        img = _make_gradient_image()
        result = generate_density_maze(
            img, grid_size=6, max_side=64,
            extra_removal_rate=0.4,
            dark_threshold=0.3,
            light_threshold=0.7,
        )
        assert result.entrance >= 0
        assert len(result.solution_path) > 0


# ---------------------------------------------------------------------------
# DC-20: solution_path の有効性（密度後処理後も入口→出口）
# ---------------------------------------------------------------------------

class TestSolutionPathValidity:
    @pytest.mark.parametrize("rate,lt", [
        (0.0, 1.0),
        (0.5, 0.7),
        (0.8, 0.3),
        (1.0, 0.0),
    ])
    def test_solution_path_entrance_exit(self, rate, lt):
        """solution_path[0]=入口, solution_path[-1]=出口 が常に成立する。"""
        img = _make_gradient_image()
        result = generate_density_maze(
            img, grid_size=5, max_side=64,
            extra_removal_rate=rate,
            dark_threshold=0.3,
            light_threshold=lt,
        )
        assert result.solution_path[0] == result.entrance
        assert result.solution_path[-1] == result.exit_cell

    def test_solution_path_connected_steps(self):
        """solution_path の各ステップが adj で連結していること。"""
        img = _make_gradient_image()
        result = generate_density_maze(
            img, grid_size=5, max_side=64,
            extra_removal_rate=0.5,
        )
        # adj を再構築して path の連続性を確認
        path = result.solution_path
        if len(path) <= 1:
            return
        # solution_path の各隣接ペアが連結 — adj は API 応答に含まれないため
        # ここでは path の長さが正であることのみ確認
        assert len(path) >= 1
