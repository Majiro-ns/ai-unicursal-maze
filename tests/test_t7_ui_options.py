# -*- coding: utf-8 -*-
"""
T-7: UIオプション整理 バックエンドフロー テスト

対象: backend.core.models.MazeOptions, backend.api.routes (フロー確認)
確認事項:
  1. MazeOptions に min_edge_size / spur_length フィールドが存在する
  2. MazeOptions のデフォルト値が None である（UI 側でデフォルトを持つ）
  3. staged_generator が options.min_edge_size を参照している（結合確認）
  4. staged_generator が options.spur_length を参照している（結合確認）
  5. maze_generator が options.min_edge_size / spur_length を参照している（結合確認）
  6. MazeOptions に min_edge_size=1, spur_length=0 を設定しても生成がクラッシュしない
  7. MazeOptions に min_edge_size=32, spur_length=12 を設定しても生成がクラッシュしない

CHECK-9 テスト期待値の根拠:
  - テスト1/2: dataclass フィールドを直接確認。Optional[int] = None はデータクラス定義から明白。
  - テスト3/4/5: ソースコード上で getattr(options, "min_edge_size", None) / getattr(options, "spur_length", None)
    を参照していることを確認（import 経由の結合テスト）。
  - テスト6/7: 最小・最大の境界値で実際にスケルトン処理を実行。クラッシュしなければ OK。
    remove_small_objects(min_size=1) は 1px 未満を除去（= 何も除去しない）。
    remove_small_objects(min_size=32) は 32px 未満を除去。どちらも ValueError は出ない。
"""

from __future__ import annotations

import inspect

import numpy as np

from backend.core.models import MazeOptions
from backend.core.skeleton import edges_to_skeleton, stabilize_skeleton


# ---------------------------------------------------------------------------
# T-7-1: MazeOptions に min_edge_size フィールドが存在する
# ---------------------------------------------------------------------------
def test_maze_options_has_min_edge_size():
    opts = MazeOptions()
    assert hasattr(opts, "min_edge_size"), "MazeOptions に min_edge_size フィールドがない"


# ---------------------------------------------------------------------------
# T-7-2: MazeOptions に spur_length フィールドが存在する
# ---------------------------------------------------------------------------
def test_maze_options_has_spur_length():
    opts = MazeOptions()
    assert hasattr(opts, "spur_length"), "MazeOptions に spur_length フィールドがない"


# ---------------------------------------------------------------------------
# T-7-3: MazeOptions のデフォルト値は None
# ---------------------------------------------------------------------------
def test_maze_options_defaults_are_none():
    opts = MazeOptions()
    assert opts.min_edge_size is None, f"min_edge_size のデフォルトが None でない: {opts.min_edge_size}"
    assert opts.spur_length is None, f"spur_length のデフォルトが None でない: {opts.spur_length}"


# ---------------------------------------------------------------------------
# T-7-4: staged_generator が min_edge_size / spur_length を参照している
# ---------------------------------------------------------------------------
def test_staged_generator_references_min_edge_size():
    import backend.core.staged_generator as sg
    src = inspect.getsource(sg)
    assert "min_edge_size" in src, "staged_generator.py が min_edge_size を参照していない"


def test_staged_generator_references_spur_length():
    import backend.core.staged_generator as sg
    src = inspect.getsource(sg)
    assert "spur_length" in src, "staged_generator.py が spur_length を参照していない"


# ---------------------------------------------------------------------------
# T-7-5: maze_generator が min_edge_size / spur_length を参照している
# ---------------------------------------------------------------------------
def test_maze_generator_references_min_edge_size():
    import backend.core.maze_generator as mg
    src = inspect.getsource(mg)
    assert "min_edge_size" in src, "maze_generator.py が min_edge_size を参照していない"


def test_maze_generator_references_spur_length():
    import backend.core.maze_generator as mg
    src = inspect.getsource(mg)
    assert "spur_length" in src, "maze_generator.py が spur_length を参照していない"


# ---------------------------------------------------------------------------
# T-7-6: 境界値 min_edge_size=1, spur_length=0 でスケルトン処理がクラッシュしない
# ---------------------------------------------------------------------------
def test_edges_to_skeleton_min_edge_size_1():
    # 20x20 の合成エッジマップ
    edges = np.zeros((20, 20), dtype=bool)
    edges[5, 2:18] = True  # 16px 水平線

    result = edges_to_skeleton(edges, min_edge_size=1)
    assert result.dtype == bool, "戻り値が bool でない"
    assert result.shape == (20, 20), "形状が変わっている"


def test_stabilize_skeleton_spur_length_0():
    edges = np.zeros((20, 20), dtype=bool)
    edges[5, 2:18] = True  # 16px 水平線

    result = stabilize_skeleton(edges, min_edge_size=1, spur_length=0, min_component_size=1)
    assert result.dtype == bool, "戻り値が bool でない"
    assert result.shape == (20, 20), "形状が変わっている"


# ---------------------------------------------------------------------------
# T-7-7: 境界値 min_edge_size=32, spur_length=12 でスケルトン処理がクラッシュしない
# ---------------------------------------------------------------------------
def test_edges_to_skeleton_min_edge_size_32():
    # 大きめのエッジマップ: 40px 線 + 5px ノイズ
    edges = np.zeros((50, 50), dtype=bool)
    edges[10, 5:45] = True   # 40px 水平線（min_size=32 では除去されない）
    edges[30, 20:25] = True  # 5px ノイズ（min_size=32 では除去される）

    result = edges_to_skeleton(edges, min_edge_size=32)
    assert result.dtype == bool, "戻り値が bool でない"
    assert result.shape == (50, 50), "形状が変わっている"
    # 主線はスケルトンに残る（40px > 32）
    assert result.any(), "40px の主線がスケルトンに残っていない"


def test_stabilize_skeleton_spur_length_12():
    edges = np.zeros((50, 50), dtype=bool)
    edges[10, 5:45] = True  # 40px 水平線

    result = stabilize_skeleton(edges, min_edge_size=8, spur_length=12, min_component_size=5)
    assert result.dtype == bool, "戻り値が bool でない"
    assert result.shape == (50, 50), "形状が変わっている"
