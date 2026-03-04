# -*- coding: utf-8 -*-
"""
T-6: オイラー路エントリポイント テスト

対象モジュール: backend.core.euler_path
確認事項:
  1. build_euler_path が EulerPathResult を返す
  2. 単純な直線スケルトン → path_points が 3 点以上返る
  3. features=None でもクラッシュしない
  4. 空スケルトン → path_points が空リストで返る
  5. graph が MazeGraph 型で返る
  6. staged_generator.py が build_euler_path をインポートできる（結合確認）

CHECK-9 テスト期待値の根拠:
  - テスト2: 3ピクセル直線スケルトン(nodes=3, edges=2) → find_unicursal_like_path は
    全ノードを訪問するビームサーチ。直線上の全3点が探索される。手検証済み。
  - テスト4: 空スケルトン(nodes=0, edges=0) → find_unicursal_like_path は
    graph.nodes が空のとき即座に [] を返す（path_finder.py:197）。
"""

from __future__ import annotations

import numpy as np

from backend.core.euler_path import EulerPathResult, build_euler_path
from backend.core.graph_builder import MazeGraph


# ---------------------------------------------------------------------------
# T-6-1: build_euler_path が EulerPathResult を返す
# ---------------------------------------------------------------------------

def test_build_euler_path_returns_euler_path_result() -> None:
    """
    build_euler_path は EulerPathResult を返すこと。

    CHECK-9: EulerPathResult は dataclass。isinstance で型確認。
    """
    skeleton = np.zeros((10, 10), dtype=bool)
    skeleton[5, 3] = True
    skeleton[5, 4] = True
    skeleton[5, 5] = True

    result = build_euler_path(skeleton)

    assert isinstance(result, EulerPathResult)


# ---------------------------------------------------------------------------
# T-6-2: 直線スケルトン → path_points が 3 点以上返る
# ---------------------------------------------------------------------------

def test_build_euler_path_returns_path_for_simple_line() -> None:
    """
    3 ピクセルの直線スケルトンから build_euler_path を呼ぶと
    path_points に 3 点以上が返ること。

    CHECK-9: 3ピクセル直線 → graph.nodes=3, edges=2。
    find_unicursal_like_path のビームサーチは beam=1 でも3点を完全に訪問する。
    """
    skeleton = np.zeros((10, 10), dtype=bool)
    skeleton[5, 3] = True
    skeleton[5, 4] = True
    skeleton[5, 5] = True

    result = build_euler_path(skeleton)

    assert len(result.path_points) >= 3


# ---------------------------------------------------------------------------
# T-6-3: features=None でもクラッシュしない
# ---------------------------------------------------------------------------

def test_build_euler_path_without_features_does_not_crash() -> None:
    """
    features=None のとき build_euler_path はクラッシュしないこと。

    CHECK-9: euler_path.py は features が None のとき apply_feature_weights を
    スキップする。find_unicursal_like_path も features=None を受け付ける。
    """
    skeleton = np.zeros((15, 15), dtype=bool)
    for x in range(3, 12):
        skeleton[7, x] = True  # 9ピクセルの水平線

    result = build_euler_path(skeleton, features=None)

    assert isinstance(result, EulerPathResult)
    assert len(result.path_points) >= 1


# ---------------------------------------------------------------------------
# T-6-4: 空スケルトン → path_points が空リストで返る
# ---------------------------------------------------------------------------

def test_build_euler_path_empty_skeleton_returns_empty_path() -> None:
    """
    全 False の空スケルトンから build_euler_path を呼ぶと
    path_points が空リストで返ること。

    CHECK-9: skeleton_to_graph(empty) → nodes={}, edges=[]。
    find_unicursal_like_path は graph.nodes が空のとき [] を返す
    （path_finder.py L197: if not graph.nodes: return []）。
    """
    skeleton = np.zeros((10, 10), dtype=bool)

    result = build_euler_path(skeleton)

    assert result.path_points == []


# ---------------------------------------------------------------------------
# T-6-5: graph が MazeGraph 型で返る
# ---------------------------------------------------------------------------

def test_build_euler_path_returns_maze_graph() -> None:
    """
    build_euler_path の result.graph は MazeGraph 型であること。

    CHECK-9: EulerPathResult.graph は skeleton_to_graph の戻り値（MazeGraph）。
    呼び出し元（staged_generator.py）が solver/decorator で直接使う。
    """
    skeleton = np.zeros((10, 10), dtype=bool)
    skeleton[5, 3] = True
    skeleton[5, 4] = True
    skeleton[5, 5] = True

    result = build_euler_path(skeleton)

    assert isinstance(result.graph, MazeGraph)
    assert len(result.graph.nodes) >= 3
    assert len(result.graph.edges) >= 2


# ---------------------------------------------------------------------------
# T-6-6: staged_generator.py が build_euler_path をインポートできる
# ---------------------------------------------------------------------------

def test_staged_generator_imports_build_euler_path() -> None:
    """
    staged_generator モジュールが build_euler_path をインポートできること。
    T-6 の結合確認テスト。

    CHECK-9: staged_generator.py の import 文に
    `from .euler_path import build_euler_path` が含まれる。
    このテストは ImportError が発生しないことを確認する。
    """
    try:
        from backend.core.staged_generator import generate_staged_maze  # noqa: F401
        imported_ok = True
    except ImportError:
        imported_ok = False

    assert imported_ok, "staged_generator.py が build_euler_path をインポートできなかった"
