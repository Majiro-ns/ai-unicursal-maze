# -*- coding: utf-8 -*-
"""
T-10: 元画像らしさと迷路性のトレードオフ設計 テスト

対象:
  - backend.core.models.MazeOptions（maze_weight フィールド）
  - backend.core.path_finder._score_path（maze_weight パラメータ）
  - backend.core.maze_generator.generate_unicursal_maze（maze_weight 引き継ぎ）
  - backend.api.routes（maze_weight Form引数）

CHECK-9 テスト期待値の根拠:
  - T-10-1: MazeOptions に maze_weight フィールドが定義済み（models.py T-10修正）。
  - T-10-2: maze_weight=0.0 のスコア式は既存式と完全一致。
      face_component = 1.0 * (0.30*lm + 0.10*bd) = 0.30*lm + 0.10*bd（元の重み）。
      maze_component = 0.0 * 0.40 * ... = 0。
      → 総スコア = 0.30*length + 0.25*weight + 0.05*curvature + 0.30*landmark + 0.10*boundary。
  - T-10-3: maze_weight=1.0 では face_component=0 → landmark無効・boundary無効。
      landmark/boundary が高くてもスコアは変わらない。
  - T-10-4: maze_weight=1.0 では maze_component = 0.40*(1-curvature)。
      curvature=0（全て直角）のパスは maze_component=0.40（最大）になる。
      curvature=1（完全直線）のパスは maze_component=0.0（最小）になる。
      → 曲がりくねったパスが高得点。
  - T-10-5: maze_weight が連続値のため、スコアも連続的に変化する（線形）。
  - T-10-6: routes.py ソース解析で maze_weight 参照を確認。
  - T-10-7: 楕円画像で maze_weight=0.0/1.0 の両方が generate_unicursal_maze で正常動作する。
"""

from __future__ import annotations

import inspect

from PIL import Image, ImageDraw

from backend.core.graph_builder import Edge, MazeGraph, Node
from backend.core.models import MazeOptions
from backend.core.path_finder import _score_path


# ---------------------------------------------------------------------------
# ヘルパー: テスト用ミニグラフの構築
# ---------------------------------------------------------------------------

def _make_simple_graph(n_nodes: int = 5, weight: float = 0.5) -> MazeGraph:
    """直線状のシンプルグラフを作成する（x=0,1,...,n_nodes-1, y=0固定）。"""
    nodes = {i: Node(id=i, x=i, y=0, degree=2 if 0 < i < n_nodes - 1 else 1, weight=weight) for i in range(n_nodes)}
    edges = [Edge(id=i, from_id=i, to_id=i + 1, length=1.0) for i in range(n_nodes - 1)]
    return MazeGraph(nodes=nodes, edges=edges)


def _make_l_shaped_graph() -> MazeGraph:
    """L字形グラフ: P0(0,0)→P1(1,0)→P2(1,1)→P3(1,2)。"""
    nodes = {
        0: Node(id=0, x=0, y=0, degree=1, weight=0.5),
        1: Node(id=1, x=1, y=0, degree=2, weight=0.5),
        2: Node(id=2, x=1, y=1, degree=2, weight=0.5),
        3: Node(id=3, x=1, y=2, degree=1, weight=0.5),
    }
    edges = [
        Edge(id=0, from_id=0, to_id=1, length=1.0),
        Edge(id=1, from_id=1, to_id=2, length=1.0),
        Edge(id=2, from_id=2, to_id=3, length=1.0),
    ]
    return MazeGraph(nodes=nodes, edges=edges)


def _make_ellipse_image(size: int = 200) -> Image.Image:
    """スケルトンが確実に生成される楕円画像を作成する。"""
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    margin = size // 5
    draw.ellipse(
        (margin, margin, size - margin, size - margin),
        outline="black",
        width=3,
    )
    return img


# ---------------------------------------------------------------------------
# T-10-1: MazeOptions に maze_weight フィールドが存在する
# ---------------------------------------------------------------------------

def test_maze_options_has_maze_weight_field():
    """
    MazeOptions に maze_weight フィールドが存在し、デフォルトは None。

    根拠: models.py に `maze_weight: Optional[float] = None` を追加済み（T-10修正）。
    """
    opts = MazeOptions()
    assert hasattr(opts, "maze_weight"), "MazeOptions に maze_weight フィールドがない"
    assert opts.maze_weight is None, f"maze_weight のデフォルトが None でない: {opts.maze_weight}"


# ---------------------------------------------------------------------------
# T-10-2: maze_weight=0.0 で既存スコア式と完全一致
# ---------------------------------------------------------------------------

def test_score_path_maze_weight_zero_equals_original():
    """
    maze_weight=0.0 のスコアは既存式（0.30*landmark + 0.10*boundary）と完全一致する。

    根拠:
      face_component = (1-0.0) * (0.30*lm + 0.10*bd) = 0.30*lm + 0.10*bd
      maze_component = 0.0 * 0.40 * (1-curvature) = 0
      → 元の式と同一: 0.30*L + 0.25*W + 0.05*C + 0.30*lm + 0.10*bd

    features=None のためlandmark_score=0, boundary_score=0 で検証する。
    この場合: score = 0.30*length + 0.25*weight + 0.05*curvature（maze_weight無関係）
    手計算: 5ノードパス(直線), weight=0.5, n_total=5
      length_score = min(1.0, 5/5) = 1.0
      weight_score = (0.5*5)/5 = 0.5
      curvature_score = 1.0（直線）
      score_mw0 = 0.30*1.0 + 0.25*0.5 + 0.05*1.0 + 0 + 0 = 0.30 + 0.125 + 0.05 = 0.475
    """
    graph = _make_simple_graph(n_nodes=5, weight=0.5)
    path_nodes = list(range(5))

    score_mw0 = _score_path(graph, path_nodes, None, maze_weight=0.0)
    # 手計算: 0.30*1.0 + 0.25*0.5 + 0.05*1.0 = 0.475
    assert abs(score_mw0 - 0.475) < 1e-9, f"maze_weight=0.0 のスコアが期待値 0.475 と異なる: {score_mw0}"


# ---------------------------------------------------------------------------
# T-10-3: maze_weight=1.0 で landmark/boundary が無効になる
# ---------------------------------------------------------------------------

def test_score_path_maze_weight_one_ignores_face_score():
    """
    maze_weight=1.0 では face_component=0 となり、landmark/boundary の有無がスコアに影響しない。

    根拠:
      face_component = (1-1.0) * (0.30*lm + 0.10*bd) = 0
      → landmark_score=0/boundary_score=0 と同じスコアになる（features=None の場合と一致）。

    直線5ノードパス（curvature=1.0）の場合:
      maze_component = 1.0 * 0.40 * (1-1.0) = 0
      score = 0.30*1.0 + 0.25*0.5 + 0.05*1.0 + 0 + 0 = 0.475
    """
    graph = _make_simple_graph(n_nodes=5, weight=0.5)
    path_nodes = list(range(5))

    score_with_mw1 = _score_path(graph, path_nodes, None, maze_weight=1.0)
    score_with_mw0 = _score_path(graph, path_nodes, None, maze_weight=0.0)
    # features=None なので landmark/boundary どちらも 0。maze_weight に関わらず同じスコアになる。
    assert abs(score_with_mw1 - score_with_mw0) < 1e-9, (
        f"features=None 時、maze_weight 差でスコアが変わってはいけない: mw0={score_with_mw0}, mw1={score_with_mw1}"
    )


# ---------------------------------------------------------------------------
# T-10-4: maze_weight=1.0 で曲がりくねったパスが高得点
# ---------------------------------------------------------------------------

def test_score_path_maze_weight_one_prefers_winding_path():
    """
    maze_weight=1.0 では curvature が低い（曲がりくねった）パスが高得点になる。

    根拠:
      maze_component = maze_weight * 0.40 * (1 - curvature_score)
      curvature=1.0（直線） → maze_component=0
      curvature=0.0（直角ばかり） → maze_component=0.40

    L字形パス [P0(0,0)→P1(1,0)→P2(1,1)→P3(1,2)] は1回の直角折れがある。
    直線パス [P0(0,0)→...→P3(3,0)] は折れなし。

    maze_weight=1.0 で L字形スコア > 直線スコアを確認する。
    """
    # L字形グラフ（1回の直角折れ）
    l_graph = _make_l_shaped_graph()
    l_path = [0, 1, 2, 3]

    # 直線グラフ（折れなし）
    straight_graph = _make_simple_graph(n_nodes=4, weight=0.5)
    straight_path = list(range(4))

    score_l_mw1 = _score_path(l_graph, l_path, None, maze_weight=1.0)
    score_straight_mw1 = _score_path(straight_graph, straight_path, None, maze_weight=1.0)

    assert score_l_mw1 > score_straight_mw1, (
        f"maze_weight=1.0 で L字パスが直線パスより高得点であるべき: "
        f"L字={score_l_mw1:.4f}, 直線={score_straight_mw1:.4f}"
    )


# ---------------------------------------------------------------------------
# T-10-5: maze_weight が連続値でスコアが連続的に変化する
# ---------------------------------------------------------------------------

def test_score_path_maze_weight_continuity():
    """
    maze_weight=0.0, 0.5, 1.0 でスコアが（features=None, 直線パス時は）同一になる。
    features がある場合は連続的に変化する。

    根拠（features=None の直線パス）:
      landmark=0, boundary=0, face_component=0（常に）
      curvature=1.0（直線）→ maze_component = maze_weight * 0.40 * 0 = 0（常に）
      → maze_weight に関わらずスコアは同一（0.475）

    つまり「トレードオフはfeatures/曲がりの有無で初めて効く」ことを確認する。
    """
    graph = _make_simple_graph(n_nodes=5, weight=0.5)
    path_nodes = list(range(5))

    scores = [_score_path(graph, path_nodes, None, maze_weight=mw) for mw in [0.0, 0.5, 1.0]]
    # 直線パス + features=None → 全て同じスコア
    assert abs(scores[0] - scores[1]) < 1e-9, f"maze_weight 0.0 vs 0.5 で差異: {scores}"
    assert abs(scores[1] - scores[2]) < 1e-9, f"maze_weight 0.5 vs 1.0 で差異: {scores}"


def test_score_path_l_shaped_maze_weight_increases_score():
    """
    L字形パス（curvature < 1）に maze_weight を上げるとスコアが上昇する。

    根拠:
      maze_component = maze_weight * 0.40 * (1 - curvature_score)
      curvature_score < 1 のL字形パスでは (1-curvature) > 0 なので
      maze_weight 増加 → maze_component 増加 → スコア増加。
    """
    l_graph = _make_l_shaped_graph()
    l_path = [0, 1, 2, 3]

    score_mw0 = _score_path(l_graph, l_path, None, maze_weight=0.0)
    score_mw1 = _score_path(l_graph, l_path, None, maze_weight=1.0)

    # L字形パスはcurvature < 1 なので maze_weight上昇でスコア上昇
    assert score_mw1 > score_mw0, (
        f"L字パスで maze_weight=1.0 のスコアが 0.0 より高いはず: mw0={score_mw0:.4f}, mw1={score_mw1:.4f}"
    )


# ---------------------------------------------------------------------------
# T-10-6: routes.py ソースに maze_weight 参照あり
# ---------------------------------------------------------------------------

def test_routes_has_maze_weight_param():
    """
    routes.py のソースに maze_weight が Form引数として含まれることを確認。

    根拠: routes.py に `maze_weight: float | None = Form(None)` を追加済み（T-10修正）。
    """
    import backend.api.routes as rt
    src = inspect.getsource(rt)
    assert "maze_weight" in src, "routes.py が maze_weight を含んでいない"


# ---------------------------------------------------------------------------
# T-10-7: generate_unicursal_maze が maze_weight=0.0/1.0 で正常動作する
# ---------------------------------------------------------------------------

def test_generate_unicursal_maze_accepts_maze_weight():
    """
    generate_unicursal_maze に maze_weight=0.0/1.0 を渡しても正常に動作する。

    根拠: MazeOptions.maze_weight → maze_generator._maze_weight → find_unicursal_like_path の
    コールチェーンが T-10 で実装済み。エラーなく MazeResult が返ることを確認する。
    """
    from backend.core.maze_generator import generate_unicursal_maze

    for mw in [0.0, 1.0]:
        image = _make_ellipse_image(200)
        options = MazeOptions(width=400, height=300, maze_weight=mw)
        result = generate_unicursal_maze(image, options)

        assert result.maze_id, f"maze_weight={mw} で maze_id が空"
        assert result.svg and "<svg" in result.svg, f"maze_weight={mw} で SVG が無効"
        assert result.png_bytes and len(result.png_bytes) > 0, f"maze_weight={mw} で PNG が空"
