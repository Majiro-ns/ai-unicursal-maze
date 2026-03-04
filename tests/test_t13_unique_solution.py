"""
tests/test_t13_unique_solution.py
===================================
T-13: 一意解迷路ロジック具体化テスト

対象関数:
  - backend.core.solver.prune_edges()
  - backend.core.solver.force_unique_solution()
  - backend.core.solver._build_subgraph()

CHECK-9 根拠:

  test_prune_edges_diamond_graph:
    ダイヤモンドグラフ (0→1→3 / 0→2→3 の2経路) で prune_edges を実行。
    1エッジ削除後、count_solutions == 1 となる。

  test_prune_edges_linear_graph_no_change:
    直線グラフ (0→1→2, 1解) は prune_edges で変化しない（削除不要）。

  test_force_unique_diamond_becomes_unique:
    ダイヤモンドグラフで force_unique_solution → is_unique=True、エッジ数が減少。

  test_force_unique_already_unique_linear:
    直線グラフ (1解) は force_unique_solution でそのまま返される (is_unique=True)。

  test_force_unique_no_solution_graph:
    非連結グラフ (0解) は force_unique_solution で is_unique=False のまま。

  test_force_unique_three_path_graph:
    3経路グラフ (0→1→4 / 0→2→4 / 0→3→4) で force_unique_solution → is_unique=True。

  test_force_unique_preserves_path_existence:
    一意化後のグラフで count_solutions >= 1 (経路が残る)。

作成: 足軽6 cmd_285k_maze_T13
"""
from __future__ import annotations

from backend.core.graph_builder import Edge, MazeGraph, Node
from backend.core.solver import (
    _build_subgraph,
    count_solutions_on_graph,
    force_unique_solution,
    prune_edges,
)


# ── ヘルパー ──────────────────────────────────────────────────────────────────

def _make_diamond_graph() -> MazeGraph:
    """
    ダイヤモンド（2経路）グラフ:
      0 ── 1
      |    |
      2 ── 3
    経路: 0→1→3 と 0→2→3 の 2 本。
    """
    nodes = {
        0: Node(id=0, x=0, y=0, degree=2, weight=None),
        1: Node(id=1, x=1, y=0, degree=2, weight=None),
        2: Node(id=2, x=0, y=1, degree=2, weight=None),
        3: Node(id=3, x=1, y=1, degree=2, weight=None),
    }
    edges = [
        Edge(id=0, from_id=0, to_id=1, length=1.0),  # 上辺
        Edge(id=1, from_id=0, to_id=2, length=1.0),  # 左辺
        Edge(id=2, from_id=1, to_id=3, length=1.0),  # 右辺
        Edge(id=3, from_id=2, to_id=3, length=1.0),  # 下辺
    ]
    return MazeGraph(nodes=nodes, edges=edges)


def _make_linear_graph() -> MazeGraph:
    """
    直線（1経路）グラフ: 0 ── 1 ── 2
    """
    nodes = {
        0: Node(id=0, x=0, y=0, degree=1, weight=None),
        1: Node(id=1, x=1, y=0, degree=2, weight=None),
        2: Node(id=2, x=2, y=0, degree=1, weight=None),
    }
    edges = [
        Edge(id=0, from_id=0, to_id=1, length=1.0),
        Edge(id=1, from_id=1, to_id=2, length=1.0),
    ]
    return MazeGraph(nodes=nodes, edges=edges)


def _make_three_path_graph() -> MazeGraph:
    """
    3経路グラフ:
      0 ──→ 1 ──→ 4
      |           ↑
      └──→ 2 ─── ┘
      |           ↑
      └──→ 3 ─── ┘
    経路: 0→1→4, 0→2→4, 0→3→4 の 3 本。
    """
    nodes = {
        0: Node(id=0, x=0, y=2, degree=3, weight=None),
        1: Node(id=1, x=1, y=0, degree=2, weight=None),
        2: Node(id=2, x=1, y=2, degree=2, weight=None),
        3: Node(id=3, x=1, y=4, degree=2, weight=None),
        4: Node(id=4, x=2, y=2, degree=3, weight=None),
    }
    edges = [
        Edge(id=0, from_id=0, to_id=1, length=1.0),
        Edge(id=1, from_id=0, to_id=2, length=1.0),
        Edge(id=2, from_id=0, to_id=3, length=1.0),
        Edge(id=3, from_id=1, to_id=4, length=1.0),
        Edge(id=4, from_id=2, to_id=4, length=1.0),
        Edge(id=5, from_id=3, to_id=4, length=1.0),
    ]
    return MazeGraph(nodes=nodes, edges=edges)


# ── prune_edges テスト ────────────────────────────────────────────────────────

def test_prune_edges_diamond_graph() -> None:
    """
    T-13 CHECK-9: ダイヤモンドグラフ（2経路）で prune_edges が 1 解に削減することを確認。

    根拠: ダイヤモンドグラフは 0→1→3 と 0→2→3 の 2 経路を持つ。
          prune_edges は解数を減らしつつ経路を残すエッジを削除する。
          結果: count_solutions == 1（一意解）。
    """
    graph = _make_diamond_graph()
    start, goal = 0, 3

    # 初期状態は 2 解
    n_before = count_solutions_on_graph(graph, start, goal, max_solutions=2)
    assert n_before == 2, f"初期解数が 2 でない: {n_before}"

    result = prune_edges(graph, start, goal)

    n_after = count_solutions_on_graph(result, start, goal, max_solutions=2)
    assert n_after == 1, f"prune_edges 後の解数が 1 でない: {n_after}"


def test_prune_edges_linear_graph_no_change() -> None:
    """
    T-13 CHECK-9: 直線グラフ（1解）は prune_edges で変化しない。

    根拠: 1解のグラフは解数 <= 1 なので prune_edges は何も削除しない。
          エッジ数が保たれることを確認。
    """
    graph = _make_linear_graph()
    start, goal = 0, 2

    result = prune_edges(graph, start, goal)

    n = count_solutions_on_graph(result, start, goal, max_solutions=2)
    assert n == 1, f"直線グラフの prune_edges 後の解数が 1 でない: {n}"
    assert len(result.edges) == len(graph.edges), (
        f"直線グラフのエッジ数が変化した: {len(result.edges)} != {len(graph.edges)}"
    )


# ── force_unique_solution テスト ──────────────────────────────────────────────

def test_force_unique_diamond_becomes_unique() -> None:
    """
    T-13 CHECK-9: ダイヤモンドグラフで force_unique_solution → is_unique=True。

    根拠: ダイヤモンドグラフは 2 解。force_unique_solution が枝削減を実施し
          1 解に変換する。変換後 is_unique=True を確認。
    """
    graph = _make_diamond_graph()
    result_graph, is_unique = force_unique_solution(graph, start=0, goal=3)

    assert is_unique is True, "force_unique_solution 後 is_unique が True でない"
    # エッジ数が元より少ない（1本以上削除された）
    assert len(result_graph.edges) < len(graph.edges), (
        f"force_unique_solution でエッジが削除されていない: "
        f"{len(result_graph.edges)} >= {len(graph.edges)}"
    )


def test_force_unique_already_unique_linear() -> None:
    """
    T-13 CHECK-9: 直線グラフ（1解）は force_unique_solution でそのまま返される。

    根拠: n_initial == 1 → 枝削減をスキップし同じグラフを返す。
          is_unique=True、エッジ数変化なし。
    """
    graph = _make_linear_graph()
    result_graph, is_unique = force_unique_solution(graph, start=0, goal=2)

    assert is_unique is True, "直線グラフで is_unique が True でない"
    assert len(result_graph.edges) == len(graph.edges), (
        f"直線グラフのエッジ数が変化した: {len(result_graph.edges)}"
    )


def test_force_unique_no_solution_graph() -> None:
    """
    T-13 CHECK-9: 非連結グラフ（0解）は force_unique_solution で is_unique=False。

    根拠: start→goal の経路がない（0解）。
          n_initial == 0 → 枝削減スキップ。is_unique=False を返す。
    """
    nodes = {
        0: Node(id=0, x=0, y=0, degree=0, weight=None),
        1: Node(id=1, x=1, y=0, degree=0, weight=None),
    }
    graph = MazeGraph(nodes=nodes, edges=[])

    _, is_unique = force_unique_solution(graph, start=0, goal=1)
    assert is_unique is False, "非連結グラフで is_unique が True になっている"


def test_force_unique_three_path_graph() -> None:
    """
    T-13 CHECK-9: 3経路グラフで force_unique_solution → is_unique=True。

    根拠: 3経路グラフ (0→1→4 / 0→2→4 / 0→3→4) は解数=3。
          prune_edges が2回のエッジ削除で解数を1に削減する。
          force_unique_solution 後 is_unique=True を確認。
    """
    graph = _make_three_path_graph()
    start, goal = 0, 4

    _, is_unique = force_unique_solution(graph, start=start, goal=goal)
    assert is_unique is True, "3経路グラフで force_unique_solution が is_unique=True にならない"


def test_force_unique_preserves_path_existence() -> None:
    """
    T-13 CHECK-9: 一意化後のグラフで経路が少なくとも1本残ることを確認。

    根拠: prune_edges は「解数が減って、かつ解が残る」条件でのみエッジを削除する。
          そのため、一意化後も必ず start→goal の経路が存在する。
    """
    graph = _make_diamond_graph()
    result_graph, is_unique = force_unique_solution(graph, start=0, goal=3)

    n = count_solutions_on_graph(result_graph, 0, 3, max_solutions=2)
    assert n >= 1, f"force_unique_solution 後に経路が消滅した（n={n}）"


def test_build_subgraph_filters_edges() -> None:
    """
    T-13 CHECK-9: _build_subgraph が指定エッジのみを持つサブグラフを返すことを確認。

    根拠: ダイヤモンドグラフ（4エッジ）から ID={0,1} のエッジのみ抽出。
          結果グラフは 2 エッジを持つ。
    """
    graph = _make_diamond_graph()
    active = {0, 1}  # エッジ id=0 (0→1) と id=1 (0→2) のみ

    subgraph = _build_subgraph(graph, active)

    assert len(subgraph.edges) == 2, (
        f"サブグラフのエッジ数が 2 でない: {len(subgraph.edges)}"
    )
    edge_ids = {e.id for e in subgraph.edges}
    assert edge_ids == {0, 1}, f"サブグラフのエッジ ID が一致しない: {edge_ids}"
    # ノードは元グラフと同じ参照
    assert len(subgraph.nodes) == 4, f"ノード数が変化した: {len(subgraph.nodes)}"
