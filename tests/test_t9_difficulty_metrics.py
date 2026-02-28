# -*- coding: utf-8 -*-
"""
T-9: 難易度指標定義 テスト

対象:
  - backend.core.solver.SolveResult（turn_count / path_length フィールド）
  - backend.core.solver.solve_path（指標計算）
  - backend.core.models.MazeResult（T-9フィールド）
  - backend.core.maze_generator.generate_unicursal_maze（dead_end_count等の格納）
  - backend.api.routes（レスポンスへの追加）

CHECK-9 テスト期待値の根拠:
  - T-9-1〜3: dataclass フィールド存在確認。定義から明白。
  - T-9-4:    デフォルト None は dataclass 定義から明白。
  - T-9-5:    直線パス（3点が一直線）は方向変化ゼロ → turn_count == 0。
              solve_path の角度判定 abs(ang2 - ang1) > 1e-3 が成立しない。
  - T-9-6:    L字パス（直角1回）は ang2 - ang1 ≈ π/2 > 1e-3 → turn_count == 1。
  - T-9-7:    path_length = len(path_points) の直接カウント。
  - T-9-8〜10: 楕円画像で generate_unicursal_maze を実行。
              スケルトン生成が確実なため、各フィールドが非None int として返る。
              dead_end_count は degree==1 ノードの数 = グラフの端点数。
  - T-9-11:   routes.py ソースに turn_count / path_length / dead_end_count の参照あり。
"""

from __future__ import annotations

import inspect
import math

from PIL import Image, ImageDraw

from backend.core.models import MazeOptions, MazeResult
from backend.core.path_finder import PathPoint
from backend.core.solver import SolveResult, solve_path


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

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
# T-9-1〜3: SolveResult フィールド確認
# ---------------------------------------------------------------------------

def test_solve_result_has_turn_count_field():
    """
    SolveResult に turn_count フィールドが存在する。

    根拠: solver.py で `turn_count: int | None = None` と定義済み（T-9修正）。
    """
    r = SolveResult(has_solution=True, num_solutions=1, difficulty_score=0.0)
    assert hasattr(r, "turn_count"), "SolveResult に turn_count フィールドがない"


def test_solve_result_has_path_length_field():
    """
    SolveResult に path_length フィールドが存在する。

    根拠: solver.py で `path_length: int | None = None` と定義済み（T-9修正）。
    """
    r = SolveResult(has_solution=True, num_solutions=1, difficulty_score=0.0)
    assert hasattr(r, "path_length"), "SolveResult に path_length フィールドがない"


def test_maze_result_has_t9_fields():
    """
    MazeResult に T-9 フィールド（turn_count / path_length / dead_end_count）が存在する。

    根拠: models.py で T-9 フィールドを追加済み。
    """
    r = MazeResult(maze_id="x", svg="<svg/>", png_bytes=b"")
    assert hasattr(r, "turn_count"), "MazeResult に turn_count フィールドがない"
    assert hasattr(r, "path_length"), "MazeResult に path_length フィールドがない"
    assert hasattr(r, "dead_end_count"), "MazeResult に dead_end_count フィールドがない"


def test_maze_result_t9_fields_default_none():
    """
    MazeResult の T-9 フィールドのデフォルト値は None。

    根拠: models.py で `Optional[int] = None` と定義。
    """
    r = MazeResult(maze_id="x", svg="<svg/>", png_bytes=b"")
    assert r.turn_count is None, f"turn_count のデフォルトが None でない: {r.turn_count}"
    assert r.path_length is None, f"path_length のデフォルトが None でない: {r.path_length}"
    assert r.dead_end_count is None, f"dead_end_count のデフォルトが None でない: {r.dead_end_count}"


# ---------------------------------------------------------------------------
# T-9-5〜6: solve_path の turn_count 検証
# ---------------------------------------------------------------------------

def test_solve_path_straight_line_no_turns():
    """
    直線パス（3点が一直線上）の turn_count は 0。

    根拠: v1=(1,0), v2=(1,0) → ang1=ang2=0 → abs(ang2-ang1)=0 < 1e-3 → カウントされない。
    手計算: P0=(0,0), P1=(1,0), P2=(2,0) の場合
      v1=(1,0), ang1=atan2(0,1)=0
      v2=(1,0), ang2=atan2(0,1)=0
      abs(0 - 0) = 0 < 1e-3 → 曲がり角なし。
    """
    pts = [PathPoint(x=float(i), y=0.0) for i in range(5)]
    result = solve_path(pts)
    assert result.turn_count == 0, f"直線パスの turn_count が 0 でない: {result.turn_count}"


def test_solve_path_l_shaped_one_turn():
    """
    L字パス（直角折れ曲がり1回）の turn_count は 1。

    根拠:
      P0=(0,0), P1=(1,0), P2=(1,1)
      i=1: v1=(1,0) ang1=0, v2=(0,1) ang2=π/2
      abs(π/2 - 0) = π/2 ≈ 1.571 > 1e-3 → turn++
      → turn_count = 1
    """
    pts = [PathPoint(0, 0), PathPoint(1, 0), PathPoint(1, 1)]
    result = solve_path(pts)
    assert result.turn_count == 1, f"L字パスの turn_count が 1 でない: {result.turn_count}"


def test_solve_path_z_shaped_two_turns():
    """
    Z字パス（2回折れ曲がり）の turn_count は 2。

    根拠:
      P0=(0,0), P1=(1,0), P2=(1,1), P3=(2,1)
      i=1: ang1=0, ang2=π/2 → turn++
      i=2: v1=(0,1) ang1=π/2, v2=(1,0) ang2=0 → abs(0-π/2)=π/2 > 1e-3 → turn++
      → turn_count = 2
    """
    pts = [PathPoint(0, 0), PathPoint(1, 0), PathPoint(1, 1), PathPoint(2, 1)]
    result = solve_path(pts)
    assert result.turn_count == 2, f"Z字パスの turn_count が 2 でない: {result.turn_count}"


# ---------------------------------------------------------------------------
# T-9-7: solve_path の path_length 検証
# ---------------------------------------------------------------------------

def test_solve_path_path_length_equals_node_count():
    """
    path_length は path_points のノード数（len）と一致する。

    根拠: solve_path 内で `path_length=n`（n = len(path_points)）として格納。
    """
    for n in [3, 7, 10]:
        pts = [PathPoint(float(i), 0.0) for i in range(n)]
        result = solve_path(pts)
        assert result.path_length == n, (
            f"path_length が {n} でない（n={n}）: {result.path_length}"
        )


# ---------------------------------------------------------------------------
# T-9-8〜10: generate_unicursal_maze の T-9 フィールド確認
# ---------------------------------------------------------------------------

def _run_maze(size: int = 200):
    """楕円画像で generate_unicursal_maze を実行して結果を返す。"""
    from backend.core.maze_generator import generate_unicursal_maze
    image = _make_ellipse_image(size)
    options = MazeOptions(width=400, height=300)
    return generate_unicursal_maze(image, options)


def test_generate_unicursal_maze_returns_turn_count():
    """
    generate_unicursal_maze の戻り値に turn_count が含まれる（非None int）。

    根拠: maze_generator.py が solve_result.turn_count を MazeResult.turn_count に格納（T-9修正）。
    楕円画像はスケルトン生成が確実なため solve_result が None にならない。
    """
    result = _run_maze()
    assert result.turn_count is not None, "generate_unicursal_maze の turn_count が None"
    assert isinstance(result.turn_count, int), f"turn_count が int でない: {type(result.turn_count)}"
    assert result.turn_count >= 0, f"turn_count が 0 未満: {result.turn_count}"


def test_generate_unicursal_maze_returns_path_length():
    """
    generate_unicursal_maze の戻り値に path_length が含まれる（正の int）。

    根拠: maze_generator.py が solve_result.path_length を MazeResult.path_length に格納（T-9修正）。
    楕円画像では path_points >= 2 が保証されるため、path_length >= 2。
    """
    result = _run_maze()
    assert result.path_length is not None, "generate_unicursal_maze の path_length が None"
    assert isinstance(result.path_length, int), f"path_length が int でない: {type(result.path_length)}"
    assert result.path_length >= 2, f"path_length が 2 未満: {result.path_length}"


def test_generate_unicursal_maze_returns_dead_end_count():
    """
    generate_unicursal_maze の戻り値に dead_end_count が含まれる（非負 int）。

    根拠: maze_generator.py で `dead_end_count = sum(1 for node in graph.nodes.values()
    if node.degree == 1)` を計算し MazeResult に格納（T-9修正）。
    degree==1のノードはグラフの端点（袋小路）。
    """
    result = _run_maze()
    assert result.dead_end_count is not None, "generate_unicursal_maze の dead_end_count が None"
    assert isinstance(result.dead_end_count, int), (
        f"dead_end_count が int でない: {type(result.dead_end_count)}"
    )
    assert result.dead_end_count >= 0, f"dead_end_count が 0 未満: {result.dead_end_count}"


# ---------------------------------------------------------------------------
# T-9-11: routes.py ソースに T-9 フィールドの参照あり
# ---------------------------------------------------------------------------

def test_routes_response_includes_t9_fields():
    """
    routes.py のレスポンスに T-9 フィールドが含まれることをソース解析で確認する。

    根拠: routes.py は result.turn_count / result.path_length / result.dead_end_count を
    レスポンス dict に格納している必要がある。
    """
    import backend.api.routes as rt
    src = inspect.getsource(rt)
    assert "turn_count" in src, "routes.py が turn_count をレスポンスに含めていない"
    assert "path_length" in src, "routes.py が path_length をレスポンスに含めていない"
    assert "dead_end_count" in src, "routes.py が dead_end_count をレスポンスに含めていない"
