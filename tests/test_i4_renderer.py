"""
tests/test_i4_renderer.py
=========================
G1レンダラー (DM-1 Part B: cmd_376k_a7) のテスト。

テスト対象:
  - I4MazeResult: dataclass インスタンス化
  - render_i4_maze(): PNG/SVG ファイル出力
  - _g1_thickness(): G1 線幅計算
  - PNG 出力サイズ検証
  - G1 暗部≒太線・明部≒細線の動作確認
  - density_maze_result_to_i4(): 既存パイプライン変換

TI-1〜TI-10: 10 テスト
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from backend.core.density.i4_renderer import (
    I4MazeResult,
    _g1_thickness,
    _norm_key,
    _build_solution_set,
    _render_png,
    density_maze_result_to_i4,
    render_i4_maze,
)


# ---------------------------------------------------------------------------
# ヘルパー: 最小 I4MazeResult を生成
# ---------------------------------------------------------------------------

def _make_tiny_result(
    rows: int = 4,
    cols: int = 4,
    cell_size: int = 5,
    dark_density: float = 0.9,
    bright_density: float = 0.1,
) -> I4MazeResult:
    """テスト用の最小 I4MazeResult。

    グリッド構成:
      - 全セルが一列に連結（左上→右下のジグザグ解経路）
      - 上半分が暗部（density=dark_density）、下半分が明部（density=bright_density）
      - 壁: 解経路以外の全隣接ペア
    """
    # 解経路: 左→右にジグザグ（行0→右端→行1→左端→行2→右端→...）
    path: list = []
    for r in range(rows):
        row_cells = [(r, c) for c in range(cols)]
        if r % 2 == 1:
            row_cells = list(reversed(row_cells))
        path.extend(row_cells)

    # 解経路の隣接ペアを open passages として登録
    open_passages: set = set()
    for i in range(len(path) - 1):
        open_passages.add(_norm_key(path[i], path[i + 1]))

    # 残り全ての隣接ペアを walls に
    walls: set = set()
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                key = _norm_key((r, c), (r, c + 1))
                if key not in open_passages:
                    walls.add(key)
            if r + 1 < rows:
                key = _norm_key((r, c), (r + 1, c))
                if key not in open_passages:
                    walls.add(key)

    # density_map: 上半分=暗部, 下半分=明部
    density = np.zeros((rows, cols), dtype=np.float64)
    half = rows // 2
    density[:half, :] = dark_density
    density[half:, :] = bright_density

    return I4MazeResult(
        grid_width=cols,
        grid_height=rows,
        cell_size_px=cell_size,
        walls=walls,
        solution_path=path,
        density_map=density,
        entrance=path[0],
        exit=path[-1],
    )


# ---------------------------------------------------------------------------
# TI-1: I4MazeResult dataclass インスタンス化
# ---------------------------------------------------------------------------

def test_ti1_i4_maze_result_instantiation():
    """TI-1: I4MazeResult が正しくインスタンス化できる。

    根拠: dataclass なので全フィールドが必須。型チェックで基本構造を確認。
    """
    result = _make_tiny_result()
    assert result.grid_width == 4
    assert result.grid_height == 4
    assert result.cell_size_px == 5
    assert isinstance(result.walls, set)
    assert isinstance(result.solution_path, list)
    assert len(result.solution_path) == 16   # 4×4 全セル通過
    assert isinstance(result.density_map, np.ndarray)
    assert result.density_map.shape == (4, 4)
    assert isinstance(result.entrance, tuple)
    assert isinstance(result.exit, tuple)


# ---------------------------------------------------------------------------
# TI-2: PNG ファイルが出力される
# ---------------------------------------------------------------------------

def test_ti2_render_png_creates_file():
    """TI-2: render_i4_maze() が PNG ファイルを生成する。

    根拠: output_path に PNG が存在し、ファイルサイズ > 0 であること。
    """
    result = _make_tiny_result(rows=4, cols=4, cell_size=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test.png")
        returned = render_i4_maze(result, out_path, format="png")
        assert os.path.isfile(returned), f"PNG が生成されていない: {returned}"
        assert os.path.getsize(returned) > 0, "PNG のファイルサイズが 0"


# ---------------------------------------------------------------------------
# TI-3: SVG ファイルが出力される
# ---------------------------------------------------------------------------

def test_ti3_render_svg_creates_file():
    """TI-3: render_i4_maze() が SVG ファイルを生成する。

    根拠: output_path に SVG が存在し、<svg> タグを含むこと。
    """
    result = _make_tiny_result(rows=4, cols=4, cell_size=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test.svg")
        returned = render_i4_maze(result, out_path, format="svg")
        assert os.path.isfile(returned), f"SVG が生成されていない: {returned}"
        with open(returned, encoding="utf-8") as f:
            content = f.read()
        assert "<svg" in content, "SVG タグが見つからない"
        assert "xmlns" in content, "xmlns 属性が欠落"


# ---------------------------------------------------------------------------
# TI-4: PNG 出力の縦横サイズ = grid × cell_size
# ---------------------------------------------------------------------------

def test_ti4_output_dimensions():
    """TI-4: PNG 出力サイズが grid_width*cell_size × grid_height*cell_size であること。

    根拠: _render_png では img_w = grid_width * cs, img_h = grid_height * cs。
    """
    rows, cols, cs = 6, 8, 7
    result = _make_tiny_result(rows=rows, cols=cols, cell_size=cs)
    png_bytes = _render_png(result)
    img = Image.open(__import__("io").BytesIO(png_bytes))
    assert img.width == cols * cs, f"幅: 期待={cols * cs}, 実際={img.width}"
    assert img.height == rows * cs, f"高さ: 期待={rows * cs}, 実際={img.height}"


# ---------------------------------------------------------------------------
# TI-5: G1 暗部は明部より太線
# ---------------------------------------------------------------------------

def test_ti5_dark_areas_have_thicker_lines():
    """TI-5: density 高い（暗部）の描画幅 > density 低い（明部）の描画幅。

    根拠: _g1_thickness(0.9) > _g1_thickness(0.1) であること。
    density > 0.7: 4-8px, density < 0.3: 1px の仕様に準拠。
    """
    thick_dark = _g1_thickness(0.9)
    thick_bright = _g1_thickness(0.1)
    assert thick_dark > thick_bright, (
        f"暗部({thick_dark:.2f}px) が明部({thick_bright:.2f}px) より太くなっていない"
    )
    # 仕様確認: 暗部は 4px 以上、明部は 2px 以下
    assert thick_dark >= 4.0, f"暗部(density=0.9)の線幅 {thick_dark:.2f}px が 4px 未満"
    assert thick_bright <= 2.0, f"明部(density=0.1)の線幅 {thick_bright:.2f}px が 2px より太い"


# ---------------------------------------------------------------------------
# TI-6: _g1_thickness 境界値
# ---------------------------------------------------------------------------

def test_ti6_g1_thickness_boundaries():
    """TI-6: _g1_thickness() の境界値確認。

    根拠:
      density=0.0 → 1.0px (BRIGHT) — _G1_THICK_BRIGHT
      density=1.0 → 8.0px (DARK)   — _G1_THICK_DARK
      density=0.5 → 4.5px (中間)   — 線形補間
    """
    assert _g1_thickness(0.0) == pytest.approx(1.0, abs=0.01), "density=0 → 1px"
    assert _g1_thickness(1.0) == pytest.approx(8.0, abs=0.01), "density=1 → 8px"
    assert _g1_thickness(0.5) == pytest.approx(4.5, abs=0.01), "density=0.5 → 4.5px"
    # クランプ確認
    assert _g1_thickness(-1.0) == pytest.approx(1.0, abs=0.01), "density<0 → 1px (クランプ)"
    assert _g1_thickness(2.0)  == pytest.approx(8.0, abs=0.01), "density>1 → 8px (クランプ)"


# ---------------------------------------------------------------------------
# TI-7: 不正フォーマットで ValueError
# ---------------------------------------------------------------------------

def test_ti7_invalid_format_raises_value_error():
    """TI-7: render_i4_maze() に無効フォーマットを指定すると ValueError。

    根拠: format ∉ {'png', 'svg'} → ValueError("サポート外のフォーマット...")
    """
    result = _make_tiny_result()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test.jpg")
        with pytest.raises(ValueError):
            render_i4_maze(result, out_path, format="jpg")


# ---------------------------------------------------------------------------
# TI-8: 解経路ピクセルが非解経路より暗い
# ---------------------------------------------------------------------------

def test_ti8_solution_path_pixels_are_darker():
    """TI-8: PNG の解経路ピクセルが非解経路（灰色）より暗い（黒寄り）。

    根拠: 解経路は (0,0,0) 黒, 非解経路は (204,204,204) 灰色。
         解経路セルの中心付近のピクセル値が非解経路より小さいこと。
    """
    # 単純な 2×2 グリッドで確認
    # 解経路: (0,0) → (0,1) → (1,1) → (1,0) （ジグザグ）
    path = [(0, 0), (0, 1), (1, 1), (1, 0)]
    open_passages = {_norm_key(path[i], path[i + 1]) for i in range(len(path) - 1)}
    walls: set = set()
    for r in range(2):
        for c in range(2):
            if c + 1 < 2:
                key = _norm_key((r, c), (r, c + 1))
                if key not in open_passages:
                    walls.add(key)
            if r + 1 < 2:
                key = _norm_key((r, c), (r + 1, c))
                if key not in open_passages:
                    walls.add(key)

    # 全セルが暗部 (density=0.8)
    density = np.full((2, 2), 0.8)
    cs = 20  # 大きめのセルサイズで確認しやすく
    result = I4MazeResult(
        grid_width=2,
        grid_height=2,
        cell_size_px=cs,
        walls=walls,
        solution_path=path,
        density_map=density,
        entrance=path[0],
        exit=path[-1],
    )
    png_bytes = _render_png(result)
    img = Image.open(__import__("io").BytesIO(png_bytes)).convert("L")
    arr = np.array(img)

    # 解経路 (0,0)→(0,1) の中点ピクセルを確認（水平方向）
    # セル(0,0)中心: (cs//2, cs//2), セル(0,1)中心: (3*cs//2, cs//2)
    # 中点: (cs, cs//2) → arr[y=cs//2, x=cs]
    mid_y = cs // 2
    mid_x = cs  # (0,0)→(0,1) の中点 x 座標
    sol_pixel = int(arr[mid_y, mid_x])

    assert sol_pixel < 128, (
        f"解経路中点ピクセル値 {sol_pixel} が期待より明るい（黒 < 128 のはず）"
    )


# ---------------------------------------------------------------------------
# TI-9: density_maze_result_to_i4() 変換
# ---------------------------------------------------------------------------

def test_ti9_density_maze_result_to_i4_conversion():
    """TI-9: density_maze_result_to_i4() が I4MazeResult を正しく生成する。

    根拠:
      - grid.rows=4, grid.cols=4 → I4MazeResult.grid_height=4, grid_width=4
      - solution_path の (row, col) 変換が正しいこと
      - density_map = 1 - luminance であること
    """
    from backend.core.density.grid_builder import CellGrid

    rows, cols = 4, 4
    lum = np.random.rand(rows, cols).astype(np.float64)
    # 簡易な adj: 全セルを一列に連結（0→1→2→...→15）
    adj: dict = {i: [] for i in range(rows * cols)}
    for i in range(rows * cols - 1):
        adj[i].append(i + 1)
        adj[i + 1].append(i)

    grid = CellGrid(rows=rows, cols=cols, luminance=lum, walls=[])
    solution_ids = list(range(rows * cols))

    result = density_maze_result_to_i4(
        grid, adj, solution_ids,
        entrance_id=0, exit_id=rows * cols - 1,
        cell_size_px=4,
    )

    assert result.grid_width == cols
    assert result.grid_height == rows
    assert result.cell_size_px == 4
    assert result.entrance == (0, 0)
    assert result.exit == (rows - 1, cols - 1)
    assert result.density_map.shape == (rows, cols)
    # density = 1 - luminance
    np.testing.assert_allclose(
        result.density_map, 1.0 - lum, atol=1e-10,
        err_msg="density_map = 1 - luminance であること"
    )
    # solution_path が (row, col) タプルのリストであること
    assert result.solution_path[0] == (0, 0)


# ---------------------------------------------------------------------------
# TI-10: 空の解経路でエラーなし
# ---------------------------------------------------------------------------

def test_ti10_empty_solution_path_renders_without_error():
    """TI-10: 解経路が空の I4MazeResult も PNG/SVG を生成できる。

    根拠: solution_path=[] のとき _build_solution_set() が空集合を返す。
         解経路を描画するループがスキップされる。
    """
    result = _make_tiny_result()
    result = I4MazeResult(
        grid_width=result.grid_width,
        grid_height=result.grid_height,
        cell_size_px=result.cell_size_px,
        walls=result.walls,
        solution_path=[],          # 空
        density_map=result.density_map,
        entrance=(0, 0),
        exit=(0, 0),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        png_path = os.path.join(tmpdir, "empty_sol.png")
        svg_path = os.path.join(tmpdir, "empty_sol.svg")
        render_i4_maze(result, png_path, format="png")
        render_i4_maze(result, svg_path, format="svg")
        assert os.path.isfile(png_path)
        assert os.path.isfile(svg_path)
