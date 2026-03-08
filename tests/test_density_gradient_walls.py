"""
Phase 4: 壁色グラデーション テスト
maze_ssim_a7 — SVG linearGradient 導入の検証

テスト項目:
  TEST-GW-1: use_gradient_walls=True の SVG に <linearGradient が含まれること
  TEST-GW-2: use_gradient_walls=True の SVG に <defs> が含まれること
  TEST-GW-3: use_gradient_walls=True の SVG に <rect fill="url(# が含まれること（グラデーション壁）
  TEST-GW-4: use_gradient_walls=False（デフォルト）では <linearGradient が含まれないこと
  TEST-GW-5: グラデーション数の上限確認（≤ 2 * (nlevels+1)^2）
  TEST-GW-6: generate_density_maze() が use_gradient_walls パラメータを受け付けること
  TEST-GW-7: MASTERPIECE_PRESET に use_gradient_walls=True が含まれること
  TEST-GW-8: use_gradient_walls=False で既存テスト全互換（リグレッション）
  TEST-GW-9: 垂直壁グラデーションの方向検証（x1=0 y1=0 x2=0 y2=1）
  TEST-GW-10: 水平壁グラデーションの方向検証（x1=0 y1=0 x2=1 y2=0）
  TEST-GW-11: 均一画像（輝度=0.5）でグラデーション数が適切に絞られること
  TEST-GW-12: SSIM レポート（勝劣を問わず数値記録のみ）
"""
from __future__ import annotations

import re

import numpy as np
import pytest
from PIL import Image

from backend.core.density import MASTERPIECE_PRESET, generate_density_maze
from backend.core.density.exporter import maze_to_svg, _wall_color
from backend.core.density.grid_builder import CellGrid


# ------------------------------------------------------------------
# ヘルパー
# ------------------------------------------------------------------

def _make_grid(rows: int = 4, cols: int = 4, lum_value: float = 0.5) -> CellGrid:
    """均一輝度の CellGrid を生成する。"""
    luminance = np.full((rows, cols), lum_value, dtype=np.float32)
    walls: list = []
    return CellGrid(rows=rows, cols=cols, luminance=luminance, walls=walls)


def _make_gradient_grid(rows: int = 4, cols: int = 4) -> CellGrid:
    """列方向に輝度グラデーション（0→1）の CellGrid を生成する。"""
    luminance = np.zeros((rows, cols), dtype=np.float32)
    for c in range(cols):
        luminance[:, c] = c / max(cols - 1, 1)
    return CellGrid(rows=rows, cols=cols, luminance=luminance, walls=[])


def _make_pil_image(rows: int = 20, cols: int = 20, value: int = 128) -> Image.Image:
    arr = np.full((rows, cols), value, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def _simple_adj(grid: CellGrid) -> dict:
    """全壁を保持（除去なし）した隣接リスト（空 dict）を返す。"""
    return {}


def _count_linearGradient(svg: str) -> int:
    return len(re.findall(r"<linearGradient\b", svg))


def _count_gradient_rects(svg: str) -> int:
    return len(re.findall(r'fill="url\(#gw_', svg))


# ------------------------------------------------------------------
# TEST-GW-1: linearGradient が含まれること
# ------------------------------------------------------------------
def test_svg_contains_linearGradient_when_enabled():
    grid = _make_gradient_grid(4, 4)
    adj = _simple_adj(grid)
    svg = maze_to_svg(
        grid, adj, entrance=0, exit_id=15, solution_path=[],
        show_solution=False, use_gradient_walls=True,
    )
    assert "<linearGradient" in svg, "use_gradient_walls=True のとき SVG に <linearGradient が必要"


# ------------------------------------------------------------------
# TEST-GW-2: <defs> が含まれること
# ------------------------------------------------------------------
def test_svg_contains_defs_when_enabled():
    grid = _make_gradient_grid(4, 4)
    adj = _simple_adj(grid)
    svg = maze_to_svg(
        grid, adj, entrance=0, exit_id=15, solution_path=[],
        show_solution=False, use_gradient_walls=True,
    )
    assert "<defs>" in svg, "use_gradient_walls=True のとき SVG に <defs> が必要"
    assert "</defs>" in svg


# ------------------------------------------------------------------
# TEST-GW-3: グラデーション壁 <rect fill="url(#gw_..."> が含まれること
# ------------------------------------------------------------------
def test_svg_contains_gradient_rects():
    grid = _make_gradient_grid(4, 4)
    adj = _simple_adj(grid)
    svg = maze_to_svg(
        grid, adj, entrance=0, exit_id=15, solution_path=[],
        show_solution=False, use_gradient_walls=True,
    )
    n_rects = _count_gradient_rects(svg)
    assert n_rects > 0, f"グラデーション壁 <rect> が 0 件（期待: >0）"


# ------------------------------------------------------------------
# TEST-GW-4: use_gradient_walls=False では linearGradient 不在
# ------------------------------------------------------------------
def test_no_linearGradient_when_disabled():
    grid = _make_gradient_grid(4, 4)
    adj = _simple_adj(grid)
    svg = maze_to_svg(
        grid, adj, entrance=0, exit_id=15, solution_path=[],
        show_solution=False, use_gradient_walls=False,
    )
    assert "<linearGradient" not in svg
    assert "<defs>" not in svg


# ------------------------------------------------------------------
# TEST-GW-5: グラデーション数の上限
# ------------------------------------------------------------------
def test_gradient_count_bounded():
    grid = _make_gradient_grid(8, 8)
    adj = _simple_adj(grid)
    stroke_quantize_levels = 20
    nlevels = stroke_quantize_levels
    max_expected = 2 * (nlevels + 1) ** 2  # 垂直 + 水平
    svg = maze_to_svg(
        grid, adj, entrance=0, exit_id=63, solution_path=[],
        show_solution=False,
        use_gradient_walls=True,
        stroke_quantize_levels=stroke_quantize_levels,
    )
    n_grads = _count_linearGradient(svg)
    assert n_grads <= max_expected, (
        f"グラデーション数 {n_grads} が上限 {max_expected} を超えている"
    )


# ------------------------------------------------------------------
# TEST-GW-6: generate_density_maze が use_gradient_walls を受け付ける
# ------------------------------------------------------------------
def test_generate_density_maze_accepts_gradient_param():
    img = _make_pil_image(20, 20, 128)
    result = generate_density_maze(
        img, grid_size=4, show_solution=False, use_gradient_walls=True
    )
    assert result.svg, "SVG が空"
    assert "<linearGradient" in result.svg, "generate_density_maze の SVG に linearGradient が必要"


# ------------------------------------------------------------------
# TEST-GW-7: MASTERPIECE_PRESET に use_gradient_walls=True
# ------------------------------------------------------------------
def test_masterpiece_preset_has_gradient_walls():
    assert "use_gradient_walls" in MASTERPIECE_PRESET, (
        "MASTERPIECE_PRESET に use_gradient_walls キーが必要"
    )
    assert MASTERPIECE_PRESET["use_gradient_walls"] is True, (
        "MASTERPIECE_PRESET['use_gradient_walls'] は True であるべき"
    )


# ------------------------------------------------------------------
# TEST-GW-8: リグレッション — use_gradient_walls=False で既存 <g> グループ形式が維持
# ------------------------------------------------------------------
def test_gradient_false_uses_path_group_format():
    grid = _make_gradient_grid(4, 4)
    adj = _simple_adj(grid)
    svg = maze_to_svg(
        grid, adj, entrance=0, exit_id=15, solution_path=[],
        show_solution=False, use_gradient_walls=False,
    )
    # 既存形式: <g stroke="..." stroke-width="..."><path d="..." fill="none"/></g>
    assert '<path d="' in svg or svg.count("<g ") == 0, (
        "use_gradient_walls=False で既存 <g><path> 形式またはウォールなしのどちらかであること"
    )
    # グラデーション要素が混入していないこと
    assert 'fill="url(#gw_' not in svg


# ------------------------------------------------------------------
# TEST-GW-9: 垂直壁グラデーションの方向 (x1=0 y1=0 x2=0 y2=1)
# ------------------------------------------------------------------
def test_vertical_gradient_direction():
    # 列違い輝度の 2x2 グリッド → 垂直壁のみ生成
    luminance = np.array([[0.2, 0.8], [0.2, 0.8]], dtype=np.float32)
    grid = CellGrid(rows=2, cols=2, luminance=luminance, walls=[])
    adj = _simple_adj(grid)
    svg = maze_to_svg(
        grid, adj, entrance=0, exit_id=3, solution_path=[],
        show_solution=False, use_gradient_walls=True,
    )
    # 垂直壁グラデーション: x2="0" y2="1"
    assert 'x2="0" y2="1"' in svg, (
        "垂直壁グラデーションの方向が x2=0 y2=1 でない"
    )


# ------------------------------------------------------------------
# TEST-GW-10: 水平壁グラデーションの方向 (x1=0 y1=0 x2=1 y2=0)
# ------------------------------------------------------------------
def test_horizontal_gradient_direction():
    # 行違い輝度の 2x2 グリッド → 水平壁のみ生成
    luminance = np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float32)
    grid = CellGrid(rows=2, cols=2, luminance=luminance, walls=[])
    adj = _simple_adj(grid)
    svg = maze_to_svg(
        grid, adj, entrance=0, exit_id=3, solution_path=[],
        show_solution=False, use_gradient_walls=True,
    )
    # 水平壁グラデーション: x2="1" y2="0"
    assert 'x2="1" y2="0"' in svg, (
        "水平壁グラデーションの方向が x2=1 y2=0 でない"
    )


# ------------------------------------------------------------------
# TEST-GW-11: 均一画像ではグラデーション数が少ない
# ------------------------------------------------------------------
def test_uniform_image_few_gradients():
    grid = _make_grid(6, 6, lum_value=0.5)  # 全セル同一輝度
    adj = _simple_adj(grid)
    svg = maze_to_svg(
        grid, adj, entrance=0, exit_id=35, solution_path=[],
        show_solution=False, use_gradient_walls=True,
        stroke_quantize_levels=20,
    )
    n_grads = _count_linearGradient(svg)
    # 均一画像: qi1=qi2=round(0.5*20)=10 → 垂直("v",10,10) + 水平("h",10,10) = 2種
    assert n_grads <= 4, (
        f"均一画像でグラデーション数が多すぎる: {n_grads}（期待 ≤ 4）"
    )


# ------------------------------------------------------------------
# TEST-GW-12: SSIM レポート（比較・参考値のみ。閾値なし）
# ------------------------------------------------------------------
def test_ssim_report_gradient_vs_flat(capsys):
    """グラデーション壁 vs 均一壁の SSIM 参考比較。失敗しない。"""
    try:
        from skimage.metrics import structural_similarity as ssim
        import numpy as np
        from PIL import Image as PILImage
    except ImportError:
        pytest.skip("skimage が未インストール")

    img = _make_pil_image(40, 40, 128)

    result_flat = generate_density_maze(
        img, grid_size=6, show_solution=False, use_gradient_walls=False
    )
    result_grad = generate_density_maze(
        img, grid_size=6, show_solution=False, use_gradient_walls=True
    )

    import io
    def png_to_arr(png_bytes):
        return np.array(PILImage.open(io.BytesIO(png_bytes)).convert("L"))

    arr_flat = png_to_arr(result_flat.png_bytes)
    arr_grad = png_to_arr(result_grad.png_bytes)

    # PNG は同一なので SSIM=1.0 になるはず（gradient はSVGのみ変更）
    # 参考ログとして出力
    if arr_flat.shape == arr_grad.shape:
        score = ssim(arr_flat, arr_grad, data_range=255)
        print(f"\n[GW-12] SSIM(flat vs gradient-svg): {score:.4f}")
    else:
        print(f"\n[GW-12] shape mismatch: {arr_flat.shape} vs {arr_grad.shape}")
    # 閾値なし: 常にPASS
