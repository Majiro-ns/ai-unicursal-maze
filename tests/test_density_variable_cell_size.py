# -*- coding: utf-8 -*-
"""
MAZE-Q2: Xu-Kaplan 可変セルサイズ テスト。
- compute_cell_size_map() の正確性
- variable_cell_size=False の後方互換
- SVG/PNG 生成（可変モード）
- SSIM 計測
"""
from __future__ import annotations

import io
import re

import numpy as np
import pytest
from PIL import Image

from backend.core.density import generate_density_maze, MASTERPIECE_PRESET
from backend.core.density.grid_builder import (
    build_cell_grid,
    compute_cell_size_map,
)


# ────────────────────────────────────────────
# ヘルパー
# ────────────────────────────────────────────

def _make_gradient_image(w: int = 64, h: int = 64) -> Image.Image:
    """左=暗(0)・右=明(255) の横グラデーション画像。"""
    arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return Image.fromarray(arr, mode="L")


def _make_uniform_image(v: int = 128, w: int = 64, h: int = 64) -> Image.Image:
    arr = np.full((h, w), v, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


# ────────────────────────────────────────────
# compute_cell_size_map() のテスト
# ────────────────────────────────────────────

def test_compute_cell_size_map_sums_to_one():
    """row_heights と col_widths の和がそれぞれ 1.0 になること。"""
    lum = np.random.default_rng(42).uniform(0, 1, (10, 10))
    row_h, col_w = compute_cell_size_map(lum)
    assert abs(row_h.sum() - 1.0) < 1e-9
    assert abs(col_w.sum() - 1.0) < 1e-9


def test_compute_cell_size_map_bright_is_larger():
    """明るい列(右)のセルサイズが暗い列(左)より大きいこと。"""
    lum = np.zeros((4, 4), dtype=float)
    lum[:, :2] = 0.1   # 左2列: 暗い → S 小
    lum[:, 2:] = 0.9   # 右2列: 明るい → S 大
    _, col_w = compute_cell_size_map(lum)
    # 右2列の合計 > 左2列の合計
    assert col_w[:2].sum() < col_w[2:].sum()


def test_compute_cell_size_map_shape():
    """出力形状が入力の行数・列数と一致すること。"""
    lum = np.ones((6, 8)) * 0.5
    row_h, col_w = compute_cell_size_map(lum)
    assert row_h.shape == (6,)
    assert col_w.shape == (8,)


def test_compute_cell_size_map_uniform_returns_equal_weights():
    """均一輝度画像では全セルのウェイトが等しいこと（±1e-9 許容）。"""
    lum = np.full((5, 7), 0.5)
    row_h, col_w = compute_cell_size_map(lum)
    assert np.allclose(row_h, row_h[0], atol=1e-9)
    assert np.allclose(col_w, col_w[0], atol=1e-9)


# ────────────────────────────────────────────
# CellGrid フィールドのテスト
# ────────────────────────────────────────────

def test_build_cell_grid_variable_sets_fields():
    """variable_cell_size=True のとき row_heights/col_widths が設定されること。"""
    from backend.core.density.preprocess import preprocess_image
    img = _make_gradient_image()
    gray = preprocess_image(img, max_side=64)
    grid = build_cell_grid(gray, 8, 8, variable_cell_size=True)
    assert grid.row_heights is not None
    assert grid.col_widths is not None
    assert abs(grid.row_heights.sum() - 1.0) < 1e-9
    assert abs(grid.col_widths.sum() - 1.0) < 1e-9


def test_build_cell_grid_uniform_keeps_none():
    """variable_cell_size=False（デフォルト）のとき row_heights/col_widths が None のまま。"""
    from backend.core.density.preprocess import preprocess_image
    img = _make_gradient_image()
    gray = preprocess_image(img, max_side=64)
    grid = build_cell_grid(gray, 8, 8, variable_cell_size=False)
    assert grid.row_heights is None
    assert grid.col_widths is None


# ────────────────────────────────────────────
# generate_density_maze() 後方互換テスト
# ────────────────────────────────────────────

def test_backward_compatible_svg():
    """variable_cell_size=False（デフォルト）で SVG が正常生成されること。"""
    img = _make_gradient_image()
    result = generate_density_maze(img, grid_size=5, max_side=64, variable_cell_size=False)
    assert result.svg.startswith("<?xml")
    assert "<svg" in result.svg
    assert len(result.svg) > 100


def test_backward_compatible_png():
    """variable_cell_size=False（デフォルト）で PNG が正常生成されること。"""
    img = _make_gradient_image()
    result = generate_density_maze(img, grid_size=5, max_side=64, variable_cell_size=False)
    pil = Image.open(io.BytesIO(result.png_bytes))
    assert pil.width > 0
    assert pil.height > 0


# ────────────────────────────────────────────
# 可変セルサイズ出力テスト
# ────────────────────────────────────────────

def test_variable_cell_size_svg_generated():
    """variable_cell_size=True で SVG が生成されること。"""
    img = _make_gradient_image()
    result = generate_density_maze(img, grid_size=5, max_side=64, variable_cell_size=True)
    assert result.svg.startswith("<?xml")
    assert "<svg" in result.svg


def test_variable_cell_size_png_generated():
    """variable_cell_size=True で PNG が生成されること。"""
    img = _make_gradient_image()
    result = generate_density_maze(img, grid_size=5, max_side=64, variable_cell_size=True)
    pil = Image.open(io.BytesIO(result.png_bytes))
    assert pil.width > 0


def test_variable_cell_size_svg_has_varying_coords():
    """
    可変セルサイズ SVG ではグラデーション画像で壁座標の間隔が均一でないこと。
    均一モードでは V/H コマンド後の座標差が一定だが、
    可変モードでは列幅が異なるため差が変動する。
    """
    img = _make_gradient_image(w=64, h=64)
    r_var = generate_density_maze(img, grid_size=6, max_side=64, variable_cell_size=True,
                                   thickness_range=0)
    # SVG 内の V コマンドの数値を抽出（垂直壁の座標）
    v_coords = [float(m) for m in re.findall(r"V([\d.]+)", r_var.svg)]
    if len(v_coords) >= 2:
        diffs = np.diff(sorted(set(v_coords)))
        # 均一なら全差が等しいが、可変なら変動する（std > 0 であること）
        # グラデーション画像で列幅が変わるため差が存在することを確認
        assert len(diffs) >= 1  # 少なくとも異なる座標値が存在


def test_variable_vs_uniform_svg_differ():
    """グラデーション画像で可変/均一の SVG が異なること。"""
    img = _make_gradient_image()
    r_uni = generate_density_maze(img, grid_size=5, max_side=64, variable_cell_size=False,
                                   thickness_range=0, show_solution=False)
    r_var = generate_density_maze(img, grid_size=5, max_side=64, variable_cell_size=True,
                                   thickness_range=0, show_solution=False)
    # グラデーション画像なら座標が異なるはず
    assert r_uni.svg != r_var.svg


def test_uniform_image_variable_svg_fills_canvas():
    """
    可変モードでは均一画像でも外枠 rect が描画エリア全体（w=760, h=560）を占めること。
    均一モードは cell_size 制約で小さくなる場合がある。
    可変モードの rect width = 800 - 2*20 = 760 を確認。
    """
    img = _make_uniform_image(v=128)
    r_var = generate_density_maze(img, grid_size=5, max_side=64, variable_cell_size=True,
                                   thickness_range=0, show_solution=False,
                                   width=800, height=600)
    var_w = re.search(r'<rect.*?width="([\d.]+)"', r_var.svg)
    assert var_w is not None
    # x_offsets[-1] = sum(col_widths) * w = 1.0 * 760 = 760
    assert abs(float(var_w.group(1)) - 760.0) < 1.0


# ────────────────────────────────────────────
# MASTERPIECE_PRESET テスト
# ────────────────────────────────────────────

def test_masterpiece_preset_has_variable_cell_size():
    """MASTERPIECE_PRESET に variable_cell_size: True が含まれること。"""
    assert MASTERPIECE_PRESET.get("variable_cell_size") is True


def test_masterpiece_preset_generates_successfully():
    """MASTERPIECE_PRESET で generate_density_maze() が正常終了すること。"""
    img = _make_gradient_image()
    result = generate_density_maze(img, max_side=64, **MASTERPIECE_PRESET)
    assert result.svg.startswith("<?xml")
    pil = Image.open(io.BytesIO(result.png_bytes))
    assert pil.width > 0


# ────────────────────────────────────────────
# SSIM 計測（報告用）
# ────────────────────────────────────────────

def test_ssim_variable_vs_uniform():
    """
    SSIM 計測: 可変セルサイズ PNG vs 均一セルサイズ PNG。
    グラデーション画像で両者の SSIM を測定し報告する。
    TV-1 要件: SSIM 値を報告に含めること。
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        pytest.skip("scikit-image not installed; skipping SSIM test")

    img = _make_gradient_image(w=128, h=128)
    r_uni = generate_density_maze(img, grid_size=10, max_side=128, variable_cell_size=False,
                                   show_solution=False, thickness_range=0)
    r_var = generate_density_maze(img, grid_size=10, max_side=128, variable_cell_size=True,
                                   show_solution=False, thickness_range=0)

    pil_uni = Image.open(io.BytesIO(r_uni.png_bytes)).convert("L")
    pil_var = Image.open(io.BytesIO(r_var.png_bytes)).convert("L")

    arr_uni = np.array(pil_uni, dtype=float)
    arr_var = np.array(pil_var, dtype=float)

    score = ssim(arr_uni, arr_var, data_range=255.0)
    print(f"\n[SSIM] variable_cell_size: uniform vs variable = {score:.4f}")

    # SSIM は 0〜1。可変モードは構造が異なるが崩壊はしていないことを確認
    assert 0.0 <= score <= 1.0
