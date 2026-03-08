# -*- coding: utf-8 -*-
"""
グレースケール壁テスト（cmd_360k_a2）。

対象:
  - _wall_color(): 境界値・公式検証
  - maze_to_svg(): rgb(...) カラーで壁を描画
  - maze_to_png(): グレースケール壁色
"""
from __future__ import annotations

import io
import re

import numpy as np
import pytest
from PIL import Image

from backend.core.density import generate_density_maze
from backend.core.density.exporter import _wall_color


# ============================================================
# _wall_color() 単体テスト
# ============================================================

def test_wall_color_black():
    """avg_lum=0.0（黒画素）→ rgb(80,80,80)（下限80）。"""
    assert _wall_color(0.0) == "rgb(80,80,80)"


def test_wall_color_white():
    """avg_lum=1.0（白画素）→ rgb(220,220,220)（完全白は除く）。"""
    assert _wall_color(1.0) == "rgb(220,220,220)"


def test_wall_color_mid():
    """avg_lum=0.5 → rgb(110,110,110)。"""
    assert _wall_color(0.5) == "rgb(110,110,110)"


def test_wall_color_format():
    """返り値が 'rgb(N,N,N)' 形式である。"""
    for lum in [0.0, 0.25, 0.5, 0.75, 1.0]:
        color = _wall_color(lum)
        assert re.fullmatch(r"rgb\(\d+,\d+,\d+\)", color), (
            f"avg_lum={lum}: 返り値 '{color}' が rgb(N,N,N) 形式でない"
        )


def test_wall_color_monotone():
    """avg_lum が増加するにつれて壁色 v が単調増加する。"""
    lums = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    vs = [int(_wall_color(l).split("(")[1].split(",")[0]) for l in lums]
    for i in range(len(vs) - 1):
        assert vs[i] <= vs[i + 1], (
            f"avg_lum={lums[i]}→{lums[i+1]} で v={vs[i]}→{vs[i+1]}: 単調増加でない"
        )


def test_wall_color_range():
    """返り値の v 値は [80, 220] の範囲に収まる（下限80）。"""
    for lum in np.linspace(0.0, 1.0, 21):
        color = _wall_color(float(lum))
        v = int(color.split("(")[1].split(",")[0])
        assert 80 <= v <= 220, f"avg_lum={lum:.2f}: v={v} が範囲外"


# ============================================================
# SVG グレースケール壁テスト
# ============================================================

def _make_half_image(w: int = 64, h: int = 64) -> Image.Image:
    """左半分=黒(0)、右半分=白(255)のグレースケール画像。"""
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[:, w // 2:] = 255
    return Image.fromarray(arr, mode="L")


def test_svg_walls_use_rgb_colors():
    """SVG の壁グループが 'rgb(' 形式の色を使用している（'black' でない）。"""
    img = _make_half_image()
    result = generate_density_maze(
        img, grid_size=8, max_side=64,
        thickness_range=1.5, extra_removal_rate=0.0,
        use_image_guided=False,
    )
    assert 'rgb(' in result.svg, "SVG に rgb( が含まれない: グレースケール壁が機能していない"
    # 旧フォーマット 'stroke="black"' が壁グループに残っていないことを確認
    wall_black = re.findall(r'<g stroke="black" stroke-width=', result.svg)
    assert len(wall_black) == 0, (
        f"SVG に 'stroke=\"black\"' の壁グループが {len(wall_black)} 件残っている"
    )


def test_svg_dark_walls_have_darker_color_than_bright():
    """左半分（暗部）の壁は右半分（明部）の壁より暗い rgb 値を持つ。"""
    img = _make_half_image(64, 64)
    result = generate_density_maze(
        img, grid_size=8, max_side=64, width=400, height=400,
        thickness_range=1.5, extra_removal_rate=0.0,
        use_image_guided=False,
    )

    width_match = re.search(r'<svg[^>]+width="(\d+)"', result.svg)
    svg_width = int(width_match.group(1)) if width_match else 400
    mid_x = svg_width / 2.0

    group_pat = re.compile(
        r'<g stroke="rgb\((\d+),\d+,\d+\)" stroke-width="[\d.]+">'
        r'<path d="([^"]+)" fill="none"/>'
        r'</g>'
    )
    mv_pat = re.compile(r'M([\d.]+) [\d.]+V')
    mh_pat = re.compile(r'M([\d.]+) [\d.]+H([\d.]+)')

    left_vs, right_vs = [], []
    for m in group_pat.finditer(result.svg):
        v = int(m.group(1))
        path_d = m.group(2)
        for vm in mv_pat.finditer(path_d):
            x = float(vm.group(1))
            (left_vs if x < mid_x else right_vs).append(v)
        for hm in mh_pat.finditer(path_d):
            avg_x = (float(hm.group(1)) + float(hm.group(2))) / 2.0
            (left_vs if avg_x < mid_x else right_vs).append(v)

    assert left_vs, "左半分に壁グループが存在しない"
    assert right_vs, "右半分に壁グループが存在しない"

    left_avg_v = float(np.mean(left_vs))
    right_avg_v = float(np.mean(right_vs))
    assert left_avg_v < right_avg_v, (
        f"左壁色v平均({left_avg_v:.1f}) ≥ 右壁色v平均({right_avg_v:.1f}): "
        "暗部の壁が明部の壁より暗くない"
    )


# ============================================================
# PNG グレースケール壁テスト
# ============================================================

def test_png_dark_walls_darker_than_bright_walls():
    """PNG の左半分ピクセル平均が右半分より低い（暗部壁が暗い）。"""
    img = _make_half_image(64, 64)
    result = generate_density_maze(
        img, grid_size=8, max_side=64, width=400, height=400,
        thickness_range=1.5, extra_removal_rate=0.0,
        use_image_guided=False,
        show_solution=False,
    )
    png = Image.open(io.BytesIO(result.png_bytes)).convert("L")
    arr = np.array(png)
    h, w = arr.shape
    left_mean = float(arr[:, : w // 2].mean())
    right_mean = float(arr[:, w // 2:].mean())
    assert left_mean < right_mean, (
        f"左({left_mean:.1f}) ≥ 右({right_mean:.1f}): PNG がグレースケール壁を反映していない"
    )


def test_png_bright_walls_not_fully_black():
    """明部（右半分）の壁ピクセルは純黒(0)でなく灰色(> 0)である。"""
    img = _make_half_image(64, 64)
    result = generate_density_maze(
        img, grid_size=8, max_side=64, width=400, height=400,
        thickness_range=0.0,   # 壁厚固定にして色だけを評価
        extra_removal_rate=0.0,
        use_image_guided=False,
        show_solution=False,
    )
    png = Image.open(io.BytesIO(result.png_bytes)).convert("L")
    arr = np.array(png)
    h, w = arr.shape
    # 右半分（明部）の最小ピクセル値が 0 より大きければ、明部壁は黒でない
    right_half = arr[:, w // 2:]
    # 白背景(255)以外のピクセルが壁。その最小値が 0 より大きいことを確認
    wall_pixels = right_half[right_half < 255]
    if len(wall_pixels) > 0:
        assert wall_pixels.min() > 0, (
            f"明部の壁ピクセル最小値={wall_pixels.min()}: 純黒が残っている"
        )
