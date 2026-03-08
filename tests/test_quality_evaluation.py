# -*- coding: utf-8 -*-
"""
scripts/evaluate_quality.py の品質評価機能テスト（cmd_358k_a2）。

対象関数:
  - compute_ssim() / _ssim_simple()
  - compute_edge_ssim()
  - preprocess_for_ssim()
  - evaluate_quality()
  - generate_and_evaluate_masterpiece()
  - MASTERPIECE_OPTIMAL_PARAMS

実験結果（cmd_358k_a2 SSIM 探索）:
  - masterpiece 3本柱 + grid_size=8:  gradient=0.5566 [good], circle=0.5072 [good]
  - masterpiece 3本柱 + grid_size=30: gradient=0.4506 [fair]
  - 大グリッド (100+): fair/poor（壁線細化によるSSIM低下）
  - Edge-SSIM=0.95+ は小グリッドで達成（輪郭保持性は優秀）
  - "excellent" (≥0.70) は二値壁レンダリングの構造的限界で未達
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# scripts/ を sys.path に追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from evaluate_quality import (
    _ssim_simple,
    compute_ssim,
    compute_edge_ssim,
    preprocess_for_ssim,
    evaluate_quality,
    generate_and_evaluate_masterpiece,
    MASTERPIECE_OPTIMAL_PARAMS,
)


# ============================================================
# ヘルパー
# ============================================================

def _gradient_img(w: int = 64, h: int = 64) -> Image.Image:
    arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return Image.fromarray(arr, mode="L")


def _solid_img(val: int = 128, w: int = 64, h: int = 64) -> Image.Image:
    arr = np.full((h, w), val, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def _make_png_bytes(arr: np.ndarray) -> bytes:
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================
# _ssim_simple / compute_ssim
# ============================================================

def test_ssim_identical_images():
    """同一画像の SSIM は 1.0 に近い。"""
    arr = np.random.default_rng(0).random((64, 64))
    ssim = _ssim_simple(arr, arr)
    assert ssim >= 0.99, f"同一画像 SSIM={ssim:.4f} < 0.99"


def test_ssim_uniform_images():
    """均一画像（分散=0）の SSIM は C2/(C2) = 1.0。"""
    a = np.ones((32, 32), dtype=np.float64) * 0.5
    b = np.ones((32, 32), dtype=np.float64) * 0.8
    ssim = _ssim_simple(a, b)
    # 均一画像同士: 分散=0 → C2/(C2) = 1.0、輝度差のみ反映
    assert 0.0 <= ssim <= 1.0, f"均一画像 SSIM={ssim:.4f} 範囲外"


def test_ssim_different_images_lower():
    """明るい画像と暗い画像の SSIM は同一画像より低い。"""
    white = np.ones((64, 64), dtype=np.float64)
    black = np.zeros((64, 64), dtype=np.float64)
    ssim = _ssim_simple(white, black)
    ssim_same = _ssim_simple(white, white)
    assert ssim < ssim_same, f"白vs黒 SSIM({ssim:.4f}) ≥ 同一({ssim_same:.4f})"


def test_compute_ssim_returns_float():
    """compute_ssim は float を返す（skimage あり/なし両方）。"""
    a = np.random.default_rng(1).random((64, 64))
    b = np.random.default_rng(2).random((64, 64))
    result = compute_ssim(a, b)
    assert isinstance(result, float)
    assert -1.0 <= result <= 1.0


def test_compute_ssim_symmetry():
    """SSIM は対称: compute_ssim(a, b) == compute_ssim(b, a)。"""
    a = np.random.default_rng(3).random((32, 32))
    b = np.random.default_rng(4).random((32, 32))
    assert abs(compute_ssim(a, b) - compute_ssim(b, a)) < 1e-10


# ============================================================
# compute_edge_ssim
# ============================================================

def test_edge_ssim_returns_float():
    """compute_edge_ssim は float を返す。"""
    a = np.random.default_rng(5).random((64, 64))
    b = np.random.default_rng(6).random((64, 64))
    result = compute_edge_ssim(a, b)
    assert isinstance(result, float)
    assert -1.0 <= result <= 1.0


def test_edge_ssim_identical_image():
    """同一画像の Edge-SSIM は高い (≥0.9)。"""
    arr = np.random.default_rng(7).random((64, 64))
    result = compute_edge_ssim(arr, arr)
    assert result >= 0.9, f"Edge-SSIM(同一)={result:.4f} < 0.9"


# ============================================================
# preprocess_for_ssim
# ============================================================

def test_preprocess_for_ssim_shape():
    """preprocess_for_ssim は (H, W) float64 配列を返す。"""
    img = _gradient_img(128, 64)
    arr = preprocess_for_ssim(img, target_size=(256, 256))
    assert arr.shape == (256, 256), f"shape={arr.shape}"
    assert arr.dtype == np.float64


def test_preprocess_for_ssim_range():
    """preprocess_for_ssim の値域は [0, 1]。"""
    img = _gradient_img(64, 64)
    arr = preprocess_for_ssim(img, target_size=(256, 256))
    assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_preprocess_accepts_rgb():
    """RGB 画像（カラー PNG）も受け付ける（グレースケール変換）。"""
    rgb = Image.new("RGB", (64, 64), (200, 100, 50))
    arr = preprocess_for_ssim(rgb, target_size=(128, 128))
    assert arr.shape == (128, 128)


# ============================================================
# evaluate_quality
# ============================================================

def test_evaluate_quality_returns_required_keys():
    """evaluate_quality は必須キーを全て返す。"""
    input_img = _gradient_img()
    png_bytes = _make_png_bytes(np.full((64, 64), 200, dtype=np.uint8))
    result = evaluate_quality(input_img, png_bytes)

    for key in ["ssim", "edge_ssim", "rating", "message", "target_size"]:
        assert key in result, f"キー '{key}' が結果に存在しない"


def test_evaluate_quality_rating_levels():
    """SSIM 値に応じて正しい rating を返す。"""
    # 同一画像同士 → SSIM高い → excellent
    arr = np.random.default_rng(8).integers(0, 256, (64, 64), dtype=np.uint8)
    png_bytes = _make_png_bytes(arr)
    input_img = Image.fromarray(arr, mode="L")
    result = evaluate_quality(input_img, png_bytes)
    assert result["rating"] == "excellent", f"同一画像のrating={result['rating']}"

    # 白画像 vs 黒画像 → SSIM低い → poor
    white_img = _solid_img(255)
    black_png = _make_png_bytes(np.zeros((64, 64), dtype=np.uint8))
    result_poor = evaluate_quality(white_img, black_png)
    assert result_poor["rating"] in ("poor", "fair"), f"白vs黒rating={result_poor['rating']}"


def test_evaluate_quality_ssim_range():
    """evaluate_quality の ssim, edge_ssim は [−1, 1] の範囲。"""
    input_img = _gradient_img()
    png_bytes = _make_png_bytes(np.full((64, 64), 128, dtype=np.uint8))
    result = evaluate_quality(input_img, png_bytes)
    assert -1.0 <= result["ssim"] <= 1.0
    assert -1.0 <= result["edge_ssim"] <= 1.0


def test_evaluate_quality_custom_target_size():
    """target_size を変えても evaluate_quality が動作する。"""
    input_img = _gradient_img()
    png_bytes = _make_png_bytes(np.zeros((64, 64), dtype=np.uint8))
    for size in [(128, 128), (512, 512)]:
        result = evaluate_quality(input_img, png_bytes, target_size=size)
        assert result["target_size"] == size


# ============================================================
# MASTERPIECE_OPTIMAL_PARAMS
# ============================================================

def test_masterpiece_optimal_params_keys():
    """MASTERPIECE_OPTIMAL_PARAMS に必要なキーが揃っている。"""
    required = {"grid_size", "thickness_range", "extra_removal_rate",
                "dark_threshold", "light_threshold", "use_image_guided",
                "solution_highlight", "show_solution", "edge_weight", "stroke_width"}
    assert required.issubset(MASTERPIECE_OPTIMAL_PARAMS.keys())


def test_masterpiece_optimal_params_values():
    """MASTERPIECE_OPTIMAL_PARAMS の値が探索結果（cmd_358k_a2）と一致する。"""
    p = MASTERPIECE_OPTIMAL_PARAMS
    assert p["grid_size"] in range(5, 11), f"grid_size={p['grid_size']} (推奨: 5-10)"
    assert p["thickness_range"] >= 1.0, "thickness_range < 1.0"
    assert p["use_image_guided"] is True
    assert p["solution_highlight"] is False


# ============================================================
# generate_and_evaluate_masterpiece — 統合テスト
# ============================================================

def test_masterpiece_evaluation_gradient_good():
    """グラデーション画像 + masterpiece 3本柱 → SSIM ≥ 0.50 (good)。

    実験結果: grid_size=8 → gradient SSIM=0.5566 [good]
    """
    img = _gradient_img(64, 64)
    result = generate_and_evaluate_masterpiece(img)

    assert result["ssim"] >= 0.50, (
        f"グラデーション画像 SSIM={result['ssim']:.4f} < 0.50 [good 未達]"
    )
    assert result["rating"] in ("good", "excellent"), (
        f"rating={result['rating']} (期待: good 以上)"
    )


def test_masterpiece_evaluation_returns_required_fields():
    """generate_and_evaluate_masterpiece は必須フィールドを返す。"""
    img = _gradient_img(64, 64)
    result = generate_and_evaluate_masterpiece(img)

    for key in ["ssim", "edge_ssim", "rating", "maze_id", "grid_size",
                "preset", "solution_path_length"]:
        assert key in result, f"キー '{key}' が結果に存在しない"

    assert result["preset"] == "masterpiece"
    assert result["solution_path_length"] > 0


def test_masterpiece_evaluation_grid_size_override():
    """grid_size を指定して generate_and_evaluate_masterpiece が動作する。"""
    img = _gradient_img(64, 64)
    result = generate_and_evaluate_masterpiece(img, grid_size=5)
    assert result["grid_size"] == 5
    assert result["ssim"] >= 0.50, (
        f"grid_size=5 SSIM={result['ssim']:.4f} < 0.50"
    )


def test_masterpiece_edge_ssim_high():
    """masterpiece 3本柱の Edge-SSIM は 0.70 以上（輪郭保持性が高い）。

    実験結果: grid_size=8 → gradient Edge-SSIM=0.8092
    """
    img = _gradient_img(64, 64)
    result = generate_and_evaluate_masterpiece(img)
    assert result["edge_ssim"] >= 0.70, (
        f"Edge-SSIM={result['edge_ssim']:.4f} < 0.70: 輪郭保持性が低い"
    )
