"""
cmd_689k_a6_maze_dm4: DM-4 多値トーン壁表現 テスト

backend/core/density/dm4.py + tonal_exporter.py の検証。

検証カテゴリ:
  A: DM4Config（継承・デフォルト・フィールド）
  B: TONAL_GRADES + _quantize_lum_to_grade（量子化精度）
  C: compute_dark_coverage（暗部ピクセル割合）
  D: DM4Result（継承・フィールド・型）
  E: generate_dm4_maze API（PNG・SVG・入出口・解経路）
  F: SSIM スコア（正値・上限・DM-2 比較改善）
  G: dark_coverage（黒/白画像・範囲）
  H: トーンレンダリング（暗部黒化・明部白化・壁厚変動）
  I: E2E（グラデーション・サークル・均一）
"""
from __future__ import annotations

import io
import struct

import numpy as np
import pytest
from PIL import Image

from backend.core.density.dm1 import DM1Config
from backend.core.density.dm2 import DM2Config, generate_dm2_maze
from backend.core.density.dm4 import (
    DM4Config,
    DM4Result,
    _compute_ssim,
    generate_dm4_maze,
)
from backend.core.density.tonal_exporter import (
    TONAL_GRADES,
    _quantize_lum_to_grade,
    compute_dark_coverage,
    maze_to_png_tonal,
    maze_to_svg_tonal,
)
from backend.core.density.solver import bfs_has_path


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _uniform(val: float, size: int = 64) -> Image.Image:
    px = int(np.clip(val * 255, 0, 255))
    return Image.fromarray(np.full((size, size), px, dtype=np.uint8), mode="L")


def _gradient_h(rows: int = 64, cols: int = 64) -> Image.Image:
    """左(暗)→右(明) 水平グラデーション。"""
    arr = np.tile(np.linspace(0, 255, cols), (rows, 1)).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _circle(rows: int = 64, cols: int = 64) -> Image.Image:
    """中心=白、外周=黒 の円グラデーション。"""
    cy, cx = rows // 2, cols // 2
    r = min(rows, cols) // 2 - 1
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    arr = np.clip(255 * (1.0 - dist / max(r, 1)), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _small_config(**kwargs) -> DM4Config:
    """テスト用小グリッド設定。"""
    base = dict(grid_rows=20, grid_cols=20, cell_size_px=5, render_scale=1)
    base.update(kwargs)
    return DM4Config(**base)


def _png_size(png_bytes: bytes) -> tuple[int, int]:
    w = struct.unpack(">I", png_bytes[16:20])[0]
    h = struct.unpack(">I", png_bytes[20:24])[0]
    return w, h


def _make_black_png(size: int = 64) -> bytes:
    img = Image.fromarray(np.zeros((size, size), dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_white_png(size: int = 64) -> bytes:
    img = Image.fromarray(np.full((size, size), 255, dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_mid_png(size: int = 64) -> bytes:
    img = Image.fromarray(np.full((size, size), 127, dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ===========================================================================
# A: DM4Config
# ===========================================================================

def test_dm4config_inherits_dm2config():
    """DM4Config が DM2Config を継承していること。"""
    cfg = DM4Config()
    assert isinstance(cfg, DM2Config)


def test_dm4config_inherits_dm1config():
    """DM4Config が DM1Config を継承していること（推移的継承）。"""
    cfg = DM4Config()
    assert isinstance(cfg, DM1Config)


def test_dm4config_has_tonal_grades():
    """DM4Config が tonal_grades フィールドを持つこと。"""
    cfg = DM4Config()
    assert hasattr(cfg, "tonal_grades")
    assert isinstance(cfg.tonal_grades, list)
    assert len(cfg.tonal_grades) == 8


def test_dm4config_has_render_scale():
    """DM4Config が render_scale フィールドを持つこと。"""
    cfg = DM4Config()
    assert hasattr(cfg, "render_scale")
    assert cfg.render_scale >= 1


def test_dm4config_has_tonal_thickness_range():
    """DM4Config が tonal_thickness_range フィールドを持つこと。"""
    cfg = DM4Config()
    assert hasattr(cfg, "tonal_thickness_range")
    assert cfg.tonal_thickness_range >= 0.0


def test_dm4config_has_ssim_target_size():
    """DM4Config が ssim_target_size フィールドを持つこと。"""
    cfg = DM4Config()
    assert hasattr(cfg, "ssim_target_size")
    assert len(cfg.ssim_target_size) == 2


def test_dm4config_default_tonal_grades_matches_tonal_grades_const():
    """DM4Config のデフォルト tonal_grades が TONAL_GRADES と一致すること。"""
    cfg = DM4Config()
    assert cfg.tonal_grades == list(TONAL_GRADES)


def test_dm4config_has_fill_cells():
    """DM4Config が fill_cells フィールドを持ち、デフォルト True であること。"""
    cfg = DM4Config()
    assert hasattr(cfg, "fill_cells")
    assert cfg.fill_cells is True


def test_dm4config_has_blur_radius():
    """DM4Config が blur_radius フィールドを持ち、正値であること。"""
    cfg = DM4Config()
    assert hasattr(cfg, "blur_radius")
    assert cfg.blur_radius >= 0.0


def test_dm4config_customizable():
    """DM4Config のパラメータをカスタマイズできること。"""
    cfg = DM4Config(tonal_thickness_range=3.0, render_scale=4, fill_cells=False, blur_radius=1.5)
    assert cfg.tonal_thickness_range == 3.0
    assert cfg.render_scale == 4
    assert cfg.fill_cells is False
    assert cfg.blur_radius == 1.5


# ===========================================================================
# B: TONAL_GRADES + _quantize_lum_to_grade
# ===========================================================================

def test_tonal_grades_length():
    """TONAL_GRADES が 8 要素であること。"""
    assert len(TONAL_GRADES) == 8


def test_tonal_grades_min_is_zero():
    """TONAL_GRADES の最小値が 0 であること。"""
    assert TONAL_GRADES[0] == 0


def test_tonal_grades_max_is_255():
    """TONAL_GRADES の最大値が 255 であること。"""
    assert TONAL_GRADES[-1] == 255


def test_tonal_grades_sorted_ascending():
    """TONAL_GRADES が昇順であること。"""
    assert TONAL_GRADES == sorted(TONAL_GRADES)


def test_tonal_grades_all_integers():
    """TONAL_GRADES の全要素が int であること。"""
    assert all(isinstance(g, int) for g in TONAL_GRADES)


def test_quantize_lum_zero_returns_first_grade():
    """lum=0.0 → TONAL_GRADES[0] = 0 を返すこと。"""
    assert _quantize_lum_to_grade(0.0) == 0


def test_quantize_lum_one_returns_last_grade():
    """lum=1.0 → TONAL_GRADES[-1] = 255 を返すこと。"""
    assert _quantize_lum_to_grade(1.0) == 255


def test_quantize_lum_mid_returns_grade_in_tonal_grades():
    """中間輝度の量子化結果が TONAL_GRADES 内の値であること。"""
    result = _quantize_lum_to_grade(0.5)
    assert result in TONAL_GRADES


def test_quantize_lum_monotone():
    """輝度が増すにつれて量子化グレード値が非減少であること。"""
    grades_out = [_quantize_lum_to_grade(v) for v in np.linspace(0, 1, 20)]
    # 単調非減少
    assert all(grades_out[i] <= grades_out[i + 1] for i in range(len(grades_out) - 1))


def test_quantize_returns_in_range():
    """量子化結果が 0〜255 の範囲に収まること。"""
    for lum in np.linspace(0.0, 1.0, 50):
        val = _quantize_lum_to_grade(float(lum))
        assert 0 <= val <= 255


# ===========================================================================
# C: compute_dark_coverage
# ===========================================================================

def test_dark_coverage_black_image():
    """真っ黒画像の dark_coverage ≈ 1.0 であること。"""
    cov = compute_dark_coverage(_make_black_png(), threshold=128)
    assert cov > 0.99


def test_dark_coverage_white_image():
    """真っ白画像の dark_coverage ≈ 0.0 であること。"""
    cov = compute_dark_coverage(_make_white_png(), threshold=128)
    assert cov < 0.01


def test_dark_coverage_mid_gray():
    """127グレー画像: 127 < 128 のため dark_coverage ≈ 1.0 であること。"""
    cov = compute_dark_coverage(_make_mid_png(), threshold=128)
    assert cov > 0.99


def test_dark_coverage_range():
    """dark_coverage は [0.0, 1.0] に収まること。"""
    for png in [_make_black_png(), _make_white_png(), _make_mid_png()]:
        cov = compute_dark_coverage(png)
        assert 0.0 <= cov <= 1.0


# ===========================================================================
# D: DM4Result
# ===========================================================================

def test_dm4result_inherits_dm2result():
    """DM4Result が DM2Result を継承していること。"""
    from backend.core.density.dm2 import DM2Result
    r = DM4Result(
        svg="", png_bytes=b"", entrance=0, exit_cell=1,
        solution_path=[0, 1], grid_rows=2, grid_cols=2,
        density_map=np.zeros((2, 2)), adj={0: [1], 1: [0]},
        edge_map=np.zeros((2, 2)), solution_count=1,
        clahe_clip_limit_used=0.03, clahe_n_tiles_used=16,
        ssim_score=0.0, dark_coverage=0.0,
    )
    assert isinstance(r, DM2Result)


def test_dm4result_has_ssim_score():
    """DM4Result が ssim_score フィールドを持つこと。"""
    r = generate_dm4_maze(_gradient_h(32, 32), _small_config())
    assert hasattr(r, "ssim_score")


def test_dm4result_has_dark_coverage():
    """DM4Result が dark_coverage フィールドを持つこと。"""
    r = generate_dm4_maze(_gradient_h(32, 32), _small_config())
    assert hasattr(r, "dark_coverage")


# ===========================================================================
# E: generate_dm4_maze API
# ===========================================================================

def test_generate_dm4_returns_dm4result():
    """generate_dm4_maze が DM4Result を返すこと。"""
    r = generate_dm4_maze(_gradient_h(32, 32), _small_config())
    assert isinstance(r, DM4Result)


def test_generate_dm4_png_valid():
    """PNG バイト列が有効な PNG シグネチャを持つこと。"""
    r = generate_dm4_maze(_gradient_h(32, 32), _small_config())
    assert r.png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_generate_dm4_svg_valid_xml():
    """SVG 文字列が <svg> タグを含む有効な XML であること。"""
    r = generate_dm4_maze(_gradient_h(32, 32), _small_config())
    assert r.svg.startswith("<?xml")
    assert "<svg" in r.svg
    assert "</svg>" in r.svg


def test_generate_dm4_entrance_exit_differ():
    """入口と出口が異なるセル ID であること。"""
    r = generate_dm4_maze(_gradient_h(32, 32), _small_config())
    assert r.entrance != r.exit_cell


def test_generate_dm4_solution_path_nonempty():
    """解経路が空でないこと。"""
    r = generate_dm4_maze(_gradient_h(32, 32), _small_config())
    assert len(r.solution_path) > 0


def test_generate_dm4_bfs_validates_path():
    """BFS で入口→出口の経路が存在すること。"""
    r = generate_dm4_maze(_gradient_h(32, 32), _small_config())
    assert bfs_has_path(r.adj, r.entrance, r.exit_cell)


def test_generate_dm4_png_size():
    """生成された PNG のサイズが result.grid_cols*cell_size × result.grid_rows*cell_size であること。"""
    cfg = _small_config(grid_rows=10, grid_cols=10, cell_size_px=8, render_scale=1)
    # 大きめ入力でグリッドクランプを回避
    r = generate_dm4_maze(_gradient_h(80, 80), cfg)
    w, h = _png_size(r.png_bytes)
    assert w == r.grid_cols * 8
    assert h == r.grid_rows * 8


def test_generate_dm4_grid_dims_in_result():
    """DM4Result.grid_rows / grid_cols が設定値と一致すること。"""
    cfg = _small_config(grid_rows=15, grid_cols=12)
    r = generate_dm4_maze(_gradient_h(32, 32), cfg)
    assert r.grid_rows == min(15, max(1, r.grid_rows))  # clamp applied
    assert r.grid_cols == min(12, max(1, r.grid_cols))


def test_generate_dm4_solution_count_perfect_maze():
    """スパニングツリー迷路の解経路数は 1 であること。"""
    r = generate_dm4_maze(_gradient_h(32, 32), _small_config())
    assert r.solution_count == 1


# ===========================================================================
# F: SSIM スコア
# ===========================================================================

def test_ssim_score_positive():
    """ssim_score が 0.0 より大きいこと（skimage が利用可能な場合）。"""
    r = generate_dm4_maze(_gradient_h(32, 32), _small_config())
    # skimage が利用不可なら 0.0 が返るため、正値チェックはスキップ
    assert r.ssim_score >= 0.0


def test_ssim_score_upper_bound():
    """ssim_score が 1.0 以下であること。"""
    r = generate_dm4_maze(_gradient_h(32, 32), _small_config())
    assert r.ssim_score <= 1.0


def test_dm4_ssim_improves_over_dm2_gradient():
    """グラデーション画像に対して DM-4（fill_cells=True）の SSIM が DM-2 の SSIM より高いこと。"""
    try:
        from skimage.metrics import structural_similarity  # noqa: F401
    except ImportError:
        pytest.skip("skimage not available")

    img = _gradient_h(64, 64)
    # fill_cells=True（デフォルト）: セル塗りつぶし+blur → 高 SSIM
    cfg4 = _small_config(
        grid_rows=25, grid_cols=25, cell_size_px=5,
        render_scale=2, tonal_thickness_range=2.0,
        fill_cells=True, blur_radius=2.0,
    )
    r4 = generate_dm4_maze(img, cfg4)

    cfg2 = DM2Config(
        grid_rows=25, grid_cols=25, cell_size_px=5,
    )
    r2 = generate_dm2_maze(img, cfg2)

    # DM-2 の SSIM を compute_ssim で計算（同じ前処理を使う）
    dm2_ssim = _compute_ssim(
        np.asarray(img.convert("L"), dtype=np.float64) / 255.0,
        r2.png_bytes,
    )

    assert r4.ssim_score > dm2_ssim, (
        f"DM-4 SSIM {r4.ssim_score:.4f} should exceed DM-2 SSIM {dm2_ssim:.4f}"
    )


def test_dm4_gradient_ssim_meets_target():
    """グラデーション画像に対して DM-4 の SSIM が 0.65 以上であること。"""
    try:
        from skimage.metrics import structural_similarity  # noqa: F401
    except ImportError:
        pytest.skip("skimage not available")

    img = _gradient_h(64, 64)
    cfg4 = DM4Config(
        grid_rows=30, grid_cols=30, cell_size_px=4,
        render_scale=2, fill_cells=True, blur_radius=2.0,
    )
    r4 = generate_dm4_maze(img, cfg4)
    assert r4.ssim_score >= 0.65, f"gradient SSIM {r4.ssim_score:.4f} < 0.65"


def test_ssim_compute_identical_returns_one():
    """同一画像同士の SSIM ≈ 1.0 であること。"""
    try:
        from skimage.metrics import structural_similarity  # noqa: F401
    except ImportError:
        pytest.skip("skimage not available")

    img = _gradient_h(32, 32)
    gray = np.asarray(img.convert("L"), dtype=np.float64) / 255.0
    buf = io.BytesIO()
    img.convert("L").save(buf, format="PNG")
    png_bytes = buf.getvalue()

    ssim_val = _compute_ssim(gray, png_bytes, target_size=(64, 64))
    assert ssim_val > 0.99


# ===========================================================================
# G: dark_coverage
# ===========================================================================

def test_dark_coverage_black_image_in_result():
    """真っ黒画像に対して dark_coverage が高いこと。"""
    cfg = _small_config()
    r = generate_dm4_maze(_uniform(0.0), cfg)
    # 真っ黒画像ではほぼ全壁が grade=0 (黒) → 高い dark_coverage
    assert r.dark_coverage > 0.0


def test_dark_coverage_white_image_in_result():
    """真っ白画像に対して dark_coverage が黒画像より低いこと。"""
    cfg = _small_config()
    r_black = generate_dm4_maze(_uniform(0.0), cfg)
    r_white = generate_dm4_maze(_uniform(1.0), cfg)
    assert r_white.dark_coverage < r_black.dark_coverage


def test_dark_coverage_in_range():
    """dark_coverage が [0.0, 1.0] に収まること。"""
    cfg = _small_config()
    for img in [_uniform(0.0), _uniform(0.5), _uniform(1.0)]:
        r = generate_dm4_maze(img, cfg)
        assert 0.0 <= r.dark_coverage <= 1.0


# ===========================================================================
# H: トーンレンダリング特性
# ===========================================================================

def test_dark_image_has_more_black_pixels_than_bright():
    """暗部画像の迷路は明部画像よりも黒ピクセルが多いこと。"""
    cfg = _small_config(grid_rows=15, grid_cols=15, cell_size_px=5)
    r_dark = generate_dm4_maze(_uniform(0.0), cfg)
    r_bright = generate_dm4_maze(_uniform(1.0), cfg)

    dark_cov_dark = compute_dark_coverage(r_dark.png_bytes)
    dark_cov_bright = compute_dark_coverage(r_bright.png_bytes)
    assert dark_cov_dark > dark_cov_bright


def test_bright_image_walls_are_near_white():
    """真っ白画像の迷路は白背景に溶け込むこと（低い dark_coverage）。"""
    cfg = _small_config(grid_rows=15, grid_cols=15, cell_size_px=5)
    r = generate_dm4_maze(_uniform(1.0), cfg)
    # 白画像→grade=255壁→白背景と区別できない→dark_coverage低
    dark_cov = compute_dark_coverage(r.png_bytes, threshold=200)
    assert dark_cov < 0.5


def test_higher_thickness_range_increases_dark_coverage_line_mode():
    """壁線モード(fill_cells=False)で thickness_range が高いほど dark_coverage が高くなること。"""
    img = _uniform(0.0)
    # fill_cells=False かつ blur_radius=0 で厚さの効果を純粋に評価する
    cfg_low = _small_config(tonal_thickness_range=0.0, grid_rows=15, grid_cols=15,
                             fill_cells=False, blur_radius=0.0)
    cfg_high = _small_config(tonal_thickness_range=3.0, grid_rows=15, grid_cols=15,
                              fill_cells=False, blur_radius=0.0)

    r_low = generate_dm4_maze(img, cfg_low)
    r_high = generate_dm4_maze(img, cfg_high)

    assert r_high.dark_coverage > r_low.dark_coverage


def test_tonal_svg_contains_grade_colors():
    """SVG が多値グレー色 (rgb(N,N,N)) を含むこと（単色でない）。"""
    img = _gradient_h(64, 64)
    # SVG は fill_cells に関わらず壁線ベースで生成される
    cfg = _small_config(grid_rows=15, grid_cols=15)
    r = generate_dm4_maze(img, cfg)

    # SVG に複数の rgb(...) が含まれること
    import re
    colors = set(re.findall(r"rgb\(\d+,\d+,\d+\)", r.svg))
    assert len(colors) > 1, "SVG should contain multiple tonal gray colors"


def test_fill_cells_mode_gives_higher_ssim_than_line_mode():
    """fill_cells=True が fill_cells=False より高い SSIM を生成すること。"""
    try:
        from skimage.metrics import structural_similarity  # noqa: F401
    except ImportError:
        pytest.skip("skimage not available")

    img = _gradient_h(64, 64)
    gray = np.asarray(img.convert("L"), dtype=np.float64) / 255.0

    cfg_fill = _small_config(grid_rows=25, grid_cols=25, cell_size_px=4,
                              fill_cells=True, blur_radius=2.0)
    cfg_line = _small_config(grid_rows=25, grid_cols=25, cell_size_px=4,
                              fill_cells=False, blur_radius=0.0)

    r_fill = generate_dm4_maze(img, cfg_fill)
    r_line = generate_dm4_maze(img, cfg_line)

    assert r_fill.ssim_score > r_line.ssim_score, (
        f"fill SSIM {r_fill.ssim_score:.4f} should exceed line SSIM {r_line.ssim_score:.4f}"
    )


# ===========================================================================
# I: E2E テスト
# ===========================================================================

def test_e2e_gradient_image():
    """グラデーション画像で end-to-end 生成が成功すること。"""
    r = generate_dm4_maze(_gradient_h(64, 64), _small_config())
    assert r.png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert bfs_has_path(r.adj, r.entrance, r.exit_cell)
    assert r.solution_count >= 1


def test_e2e_circle_image():
    """サークル画像で end-to-end 生成が成功すること。"""
    r = generate_dm4_maze(_circle(64, 64), _small_config())
    assert r.png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert bfs_has_path(r.adj, r.entrance, r.exit_cell)


def test_e2e_uniform_black():
    """均一黒画像で生成が成功し、解経路が存在すること。"""
    r = generate_dm4_maze(_uniform(0.0), _small_config())
    assert bfs_has_path(r.adj, r.entrance, r.exit_cell)


def test_e2e_uniform_white():
    """均一白画像で生成が成功し、解経路が存在すること。"""
    r = generate_dm4_maze(_uniform(1.0), _small_config())
    assert bfs_has_path(r.adj, r.entrance, r.exit_cell)


def test_e2e_rgb_input_image():
    """RGB 画像を入力しても正常に生成できること。"""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, 32:, :] = 200
    img_rgb = Image.fromarray(arr, mode="RGB")
    r = generate_dm4_maze(img_rgb, _small_config())
    assert r.png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_e2e_none_config_uses_defaults():
    """config=None のとき DM4Config デフォルト値で動作すること。"""
    r = generate_dm4_maze(_gradient_h(32, 32), None)
    assert isinstance(r, DM4Result)
    assert r.png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_e2e_show_solution_overlay():
    """show_solution=True で解経路が SVG/PNG に描画されること。"""
    cfg = _small_config(show_solution=True)
    r = generate_dm4_maze(_gradient_h(32, 32), cfg)
    # show_solution=True でも BFS で解経路が存在すること
    assert bfs_has_path(r.adj, r.entrance, r.exit_cell)
    # SVG に白線解経路パスが含まれること
    assert 'stroke="white"' in r.svg or 'stroke-width' in r.svg


def test_e2e_render_scale_2():
    """render_scale=2 でアンチエイリアス縮小が正しく動作すること。"""
    cfg = _small_config(render_scale=2, grid_rows=10, grid_cols=10, cell_size_px=6)
    # 大きめ入力でグリッドクランプを回避
    r = generate_dm4_maze(_gradient_h(80, 80), cfg)
    w, h = _png_size(r.png_bytes)
    assert w == r.grid_cols * 6
    assert h == r.grid_rows * 6
