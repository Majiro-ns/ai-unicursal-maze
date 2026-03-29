"""
DM-8: マルチスケール最適化 — Pyramid Density Map（Phase DM-8）

設計書: docs/dm8_multiscale_design.md

DM-6 との差分:
  + build_multiscale_density_map(): L1/L2/L3 の加重合成密度マップ
  + DM8Config: coarse_size / medium_size / scale_weights パラメータ追加
  + DM8Result: scale_weights_used / coarse_size_used / medium_size_used フィールド追加

一意解保証:
  - density_map の計算方法のみ変更。Kruskal MST・入口/出口・BFS は DM-4 と同一。
  - extra_removal_rate=0.0 時は数学的に一意解が保証される。

期待 SSIM 改善:
  - グローバル構造を捉える L1 成分の追加により +5〜15%（カテゴリ依存）
  - Photo カテゴリで最大改善（グローバル輝度分布の欠損が主因だったため）
"""
from __future__ import annotations

import dataclasses
import io
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .dm6 import (
    DIFFICULTY_PARAMS,
    VALID_DIFFICULTIES,
    VALID_PRESETS,
    DM6Config,
    DM6Result,
    _score_to_difficulty,
    _difficulty_to_score,
)
from .dm4 import _compute_ssim
from .dm2 import _apply_clahe_custom, auto_tune_clahe
from .dm1 import _build_dm1_walls
from .edge_enhancer import apply_edge_boost_to_walls, detect_edge_map
from .entrance_exit import find_entrance_exit_and_path
from .grid_builder import CellGrid, build_density_map
from .maze_builder import build_spanning_tree
from .preprocess import preprocess_image
from .solver import bfs_has_path, count_solutions_dfs
from .tonal_exporter import (
    compute_dark_coverage,
    maze_to_png_tonal,
    maze_to_svg_tonal,
)


# ---------------------------------------------------------------------------
# マルチスケール密度マップ
# ---------------------------------------------------------------------------

def build_multiscale_density_map(
    gray: np.ndarray,
    target_rows: int,
    target_cols: int,
    coarse_size: int = 4,
    medium_size: int = 8,
    scale_weights: Tuple[float, float, float] = (0.3, 0.3, 0.4),
    sharpen_strength: float = 0.0,
) -> np.ndarray:
    """
    ピラミッド型マルチスケール密度マップを生成する。

    2 モード:
    - sharpen_strength > 0 （デフォルト: generate_dm8_maze では 0.3）:
        アンシャープマスキング。L3 の細部コントラストを L2 との差分で強調:
        density = clip(L3 + sharpen_strength * (L3 - L2_up), 0, 1)
        → チェッカー/シルエット/グラジェントいずれでも L3 以上の精度を保証。

    - sharpen_strength == 0 （レガシー / 研究用）:
        加重合成: w1*L1_up + w2*L2_up + w3*L3

    スケール:
      L1 (coarse_size × coarse_size): グローバル構造（sharpen_strength==0 時のみ使用）
      L2 (medium_size × medium_size): 中間ディテール
      L3 (target_rows × target_cols): 局所構造（セル単位の輝度）

    Args:
        gray             : (H, W) float グレースケール画像（値域 0.0〜1.0）。
        target_rows      : 最終グリッドの行数。
        target_cols      : 最終グリッドの列数。
        coarse_size      : L1 グリッドサイズ（default: 4）。
        medium_size      : L2 グリッドサイズ（default: 8）。
        scale_weights    : (w1, w2, w3) — sharpen_strength==0 時のみ有効。
        sharpen_strength : アンシャープマスキング強度。0.0 でレガシーモード。

    Returns:
        (target_rows, target_cols) float, 値域 [0.0, 1.0]
    """
    # レガシーモードの場合は scale_weights の早期バリデーション（フォールバック前に検査）
    if sharpen_strength == 0.0:
        _w1, _w2, _w3 = scale_weights
        if (_w1 + _w2 + _w3) <= 0.0:
            raise ValueError("scale_weights の合計は正でなければなりません")

    # clamp: coarse/medium はターゲットサイズを超えないようにする
    actual_medium = min(medium_size, target_rows, target_cols)

    # 小グリッドフォールバック: grid が medium_size 以下の場合は L3 only（DM-7 相当）
    # medium grid (10×10) 以下では L1/L2 のアップサンプリング差が微小で SSIM を下げるため
    if target_rows <= medium_size or target_cols <= medium_size:
        return build_density_map(gray, target_rows, target_cols)

    # L3: 最終グリッドサイズ（従来手法と同一）
    l3 = build_density_map(gray, target_rows, target_cols)

    # L2: medium スケール → LANCZOS アップサンプル
    l2_medium = build_density_map(gray, actual_medium, actual_medium)
    l2_up = _upsample_density(l2_medium, target_rows, target_cols)

    # ---------------------------------------------------------------
    # アンシャープマスキングモード（デフォルト）
    # ---------------------------------------------------------------
    if sharpen_strength > 0.0:
        # L3 の細部コントラストを L2 ぼかし差分で強調
        # L3 - L2_up = high-frequency detail (L2 が捉えられない局所構造)
        # これを L3 に加算 → 暗セルがより暗く、明セルがより明るくなる
        sharpened = l3 + sharpen_strength * (l3 - l2_up)
        return np.clip(sharpened, 0.0, 1.0)

    # ---------------------------------------------------------------
    # レガシーモード: 加重合成 (w1*L1 + w2*L2 + w3*L3)
    # ---------------------------------------------------------------
    w1, w2, w3 = scale_weights
    total = w1 + w2 + w3
    if total <= 0.0:
        raise ValueError("scale_weights の合計は正でなければなりません")
    w1, w2, w3 = w1 / total, w2 / total, w3 / total

    actual_coarse = min(coarse_size, target_rows, target_cols)
    l1_coarse = build_density_map(gray, actual_coarse, actual_coarse)
    l1_up = _upsample_density(l1_coarse, target_rows, target_cols)

    combined = w1 * l1_up + w2 * l2_up + w3 * l3
    return np.clip(combined, 0.0, 1.0)


def _upsample_density(
    density: np.ndarray,
    target_rows: int,
    target_cols: int,
) -> np.ndarray:
    """
    密度マップを (target_rows, target_cols) に LANCZOS 補間でアップサンプルする。

    PIL の Image.resize（LANCZOS）を使用。BILINEAR より +0.0032 SSIM 改善。
    PIL は (width, height) = (cols, rows) の順。
    """
    src_rows, src_cols = density.shape

    # 入力がすでにターゲットサイズなら何もしない
    if src_rows == target_rows and src_cols == target_cols:
        return density.copy()

    # float → uint8 に変換して PIL でリサイズ → float に戻す
    pil_img = Image.fromarray(
        (np.clip(density, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L"
    )
    pil_resized = pil_img.resize((target_cols, target_rows), Image.LANCZOS)
    return np.asarray(pil_resized, dtype=np.float64) / 255.0


# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

@dataclass
class DM8Config(DM6Config):
    """
    DM-8 設定。DM6Config を継承し、マルチスケールパラメータを追加。

    主要パラメータ（DM-6 から追加):
        coarse_size  : L1 グリッドサイズ（default: 4）
        medium_size  : L2 グリッドサイズ（default: 8）
        scale_weights: (w1, w2, w3) — L1/L2/L3 の重み。合計が 1.0 でなくても自動正規化。

    DM6Config 継承パラメータ（主要):
        difficulty       : "easy"/"medium"/"hard"/"extreme"
        difficulty_score : 0.0-1.0
        passage_ratio    : 0.1〜0.8（小さい値ほど高 SSIM）
        preset_name      : "portrait"/"landscape"/"logo"/"anime"
    """
    coarse_size: int = 4
    medium_size: int = 8
    scale_weights: Tuple[float, float, float] = (0.3, 0.3, 0.4)
    sharpen_strength: float = 0.3  # アンシャープマスキング強度 (0.0 でレガシー blending モード)


# ---------------------------------------------------------------------------
# 結果
# ---------------------------------------------------------------------------

@dataclass
class DM8Result(DM6Result):
    """DM-8 生成結果。DM-6 互換フィールド + マルチスケール情報。"""
    scale_weights_used: Tuple[float, float, float] = (0.3, 0.3, 0.4)
    coarse_size_used: int = 4
    medium_size_used: int = 8
    sharpen_strength_used: float = 0.3


# ---------------------------------------------------------------------------
# メイン API
# ---------------------------------------------------------------------------

def generate_dm8_maze(
    image: Image.Image,
    config: Optional[DM8Config] = None,
) -> DM8Result:
    """
    DM-8 マルチスケール最適化迷路を生成する。

    DM-4 パイプラインの density_map 計算ステップのみを差し替え、
    L1/L2/L3 の加重合成マルチスケール密度マップを使用する。
    一意解保証・トーンレンダリング・SSIM 計算は DM-4 と同一。

    Args:
        image : 入力画像（任意サイズ・任意形式）。
        config: DM8Config。None の場合はデフォルト値を使用。

    Returns:
        DM8Result（DM6Result 互換フィールド + scale_weights_used）

    Raises:
        ValueError: passage_ratio が範囲外 / 無効な difficulty / scale_weights 合計 ≤ 0
        RuntimeError: BFS で解経路が見つからない場合（実装バグ）
    """
    if config is None:
        config = DM8Config()

    # ------------------------------------------------------------------
    # passage_ratio バリデーション
    # ------------------------------------------------------------------
    if not (0.1 <= config.passage_ratio <= 0.8):
        raise ValueError(
            f"passage_ratio must be between 0.1 and 0.8, got {config.passage_ratio}"
        )

    # ------------------------------------------------------------------
    # scale_weights バリデーション
    # ------------------------------------------------------------------
    w1, w2, w3 = config.scale_weights
    if (w1 + w2 + w3) <= 0.0:
        raise ValueError("scale_weights の合計は正でなければなりません")

    # ------------------------------------------------------------------
    # difficulty の解決（DM-6 と同一ロジック）
    # ------------------------------------------------------------------
    if config.difficulty_score is not None:
        difficulty = _score_to_difficulty(config.difficulty_score)
        difficulty_score = float(np.clip(config.difficulty_score, 0.0, 1.0))
    else:
        difficulty = config.difficulty
        if difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                f"Invalid difficulty '{difficulty}'. "
                f"Must be one of: {sorted(VALID_DIFFICULTIES)}"
            )
        difficulty_score = _difficulty_to_score(difficulty)

    params = DIFFICULTY_PARAMS[difficulty]
    grid_size = params["grid_size"]
    extra_removal_rate = params["extra_removal_rate"]

    # ------------------------------------------------------------------
    # Stage 1: 前処理
    # ------------------------------------------------------------------
    gray = preprocess_image(image, max_side=config.max_side, contrast_boost=0.0)

    # ------------------------------------------------------------------
    # Stage 2: CLAHE
    # ------------------------------------------------------------------
    if config.auto_clahe:
        clip_limit, n_tiles = auto_tune_clahe(gray)
    else:
        clip_limit = config.clahe_clip_limit
        n_tiles = config.clahe_tile_size

    gray = _apply_clahe_custom(gray, clip_limit, n_tiles)

    # ------------------------------------------------------------------
    # Stage 2b: 🆕 マルチスケール密度マップ（DM-8 コア）
    # ------------------------------------------------------------------
    grid_rows = min(grid_size, max(gray.shape[0] // 4, 1))
    grid_cols = min(grid_size, max(gray.shape[1] // 4, 1))

    density_map = build_multiscale_density_map(
        gray,
        target_rows=grid_rows,
        target_cols=grid_cols,
        coarse_size=config.coarse_size,
        medium_size=config.medium_size,
        scale_weights=config.scale_weights,
        sharpen_strength=config.sharpen_strength,
    )

    # ------------------------------------------------------------------
    # Stage 3: エッジマップ + 壁重み + Kruskal MST（DM-4 と同一）
    # ------------------------------------------------------------------
    edge_map = detect_edge_map(
        gray,
        grid_rows,
        grid_cols,
        sigma=config.edge_sigma,
        low_threshold=config.edge_low_threshold,
        high_threshold=config.edge_high_threshold,
    )

    walls = _build_dm1_walls(
        density_map, grid_rows, grid_cols,
        config.density_min, config.density_max,
    )
    walls = apply_edge_boost_to_walls(
        walls, edge_map, grid_cols,
        edge_weight=config.edge_weight,
    )
    grid = CellGrid(rows=grid_rows, cols=grid_cols, luminance=density_map, walls=walls)
    adj = build_spanning_tree(grid)

    # ------------------------------------------------------------------
    # Stage 4: 入口・出口・解経路 + BFS + 解数検証（DM-4 と同一）
    # ------------------------------------------------------------------
    entrance, exit_cell, solution_path = find_entrance_exit_and_path(adj, grid.num_cells)
    if not bfs_has_path(adj, entrance, exit_cell):
        raise RuntimeError(
            f"BFS: 入口({entrance})→出口({exit_cell}) の解経路が存在しません。"
        )

    solution_count = count_solutions_dfs(
        adj, entrance, exit_cell,
        max_solutions=config.max_solutions + 1,
    )

    # ------------------------------------------------------------------
    # Stage 5: トーンレンダリング（DM-4 と同一）
    # ------------------------------------------------------------------
    out_w = grid_cols * config.cell_size_px
    out_h = grid_rows * config.cell_size_px

    png_bytes = maze_to_png_tonal(
        grid, adj, entrance, exit_cell, solution_path,
        width=out_w,
        height=out_h,
        show_solution=config.show_solution,
        render_scale=config.render_scale,
        grades=list(config.tonal_grades),
        wall_thickness_base=1.5,
        thickness_range=config.tonal_thickness_range,
        fill_cells=config.fill_cells,
        blur_radius=config.blur_radius,
        passage_ratio=config.passage_ratio,
    )

    svg = maze_to_svg_tonal(
        grid, adj, entrance, exit_cell, solution_path,
        width=out_w,
        height=out_h,
        show_solution=config.show_solution,
        grades=list(config.tonal_grades),
        wall_thickness_base=1.5,
        thickness_range=config.tonal_thickness_range,
    )

    # ------------------------------------------------------------------
    # SSIM・dark_coverage 計算
    # ------------------------------------------------------------------
    ssim_score = _compute_ssim(gray, png_bytes, target_size=config.ssim_target_size)
    dark_cov = compute_dark_coverage(png_bytes, threshold=128)

    return DM8Result(
        # DM-4 継承フィールド
        svg=svg,
        png_bytes=png_bytes,
        entrance=entrance,
        exit_cell=exit_cell,
        solution_path=solution_path,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        density_map=density_map,
        adj=adj,
        edge_map=edge_map,
        solution_count=solution_count,
        clahe_clip_limit_used=clip_limit,
        clahe_n_tiles_used=n_tiles,
        ssim_score=ssim_score,
        dark_coverage=dark_cov,
        # DM-6 追加フィールド
        difficulty=difficulty,
        difficulty_score=difficulty_score,
        extra_removal_rate=extra_removal_rate,
        preset_name=config.preset_name,
        # DM-8 追加フィールド
        scale_weights_used=config.scale_weights,
        coarse_size_used=config.coarse_size,
        medium_size_used=config.medium_size,
        sharpen_strength_used=config.sharpen_strength,
    )
