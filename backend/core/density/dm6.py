"""
DM-6: Bayesian最適化 + 難易度制御（Phase DM-6）

設計書: docs/maze-artisan-masterpiece-requirements.md Phase DM-6

DM-4/DM-5 との関係:
  - DM-6 は DM-4 を基盤とする難易度制御レイヤー（印刷最適化 DM-5 と直交）
  - difficulty: "easy"/"medium"/"hard"/"extreme" → grid_size + extra_removal_rate 自動設定
  - difficulty_score (float 0.0-1.0) → difficulty ラベルに変換可能
  - プリセット名（portrait/landscape/logo/anime）との組み合わせ対応

成功基準:
  easy:    grid_size=6,  extra_removal_rate=0.40
  medium:  grid_size=10, extra_removal_rate=0.15
  hard:    grid_size=14, extra_removal_rate=0.05
  extreme: grid_size=16, extra_removal_rate=0.00
"""
from __future__ import annotations

import dataclasses
import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .dm4 import DM4Config, DM4Result, generate_dm4_maze


# ---------------------------------------------------------------------------
# 難易度定数
# ---------------------------------------------------------------------------

# difficulty → (grid_size, extra_removal_rate)
DIFFICULTY_PARAMS: Dict[str, Dict] = {
    "easy":    {"grid_size": 6,  "extra_removal_rate": 0.40},
    "medium":  {"grid_size": 10, "extra_removal_rate": 0.15},
    "hard":    {"grid_size": 14, "extra_removal_rate": 0.05},
    "extreme": {"grid_size": 16, "extra_removal_rate": 0.00},
}

# difficulty → 代表 difficulty_score
_DIFFICULTY_SCORE_CENTER: Dict[str, float] = {
    "easy":    0.125,
    "medium":  0.375,
    "hard":    0.625,
    "extreme": 0.875,
}

# difficulty_score 区間境界
_DIFFICULTY_THRESHOLDS: Tuple[float, ...] = (0.25, 0.50, 0.75)

# プリセット名セット（dm6_optimizer.py と共有）
VALID_PRESETS = frozenset({"portrait", "landscape", "logo", "anime"})

VALID_DIFFICULTIES = frozenset(DIFFICULTY_PARAMS.keys())


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _score_to_difficulty(score: float) -> str:
    """
    difficulty_score (0.0-1.0) を difficulty ラベルに変換する。

    区間:
      [0.00, 0.25) → "easy"
      [0.25, 0.50) → "medium"
      [0.50, 0.75) → "hard"
      [0.75, 1.00] → "extreme"
    """
    score = float(np.clip(score, 0.0, 1.0))
    t0, t1, t2 = _DIFFICULTY_THRESHOLDS
    if score < t0:
        return "easy"
    elif score < t1:
        return "medium"
    elif score < t2:
        return "hard"
    else:
        return "extreme"


def _difficulty_to_score(difficulty: str) -> float:
    """difficulty ラベルを代表 difficulty_score に変換する。"""
    return _DIFFICULTY_SCORE_CENTER.get(difficulty, 0.375)


# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

@dataclass
class DM6Config(DM4Config):
    """
    DM-6 設定。DM4Config を継承し、難易度制御パラメータを追加。

    主要パラメータ:
        difficulty       : "easy"/"medium"/"hard"/"extreme"（デフォルト: "medium"）
        difficulty_score : 0.0-1.0（指定時は difficulty より優先）
        extra_removal_rate: ループ生成率（0.0=完全スパニングツリー, 1.0=最多ループ）
        preset_name      : カテゴリプリセット名（"portrait"等、任意）

    Note: difficulty/difficulty_score が指定された場合、generate_dm6_maze() が
    grid_rows/grid_cols/extra_removal_rate を自動的に上書きする。
    """
    difficulty: str = "medium"
    difficulty_score: Optional[float] = None
    extra_removal_rate: float = 0.15   # medium デフォルト
    preset_name: Optional[str] = None


# ---------------------------------------------------------------------------
# 結果
# ---------------------------------------------------------------------------

@dataclass
class DM6Result(DM4Result):
    """DM-6 生成結果。DM-4 互換フィールド + 難易度情報。"""
    difficulty: str = "medium"
    difficulty_score: float = 0.375
    extra_removal_rate: float = 0.15
    preset_name: Optional[str] = None


# ---------------------------------------------------------------------------
# メイン API
# ---------------------------------------------------------------------------

def generate_dm6_maze(
    image: Image.Image,
    config: Optional[DM6Config] = None,
) -> DM6Result:
    """
    DM-6 難易度制御迷路を生成する。

    difficulty または difficulty_score から grid_size と extra_removal_rate を
    自動決定し、DM-4 パイプラインで迷路を生成する。

    Args:
        image : 入力画像（任意サイズ・任意形式）。
        config: DM6Config。None の場合はデフォルト値を使用。

    Returns:
        DM6Result（DM4Result 互換フィールド + 難易度情報）

    Raises:
        ValueError: 無効な difficulty が指定された場合。
    """
    if config is None:
        config = DM6Config()

    # ------------------------------------------------------------------
    # passage_ratio バリデーション（0.1〜0.8）
    # ------------------------------------------------------------------
    if not (0.1 <= config.passage_ratio <= 0.8):
        raise ValueError(
            f"passage_ratio must be between 0.1 and 0.8, got {config.passage_ratio}"
        )

    # ------------------------------------------------------------------
    # difficulty の解決（difficulty_score が優先）
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

    # ------------------------------------------------------------------
    # 難易度パラメータの適用
    # ------------------------------------------------------------------
    params = DIFFICULTY_PARAMS[difficulty]
    grid_size = params["grid_size"]
    extra_removal_rate = params["extra_removal_rate"]

    # grid_rows/grid_cols/extra_removal_rate を難易度値で上書き
    dm4_config = dataclasses.replace(
        config,
        grid_rows=grid_size,
        grid_cols=grid_size,
        extra_removal_rate=extra_removal_rate,
    )

    # ------------------------------------------------------------------
    # DM-4 パイプライン実行
    # ------------------------------------------------------------------
    dm4_result = generate_dm4_maze(image, dm4_config)

    return DM6Result(
        # DM-4 継承フィールド
        svg=dm4_result.svg,
        png_bytes=dm4_result.png_bytes,
        entrance=dm4_result.entrance,
        exit_cell=dm4_result.exit_cell,
        solution_path=dm4_result.solution_path,
        grid_rows=dm4_result.grid_rows,
        grid_cols=dm4_result.grid_cols,
        density_map=dm4_result.density_map,
        adj=dm4_result.adj,
        edge_map=dm4_result.edge_map,
        solution_count=dm4_result.solution_count,
        clahe_clip_limit_used=dm4_result.clahe_clip_limit_used,
        clahe_n_tiles_used=dm4_result.clahe_n_tiles_used,
        ssim_score=dm4_result.ssim_score,
        dark_coverage=dm4_result.dark_coverage,
        # DM-6 追加フィールド
        difficulty=difficulty,
        difficulty_score=difficulty_score,
        extra_removal_rate=extra_removal_rate,
        preset_name=config.preset_name,
    )
