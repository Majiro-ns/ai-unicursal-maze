"""
DM-6 Bayesian最適化 (optuna) — SSIM最大化パラメータ探索（Phase DM-6）

設計書: docs/maze-artisan-masterpiece-requirements.md Phase DM-6

機能:
  - optuna TPE Sampler (Bayesian) で SSIM を最大化するパラメータを探索
  - パラメータ空間: grid_size/thickness/extra_removal/edge_weight/dark_threshold/tonal_levels
  - 画像カテゴリ別プリセット4種（portrait/landscape/logo/anime）
  - 最適パラメータを JSON シリアライズ可能な dict で返却

使用方法:
    from backend.core.density.dm6_optimizer import optimize_for_image, generate_preset
    result = optimize_for_image(image, n_trials=100, category="portrait")
    # result = {"category": "portrait", "best_params": {...}, "best_value": 0.71, "n_trials": 100}
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# カテゴリ別パラメータ制約
# ---------------------------------------------------------------------------

# 各カテゴリの最適化空間制約
# portrait:  細部重視 → 高 tonal_levels, 細かい grid_size
# landscape: 広域構造 → 中間 grid_size, 中間 tonal_levels
# logo:      高コントラスト → 低 tonal_levels（2値近傍）, シンプルグリッド
# anime:     輪郭強調 → 高 edge_weight, 中間 tonal_levels
CATEGORY_CONSTRAINTS: Dict[str, Dict[str, Any]] = {
    "portrait": {
        "grid_size":    (8,  16),
        "thickness":    (1.5, 3.0),
        "extra_removal":(0.0, 0.5),
        "edge_weight":  (0.3, 1.0),
        "dark_threshold":(0.1, 0.4),
        "tonal_levels": (8,  16),
    },
    "landscape": {
        "grid_size":    (4,  12),
        "thickness":    (1.0, 2.5),
        "extra_removal":(0.0, 0.8),
        "edge_weight":  (0.0, 0.6),
        "dark_threshold":(0.15, 0.5),
        "tonal_levels": (4,  12),
    },
    "logo": {
        "grid_size":    (4,  10),
        "thickness":    (1.0, 2.0),
        "extra_removal":(0.0, 0.3),
        "edge_weight":  (0.5, 1.0),
        "dark_threshold":(0.1, 0.35),
        "tonal_levels": (2,   6),
    },
    "anime": {
        "grid_size":    (6,  14),
        "thickness":    (1.5, 3.0),
        "extra_removal":(0.0, 0.4),
        "edge_weight":  (0.4, 1.0),
        "dark_threshold":(0.1, 0.4),
        "tonal_levels": (6,  14),
    },
}

VALID_CATEGORIES = frozenset(CATEGORY_CONSTRAINTS.keys())


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _levels_to_grades(n: int) -> List[int]:
    """
    tonal_levels (int) を TONAL_GRADES 形式のリストに変換する。

    例: n=2 → [0, 255]
        n=4 → [0, 85, 170, 255]
        n=8 → [0, 36, 73, 109, 146, 182, 219, 255]
    """
    n = max(2, int(n))
    return [round(i * 255 / (n - 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# メイン最適化 API
# ---------------------------------------------------------------------------

def optimize_for_image(
    image: Image.Image,
    n_trials: int = 100,
    category: str = "portrait",
    seed: Optional[int] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    optuna Bayesian最適化で SSIM 最大化パラメータを探索する。

    TPE Sampler を使用してパラメータ空間を効率的に探索し、
    入力画像に対して最も高い SSIM を達成するパラメータセットを返す。

    Args:
        image    : 入力画像（最適化の基準となる画像）。
        n_trials : 試行回数（デフォルト 100, テスト用途では 10 程度）。
        category : プリセットカテゴリ（"portrait"/"landscape"/"logo"/"anime"）。
        seed     : 再現性のための乱数シード（None=ランダム）。
        timeout  : タイムアウト秒数（None=制限なし）。

    Returns:
        dict with keys:
          "category"    : str
          "n_trials"    : int（実行した試行数）
          "best_value"  : float（最高 SSIM スコア）
          "best_params" : dict（最適パラメータ）
            - "grid_size"     : int
            - "thickness"     : float
            - "extra_removal" : float
            - "edge_weight"   : float
            - "dark_threshold": float
            - "tonal_levels"  : int
    """
    import optuna
    from .dm4 import DM4Config, generate_dm4_maze

    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category '{category}'. Must be one of: {sorted(VALID_CATEGORIES)}"
        )

    constraints = CATEGORY_CONSTRAINTS[category]

    def _build_config(trial: "optuna.Trial") -> DM4Config:
        gs_min, gs_max = constraints["grid_size"]
        tk_min, tk_max = constraints["thickness"]
        er_min, er_max = constraints["extra_removal"]
        ew_min, ew_max = constraints["edge_weight"]
        dt_min, dt_max = constraints["dark_threshold"]
        tl_min, tl_max = constraints["tonal_levels"]

        grid_size     = trial.suggest_int("grid_size", gs_min, gs_max)
        thickness     = trial.suggest_float("thickness", tk_min, tk_max)
        _extra_removal = trial.suggest_float("extra_removal", er_min, er_max)  # stored only
        edge_weight   = trial.suggest_float("edge_weight", ew_min, ew_max)
        dark_threshold = trial.suggest_float("dark_threshold", dt_min, dt_max)
        tonal_levels  = trial.suggest_int("tonal_levels", tl_min, tl_max)

        return DM4Config(
            grid_rows=grid_size,
            grid_cols=grid_size,
            tonal_thickness_range=float(thickness),
            edge_weight=float(edge_weight),
            tonal_grades=_levels_to_grades(tonal_levels),
        )

    def objective(trial: "optuna.Trial") -> float:
        try:
            config = _build_config(trial)
            result = generate_dm4_maze(image, config)
            return float(result.ssim_score)
        except Exception as exc:
            logger.debug("Trial %d failed: %s", trial.number, exc)
            return 0.0

    # optuna ログを抑制（通常利用時のノイズ削減）
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_params = dict(study.best_params)

    return {
        "category":    category,
        "n_trials":    len(study.trials),
        "best_value":  float(study.best_value),
        "best_params": best_params,
    }


def generate_preset(
    category: str,
    image: Image.Image,
    n_trials: int = 5,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    カテゴリプリセットを生成する。

    最低 n_trials 試行でパラメータを探索し、
    JSON シリアライズ可能な dict を返す。
    必ず 4 キー（category/best_params/best_value/n_trials）を含む。

    Args:
        category : "portrait"/"landscape"/"logo"/"anime"
        image    : 代表画像（プリセットの最適化基準）
        n_trials : 最適化試行回数（デフォルト 5）
        seed     : 再現性シード

    Returns:
        {"category": str, "best_params": dict, "best_value": float, "n_trials": int}
    """
    result = optimize_for_image(image, n_trials=n_trials, category=category, seed=seed)
    # 保証: 必ず4キーを含む
    return {
        "category":    result["category"],
        "best_params": result["best_params"],
        "best_value":  result["best_value"],
        "n_trials":    result["n_trials"],
    }


def load_preset(preset_path: str) -> Dict[str, Any]:
    """JSON ファイルからプリセットを読み込む。"""
    import json
    with open(preset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_preset(preset: Dict[str, Any], output_path: str) -> None:
    """プリセット dict を JSON ファイルに保存する。"""
    import json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(preset, f, indent=2, ensure_ascii=False)
