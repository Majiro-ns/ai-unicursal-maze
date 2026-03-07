from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional


@dataclass
class MazeOptions:
    width: Optional[int] = None
    height: Optional[int] = None
    stroke_width: Optional[float] = None
    line_mode: Literal["default", "detail"] | None = None
    face_band_top: Optional[float] = None
    face_band_bottom: Optional[float] = None
    face_band_left: Optional[float] = None
    face_band_right: Optional[float] = None
    face_out_min_component_size: Optional[int] = None
    face_out_spur_length: Optional[int] = None
    face_in_spur_length: Optional[int] = None
    face_canny_face_low: Optional[float] = None
    face_canny_face_high: Optional[float] = None
    face_canny_bg_low: Optional[float] = None
    face_canny_bg_high: Optional[float] = None
    face_gamma: Optional[float] = None
    face_smooth_sigma: Optional[float] = None
    use_overlay: Optional[bool] = None
    use_face_canny_detail: Optional[bool] = None
    stage: Optional[str] = None  # パイプラインのどの段階まで実行するか
    debug_path_scoring: Optional[bool] = None
    # T-7: 迷路の粗さオプション
    min_edge_size: Optional[int] = None  # スケルトン前ノイズ除去閾値（小さいほどノイズ保持、大きいほど除去）
    spur_length: Optional[int] = None    # スパー最大長（小さいほど細かい突起保持、大きいほど除去＝迷路が粗い）
    # T-10: 顔らしさ↔迷路性トレードオフ
    maze_weight: Optional[float] = None  # 0.0=顔らしさ優先, 1.0=迷路性優先（デフォルト0.0）


@dataclass
class MazeResult:
    """
    迷路生成結果。T-9 難易度指標を格納（定義は docs/DIFFICULTY_METRICS.md 参照）。
    """
    maze_id: str
    svg: str
    png_bytes: bytes
    timings: Dict[str, float] | None = None
    num_solutions: Optional[int] = None
    difficulty_score: Optional[float] = None   # T-9: 0.0〜1.0、曲がり角の密度
    turn_count: Optional[int] = None          # T-9: 曲がり角の数
    path_length: Optional[int] = None         # T-9: 経路長（解経路のノード数）
    dead_end_count: Optional[int] = None       # T-9: 袋小路の数（degree==1 のノード数）
    path_weight_debug_png: Optional[bytes] = None
