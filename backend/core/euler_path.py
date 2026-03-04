from __future__ import annotations

"""
Euler path entry point.
スケルトン → グラフ → 一筆書き路（オイラー路近似）の一括パイプライン。

T-6: staged_generator.py の inline graph+path flow をここに集約し、
      呼び出し元が1関数で完結できるよう設計した。
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .features import FeatureSummary
from .graph_builder import MazeGraph, apply_feature_weights, skeleton_to_graph
from .path_finder import PathPoint, find_unicursal_like_path


@dataclass
class EulerPathResult:
    """
    build_euler_path の戻り値。
    - path_points : 一筆書き路の点列（PathPoint のリスト）
    - graph       : 内部グラフ（呼び出し元が solver / decorator / debug で利用）
    """

    path_points: List[PathPoint] = field(default_factory=list)
    graph: MazeGraph = field(default_factory=lambda: MazeGraph(nodes={}, edges=[]))


def build_euler_path(
    skeleton: np.ndarray,
    *,
    features: FeatureSummary | None = None,
    start_candidates_override: List[int] | None = None,
    min_component_size: int = 1,
    max_steps: int = 10_000,
    debug: bool = False,
) -> EulerPathResult:
    """
    スケルトン画像から一筆書き路（オイラー路近似）を生成するエントリポイント。

    パイプライン:
        skeleton
        → skeleton_to_graph(min_component_size)   # ピクセル → グラフ変換
        → apply_feature_weights(features)          # 重み付け（featuresがあるとき）
        → find_unicursal_like_path(...)            # ビームサーチで最良路を探索
        → EulerPathResult(path_points, graph)

    Args:
        skeleton:                  shape=(H, W) の 2 次元 bool 配列
        features:                  特徴量（シルエット・ランドマーク等）。None なら重み付けスキップ
        start_candidates_override: 探索開始ノードIDを外から指定する場合（backbone端点等）
        min_component_size:        グラフから除去する小コンポーネントの最小サイズ
        max_steps:                 ビームサーチの最大ステップ数
        debug:                     True のときスコアを logger.debug で出力

    Returns:
        EulerPathResult:
            .path_points — 一筆書き路の点列（空リストの場合は路探索失敗）
            .graph       — MazeGraph（solver / decorator / debug 用）
    """
    graph = skeleton_to_graph(skeleton, min_component_size=min_component_size)
    if features is not None:
        apply_feature_weights(graph, features)
    path_points = find_unicursal_like_path(
        graph,
        features=features,
        max_steps=max_steps,
        debug=debug,
        start_candidates_override=start_candidates_override,
    )
    return EulerPathResult(path_points=path_points, graph=graph)
