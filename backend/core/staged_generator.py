from __future__ import annotations

import time
import uuid
import logging
from typing import Dict, List

import numpy as np
from PIL import Image

from .decorator import generate_dummy_branches
from .features import FeatureSummary, extract_features_from_edges
from .backbone import compute_backbone_endpoints
from .euler_path import build_euler_path
from .line_drawing import extract_line_drawing
logger = logging.getLogger(__name__)
from .maze_generator import (
    _edges_to_png,
    _edges_to_svg_path,
    _path_points_to_png,
    _path_points_to_svg_path,
    _paths_to_png,
    _resolve_size,
    _render_path_weight_debug_png,
    enforce_uniqueness,
)
from .models import MazeOptions, MazeResult
from .path_finder import PathPoint
from .skeleton import edges_to_skeleton, remove_short_spurs
from .solver import count_solutions_on_graph, pick_start_goal_from_path, solve_path


def generate_staged_maze(image: Image.Image, options: MazeOptions, stage: str) -> MazeResult:
    """
    パイプラインを段階ごとに可視化するための簡易ジェネレータ。

    stage:
      - "line"      : 線画＋特徴抽出まで
      - "unicursal" : 一筆パスまで
      - "maze"      : 一筆パスのみ描画（ダミー枝なし）＋ solver 情報

    "dummy" 相当の最終形は既存の generate_unicursal_maze に任せる。
    """
    maze_id = f"{uuid.uuid4()}_{stage}"

    width, height = _resolve_size(options)
    debug_path_scoring = bool(getattr(options, "debug_path_scoring", False))
    debug_path_png: bytes | None = None
    line_mode = getattr(options, "line_mode", None) or "default"

    timings: Dict[str, float] = {}
    t_start = time.perf_counter()

    # フェーズ1: 線画抽出＋顔セマンティクス
    t_line_start = time.perf_counter()
    edges = extract_line_drawing(
        image,
        max_side=512,
        mode=line_mode,
        face_band_top=getattr(options, "face_band_top", None),
        face_band_bottom=getattr(options, "face_band_bottom", None),
        face_band_left=getattr(options, "face_band_left", None),
        face_band_right=getattr(options, "face_band_right", None),
        face_canny_face_low=getattr(options, "face_canny_face_low", None),
        face_canny_face_high=getattr(options, "face_canny_face_high", None),
        face_canny_bg_low=getattr(options, "face_canny_bg_low", None),
        face_canny_bg_high=getattr(options, "face_canny_bg_high", None),
        face_gamma=getattr(options, "face_gamma", None),
        face_smooth_sigma=getattr(options, "face_smooth_sigma", None),
    ).astype(bool)

    face_mask = None
    landmark_edges = None
    try:
        from .semantics import extract_face_mask, extract_face_landmark_edges  # type: ignore[import]

        h_e, w_e = edges.shape
        face_mask = extract_face_mask(image, target_size=(w_e, h_e))
        landmark_edges = extract_face_landmark_edges(image, target_size=(w_e, h_e))
        if landmark_edges is not None:
            edges = np.logical_or(edges, landmark_edges)
    except Exception as e:
        logger.debug("face semantics failed: %s", e)
        face_mask = None
        landmark_edges = None

    feature_summary: FeatureSummary = extract_features_from_edges(
        edges,
        face_mask=face_mask,
        landmark_mask=landmark_edges,
    )
    # T-12: face_band を FeatureSummary に渡す（detail モードまたは face_band 指定時）
    h_e, w_e = edges.shape
    if line_mode == "detail" or any(
        getattr(options, k, None) is not None
        for k in ("face_band_top", "face_band_bottom", "face_band_left", "face_band_right")
    ):
        from .line_drawing import _build_face_band_mask

        top_r = 0.2 if getattr(options, "face_band_top", None) is None else float(options.face_band_top)
        bottom_r = 0.8 if getattr(options, "face_band_bottom", None) is None else float(options.face_band_bottom)
        left_r = 0.0 if getattr(options, "face_band_left", None) is None else float(options.face_band_left)
        right_r = 1.0 if getattr(options, "face_band_right", None) is None else float(options.face_band_right)
        top_r, bottom_r = max(0.0, min(1.0, top_r)), max(0.0, min(1.0, bottom_r))
        left_r, right_r = max(0.0, min(1.0, left_r)), max(0.0, min(1.0, right_r))
        if bottom_r <= top_r:
            bottom_r = min(1.0, top_r + 0.1)
        if right_r <= left_r:
            right_r = min(1.0, left_r + 0.1)
        feature_summary.face_band_mask = _build_face_band_mask(h_e, w_e, top_r, bottom_r, left_r, right_r)

    t_line_end = time.perf_counter()
    timings["line_and_features_ms"] = (t_line_end - t_line_start) * 1000.0

    # stage="line" の場合はここで終了
    if stage == "line":
        path_d = _edges_to_svg_path(edges, width=width, height=height, margin=20)
        png_bytes = _edges_to_png(edges, width=width, height=height, margin=20)
        timings["render_ms"] = (time.perf_counter() - t_line_end) * 1000.0
        timings["total_ms"] = (time.perf_counter() - t_start) * 1000.0
        svg = _wrap_svg(path_d, width, height, stroke_width=options.stroke_width or 6.0)
        return MazeResult(
            maze_id=maze_id,
            svg=svg,
            png_bytes=png_bytes,
            timings=timings,
            path_weight_debug_png=debug_path_png,
        )

    # フェーズ2: スケルトン→グラフ→一筆パス
    t_path_start = time.perf_counter()
    _min_edge_size = getattr(options, "min_edge_size", None) or 8
    skeleton = edges_to_skeleton(edges, min_edge_size=_min_edge_size)
    # spur_length (T-7) が指定されていれば優先、なければ従来の face_in_spur_length/デフォルト
    _spur_length = getattr(options, "spur_length", None)
    spur_in = _spur_length if _spur_length is not None else (getattr(options, "face_in_spur_length", None) or 4)
    spur_out = getattr(options, "face_out_spur_length", None) or 7
    comp_out = getattr(options, "face_out_min_component_size", None) or 14
    if face_mask is not None and isinstance(face_mask, np.ndarray) and face_mask.shape == skeleton.shape:
        try:
            from skimage import morphology  # type: ignore[import]

            inside = np.logical_and(skeleton, face_mask)
            outside = np.logical_and(skeleton, np.logical_not(face_mask))
            outside = morphology.remove_small_objects(outside, max_size=max(0, comp_out - 1), connectivity=2)
            inside = remove_short_spurs(inside, max_length=spur_in)
            outside = remove_short_spurs(outside, max_length=spur_out)
            skeleton = np.logical_or(inside, outside)
        except Exception as e:
            logger.debug("face-aware pruning failed: %s", e)
            skeleton = remove_short_spurs(skeleton, max_length=spur_in)
    else:
        skeleton = remove_short_spurs(skeleton, max_length=spur_in)
    start_candidates_override = None
    if feature_summary.silhouette_mask is not None or feature_summary.landmark_mask is not None:
        boundary_mask = feature_summary.silhouette_mask
        if boundary_mask is not None:
            try:
                from skimage import segmentation  # type: ignore[import]

                boundary_mask = segmentation.find_boundaries(boundary_mask, mode="inner")
            except Exception:
                boundary_mask = feature_summary.silhouette_mask
        a, b = compute_backbone_endpoints(boundary_mask, feature_summary.landmark_mask)
        if a is not None and b is not None:
            start_candidates_override = [a, b]

    euler_result = build_euler_path(
        skeleton,
        features=feature_summary,
        start_candidates_override=start_candidates_override,
        debug=debug_path_scoring,
    )
    graph = euler_result.graph
    path_points = euler_result.path_points
    dummy_branches: List[List[PathPoint]] | None = None
    if len(path_points) >= 2:
        dummy_branches = generate_dummy_branches(graph, path_points)
    t_path_end = time.perf_counter()
    timings["graph_and_path_ms"] = (t_path_end - t_path_start) * 1000.0

    # 一筆パスが取れなかった場合は線画にフォールバック
    if len(path_points) < 2:
        path_d = _edges_to_svg_path(skeleton, width=width, height=height, margin=20)
        png_bytes = _edges_to_png(skeleton, width=width, height=height, margin=20)
        timings["render_ms"] = (time.perf_counter() - t_path_end) * 1000.0
        timings["total_ms"] = (time.perf_counter() - t_start) * 1000.0
        svg = _wrap_svg(path_d, width, height, stroke_width=options.stroke_width or 6.0)
        return MazeResult(
            maze_id=maze_id,
            svg=svg,
            png_bytes=png_bytes,
            timings=timings,
        )

    # フェーズ3: 一筆路 (main path のみ描画)
    t_render_start = time.perf_counter()
    h, w = skeleton.shape

    num_solutions: int | None = None
    difficulty_score: float | None = None

    if stage == "maze":
        removed_edges: list[tuple[int, int]] = []
        # graph ????????????????
        start_id, goal_id = pick_start_goal_from_path(graph, path_points)
        num_solutions_hint = None
        if start_id is not None and goal_id is not None:
            num_solutions_hint = count_solutions_on_graph(
                graph,
                start_id,
                goal_id,
                max_solutions=2,
            )
        if num_solutions_hint is not None and num_solutions_hint >= 2:
            removed_edges = enforce_uniqueness(graph, path_points)
            if start_id is not None and goal_id is not None:
                num_solutions_hint = count_solutions_on_graph(
                    graph,
                    start_id,
                    goal_id,
                    max_solutions=2,
                )
            if removed_edges:
                logger.debug("uniqueness pass removed=%s new_num_solutions=%s", removed_edges, num_solutions_hint)
        solve_result = solve_path(path_points, num_solutions_hint=num_solutions_hint)
        num_solutions = solve_result.num_solutions
        difficulty_score = solve_result.difficulty_score
    if stage == "unicursal":
        # 背景なしで一本線だけ描画
        path_d = _path_points_to_svg_path(
            path_points,
            width=width,
            height=height,
            margin=20,
            src_width=w,
            src_height=h,
        )
        png_bytes = _path_points_to_png(
            path_points,
            width=width,
            height=height,
            margin=20,
            src_width=w,
            src_height=h,
        )
        if debug_path_scoring:
            debug_path_png = _render_path_weight_debug_png(path_points, graph, width, height, margin=20, src_width=w, src_height=h) or debug_path_png
    else:
        # "maze": main path のみ、既存の複数パス描画ヘルパーを再利用（branches は None）
        path_d = _paths_to_svg_path(
            path_points,
            None,
            width=width,
            height=height,
            margin=20,
            src_width=w,
            src_height=h,
        )
        png_bytes = _paths_to_png(
            path_points,
            None,
            width=width,
            height=height,
            margin=20,
            src_width=w,
            src_height=h,
            background_edges=None,
            background_edges_internal=None,
            background_gray=None,
            landmark_edges=None,
        )
        if debug_path_scoring:
            debug_path_png = _render_path_weight_debug_png(path_points, graph, width, height, margin=20, src_width=w, src_height=h) or debug_path_png

    t_render_end = time.perf_counter()
    timings["render_ms"] = (t_render_end - t_render_start) * 1000.0
    timings["total_ms"] = (t_render_end - t_start) * 1000.0

    svg = _wrap_svg(path_d, width, height, stroke_width=options.stroke_width or 6.0)
    return MazeResult(
        maze_id=maze_id,
        svg=svg,
        png_bytes=png_bytes,
        timings=timings,
        num_solutions=num_solutions,
        difficulty_score=difficulty_score,
        path_weight_debug_png=debug_path_png,
    )


def _wrap_svg(path_d: str, width: int, height: int, stroke_width: float) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="5" y="5" width="{width - 10}" height="{height - 10}"
        fill="white" stroke="black" stroke-width="{stroke_width}"/>
  <path d="{path_d}"
        fill="none" stroke="black" stroke-linecap="round"
        stroke-linejoin="round" stroke-width="{stroke_width}"/>
</svg>
"""

