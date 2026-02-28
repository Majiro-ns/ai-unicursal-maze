from __future__ import annotations

import io
import logging
import time
import uuid
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import canny

from .decorator import generate_dummy_branches
from .features import FeatureSummary, extract_features_from_edges
from .graph_builder import MazeGraph, apply_feature_weights, skeleton_to_graph
from .backbone import compute_backbone_endpoints
from .line_drawing import extract_line_drawing
from .models import MazeOptions, MazeResult
from .path_finder import PathPoint, find_unicursal_like_path
from .skeleton import edges_to_skeleton, remove_short_spurs
from .solver import (
    SolveResult,
    count_solutions_on_graph,
    pick_start_goal_from_path,
    solve_path,
)

logger = logging.getLogger(__name__)


def _extract_edges(image: Image.Image, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    入力画像からエッジマップを抽出して返す。
    戻り値は shape = (H, W) の bool 配列で、True がエッジピクセル。
    """
    arr = np.asarray(image)

    if arr.ndim == 2:
        gray = arr.astype(float) / 255.0
    else:
        gray = rgb2gray(arr)

    if gray.shape != target_size:
        gray_resized = transform.resize(
            gray,
            target_size,
            anti_aliasing=True,
        )
    else:
        gray_resized = gray

    edges = canny(gray_resized, sigma=2.0)
    return edges


def _resolve_size(options: MazeOptions) -> Tuple[int, int]:
    width = options.width or 800
    height = options.height or 600
    width = max(100, width)
    height = max(100, height)
    return width, height


def _edges_to_svg_path(edges: np.ndarray, width: int, height: int, margin: int = 20) -> str:
    """
    エッジマップからシルエット近似の折れ線 SVG path を生成する。
    """
    if edges.ndim != 2:
        raise ValueError("edges は 2 次元配列である必要があります")

    h, w = edges.shape
    if h == 0 or w == 0:
        return ""

    inner_width = max(1, width - 2 * margin)
    inner_height = max(1, height - 2 * margin)
    scale_x = inner_width / float(w)
    scale_y = inner_height / float(h)

    num_slices = int(min(100, max(1, w)))
    if num_slices <= 1:
        ys_all, xs_all = np.where(edges)
        if ys_all.size == 0:
            return ""
        x_mean = xs_all.mean()
        y_mean = ys_all.mean()
        x_pos = margin + (x_mean + 0.5) * scale_x
        y_pos = margin + (y_mean + 0.5) * scale_y
        return f"M {x_pos:.2f} {y_pos:.2f}"

    points: List[Tuple[float, float]] = []

    for i in range(num_slices):
        x_start = int(round(i * w / num_slices))
        x_end = int(round((i + 1) * w / num_slices)) - 1
        if x_start > x_end:
            continue

        slice_edges = edges[:, x_start : x_end + 1]
        ys, xs_local = np.where(slice_edges)
        if ys.size == 0:
            continue

        y_mean = float(ys.mean())
        x_center = (x_start + x_end) * 0.5

        x_pos = margin + (x_center + 0.5) * scale_x
        y_pos = margin + (y_mean + 0.5) * scale_y

        points.append((x_pos, y_pos))

    if not points:
        return ""

    commands: List[str] = []
    x0, y0 = points[0]
    commands.append(f"M {x0:.2f} {y0:.2f}")
    for x, y in points[1:]:
        commands.append(f"L {x:.2f} {y:.2f}")

    return " ".join(commands)


def _edge_run_to_path(
    x_start: int,
    x_end: int,
    y: int,
    scale_x: float,
    scale_y: float,
    margin: int,
) -> str:
    x1 = margin + (x_start + 0.5) * scale_x
    x2 = margin + (x_end + 0.5) * scale_x
    y_pos = margin + (y + 0.5) * scale_y
    return f"M {x1:.2f} {y_pos:.2f} L {x2:.2f} {y_pos:.2f}"


def _edges_to_png(edges: np.ndarray, width: int, height: int, margin: int = 20) -> bytes:
    if edges.ndim != 2:
        raise ValueError("edges は 2 次元配列である必要があります")

    h, w = edges.shape
    img = Image.new("RGB", (width, height), "white")

    if h == 0 or w == 0:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    draw = ImageDraw.Draw(img)

    inner_width = max(1, width - 2 * margin)
    inner_height = max(1, height - 2 * margin)
    scale_x = inner_width / float(w)
    scale_y = inner_height / float(h)

    for y in range(h):
        row = edges[y]
        in_run = False
        run_start = 0
        for x in range(w):
            val = bool(row[x])
            if val and not in_run:
                in_run = True
                run_start = x
            elif not val and in_run:
                in_run = False
                run_end = x - 1
                _draw_edge_run(
                    draw,
                    run_start,
                    run_end,
                    y,
                    scale_x,
                    scale_y,
                    margin,
                )
        if in_run:
            run_end = w - 1
            _draw_edge_run(
                draw,
                run_start,
                run_end,
                y,
                scale_x,
                scale_y,
                margin,
            )

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _draw_edge_run(
    draw: ImageDraw.ImageDraw,
    x_start: int,
    x_end: int,
    y: int,
    scale_x: float,
    scale_y: float,
    margin: int,
    color: str = "black",
    width: int = 1,
) -> None:
    x1 = margin + (x_start + 0.5) * scale_x
    x2 = margin + (x_end + 0.5) * scale_x
    y_pos = margin + (y + 0.5) * scale_y
    draw.line((x1, y_pos, x2, y_pos), fill=color, width=width)


def _path_points_to_svg_path(
    points: List[PathPoint],
    width: int,
    height: int,
    margin: int = 20,
    src_width: int | None = None,
    src_height: int | None = None,
) -> str:
    if not points:
        return ""

    if src_width is None or src_height is None:
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        src_width = int(max(xs) + 1)
        src_height = int(max(ys) + 1)

    inner_width = max(1, width - 2 * margin)
    inner_height = max(1, height - 2 * margin)
    scale_x = inner_width / float(src_width)
    scale_y = inner_height / float(src_height)

    commands: List[str] = []
    first = points[0]
    x0 = margin + (first.x + 0.5) * scale_x
    y0 = margin + (first.y + 0.5) * scale_y
    commands.append(f"M {x0:.2f} {y0:.2f}")
    for p in points[1:]:
        x = margin + (p.x + 0.5) * scale_x
        y = margin + (p.y + 0.5) * scale_y
        commands.append(f"L {x:.2f} {y:.2f}")
    return " ".join(commands)


def _path_points_to_png(
    points: List[PathPoint],
    width: int,
    height: int,
    margin: int = 20,
    src_width: int | None = None,
    src_height: int | None = None,
) -> bytes:
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    if not points:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    if src_width is None or src_height is None:
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        src_width = int(max(xs) + 1)
        src_height = int(max(ys) + 1)

    inner_width = max(1, width - 2 * margin)
    inner_height = max(1, height - 2 * margin)
    scale_x = inner_width / float(src_width)
    scale_y = inner_height / float(src_height)

    last: tuple[float, float] | None = None
    for p in points:
        x = margin + (p.x + 0.5) * scale_x
        y = margin + (p.y + 0.5) * scale_y
        if last is not None:
            draw.line((last[0], last[1], x, y), fill="black", width=2)
        last = (x, y)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _paths_to_svg_path(
    main: List[PathPoint],
    branches: List[List[PathPoint]] | None,
    width: int,
    height: int,
    margin: int,
    src_width: int,
    src_height: int,
) -> str:
    parts: List[str] = []
    if main:
        parts.append(
            _path_points_to_svg_path(
                main,
                width=width,
                height=height,
                margin=margin,
                src_width=src_width,
                src_height=src_height,
            )
        )
    if branches:
        for b in branches:
            if not b:
                continue
            parts.append(
                _path_points_to_svg_path(
                    b,
                    width=width,
                    height=height,
                    margin=margin,
                    src_width=src_width,
                    src_height=src_height,
                )
            )
    return " ".join(p for p in parts if p)


def _paths_to_png(
    main: List[PathPoint],
    branches: List[List[PathPoint]] | None,
    width: int,
    height: int,
    margin: int,
    src_width: int,
    src_height: int,
    background_edges: np.ndarray | None = None,
    background_edges_internal: np.ndarray | None = None,
    background_gray: Image.Image | None = None,
    landmark_edges: np.ndarray | None = None,
) -> bytes:
    # 下絵（グレースケール）が指定されていれば、それを薄く背景として敷く。
    if background_gray is not None:
        if background_gray.size != (width, height):
            bg_resized = background_gray.resize((width, height))
        else:
            bg_resized = background_gray
        if bg_resized.mode != "L":
            bg_l = bg_resized.convert("L")
        else:
            bg_l = bg_resized
        white = Image.new("L", (width, height), 255)
        # alpha を小さめにして線が主役になるようにする。
        base_l = Image.blend(white, bg_l, alpha=0.3)
        img = base_l.convert("RGB")
    else:
        img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # 背景として線画マスクを描画
    # - シルエット境界: やや濃いグレー
    # - 内部線: 薄いグレー
    if background_edges is not None and background_edges.ndim == 2:
        h, w = background_edges.shape
        inner_width = max(1, width - 2 * margin)
        inner_height = max(1, height - 2 * margin)
        scale_x = inner_width / float(w)
        scale_y = inner_height / float(h)
        for y in range(h):
            row = background_edges[y]
            in_run = False
            run_start = 0
            for x in range(w):
                val = bool(row[x])
                if val and not in_run:
                    in_run = True
                    run_start = x
                elif not val and in_run:
                    in_run = False
                    run_end = x - 1
                    _draw_edge_run(
                        draw,
                        run_start,
                        run_end,
                        y,
                        scale_x,
                        scale_y,
                        margin,
                        color="#777777",
                        width=1,
                    )
            if in_run:
                run_end = w - 1
                _draw_edge_run(
                    draw,
                    run_start,
                    run_end,
                    y,
                    scale_x,
                    scale_y,
                    margin,
                    color="#777777",
                    width=1,
                )

    if background_edges_internal is not None and background_edges_internal.ndim == 2:
        h, w = background_edges_internal.shape
        inner_width = max(1, width - 2 * margin)
        inner_height = max(1, height - 2 * margin)
        scale_x = inner_width / float(w)
        scale_y = inner_height / float(h)
        for y in range(h):
            row = background_edges_internal[y]
            in_run = False
            run_start = 0
            for x in range(w):
                val = bool(row[x])
                if val and not in_run:
                    in_run = True
                    run_start = x
                elif not val and in_run:
                    in_run = False
                    run_end = x - 1
                    _draw_edge_run(
                        draw,
                        run_start,
                        run_end,
                        y,
                        scale_x,
                        scale_y,
                        margin,
                        color="#BBBBBB",
                        width=1,
                    )
            if in_run:
                run_end = w - 1
                _draw_edge_run(
                    draw,
                    run_start,
                    run_end,
                    y,
                    scale_x,
                    scale_y,
                    margin,
                    color="#BBBBBB",
                    width=1,
                )

    # 顔ランドマーク線（目・口・輪郭）は、背景の上から少し太め・濃い線で描画する。
    if landmark_edges is not None and landmark_edges.ndim == 2:
        h, w = landmark_edges.shape
        inner_width = max(1, width - 2 * margin)
        inner_height = max(1, height - 2 * margin)
        scale_x = inner_width / float(w)
        scale_y = inner_height / float(h)
        for y in range(h):
            row = landmark_edges[y]
            in_run = False
            run_start = 0
            for x in range(w):
                val = bool(row[x])
                if val and not in_run:
                    in_run = True
                    run_start = x
                elif not val and in_run:
                    in_run = False
                    run_end = x - 1
                    _draw_edge_run(
                        draw,
                        run_start,
                        run_end,
                        y,
                        scale_x,
                        scale_y,
                        margin,
                        color="black",
                        width=2,
                    )
            if in_run:
                run_end = w - 1
                _draw_edge_run(
                    draw,
                    run_start,
                    run_end,
                    y,
                    scale_x,
                    scale_y,
                    margin,
                    color="black",
                    width=2,
                )

    def draw_path(points: List[PathPoint]) -> None:
        if not points:
            return
        inner_width = max(1, width - 2 * margin)
        inner_height = max(1, height - 2 * margin)
        scale_x = inner_width / float(src_width)
        scale_y = inner_height / float(src_height)
        last: tuple[float, float] | None = None
        for p in points:
            x = margin + (p.x + 0.5) * scale_x
            y = margin + (p.y + 0.5) * scale_y
            if last is not None:
                draw.line((last[0], last[1], x, y), fill="black", width=2)
            last = (x, y)

    draw_path(main)
    if branches:
        for b in branches:
            draw_path(b)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()




def _weight_to_color(weight: float) -> tuple[int, int, int]:
    w = max(0.0, min(1.0, weight))
    r = int(255 * w)
    b = int(255 * (1.0 - w))
    g = int(80 + (1.0 - abs(0.5 - w) * 2.0) * 100)
    return (r, g, b)


def _render_path_weight_debug_png(
    path_points: List[PathPoint],
    graph: MazeGraph,
    width: int,
    height: int,
    *,
    margin: int = 20,
    src_width: int | None = None,
    src_height: int | None = None,
) -> bytes | None:
    if not path_points or not graph.nodes:
        return None

    if src_width is None or src_height is None:
        xs = [p.x for p in path_points]
        ys = [p.y for p in path_points]
        src_width = int(max(xs) + 1)
        src_height = int(max(ys) + 1)

    inner_width = max(1, width - 2 * margin)
    inner_height = max(1, height - 2 * margin)
    scale_x = inner_width / float(src_width)
    scale_y = inner_height / float(src_height)

    coord_to_weight: Dict[tuple[int, int], float] = {
        (node.x, node.y): (node.weight if node.weight is not None else 0.0)
        for node in graph.nodes.values()
    }

    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    last: tuple[float, float] | None = None
    last_weight: float | None = None
    for p in path_points:
        x = margin + (p.x + 0.5) * scale_x
        y = margin + (p.y + 0.5) * scale_y
        coord = (int(round(p.x)), int(round(p.y)))
        w = coord_to_weight.get(coord, 0.0)
        color = _weight_to_color(w)
        if last is not None and last_weight is not None:
            mid_weight = (w + last_weight) * 0.5
            mid_color = _weight_to_color(mid_weight)
            draw.line((last[0], last[1], x, y), fill=mid_color, width=4)
        last = (x, y)
        last_weight = w

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def _compute_face_likeness_score(
    path_points: List[PathPoint],
    face_mask: np.ndarray | None,
    landmark_edges: np.ndarray | None,
) -> Dict[str, float | int | None]:
    """
    顔らしさスコアを計算する。

    Returns:
        dict with keys:
        - landmark_pass_rate: ランドマーク通過率（0.0〜1.0）
        - landmark_pass_count: ランドマーク通過点数
        - landmark_total: ランドマーク総ピクセル数
        - face_mask_pass_rate: 顔マスク内通過率（0.0〜1.0）
        - face_mask_pass_count: 顔マスク内通過点数
        - path_total: パス総点数
    """
    result: Dict[str, float | int | None] = {
        "landmark_pass_rate": None,
        "landmark_pass_count": 0,
        "landmark_total": 0,
        "face_mask_pass_rate": None,
        "face_mask_pass_count": 0,
        "path_total": len(path_points),
    }

    if not path_points:
        return result

    # パスの座標を整数に変換
    path_coords = [(int(round(p.x)), int(round(p.y))) for p in path_points]

    # ランドマーク通過率の計算
    if landmark_edges is not None and landmark_edges.ndim == 2:
        h, w = landmark_edges.shape
        landmark_total = int(landmark_edges.sum())
        result["landmark_total"] = landmark_total

        if landmark_total > 0:
            landmark_pass_count = 0
            for x, y in path_coords:
                if 0 <= y < h and 0 <= x < w:
                    if landmark_edges[y, x]:
                        landmark_pass_count += 1
            result["landmark_pass_count"] = landmark_pass_count
            result["landmark_pass_rate"] = landmark_pass_count / landmark_total

    # 顔マスク内通過率の計算
    if face_mask is not None and face_mask.ndim == 2:
        h, w = face_mask.shape
        face_mask_pass_count = 0
        for x, y in path_coords:
            if 0 <= y < h and 0 <= x < w:
                if face_mask[y, x]:
                    face_mask_pass_count += 1
        result["face_mask_pass_count"] = face_mask_pass_count
        if len(path_points) > 0:
            result["face_mask_pass_rate"] = face_mask_pass_count / len(path_points)

    return result


def enforce_uniqueness(graph: MazeGraph, path_points: List[PathPoint]) -> List[tuple[int, int]]:
    """
    Greedy uniqueness helper: if an alternative route exists between consecutive path nodes,
    cut one side branch edge to reduce multiple solutions. Returns removed edges.
    """
    if not graph.nodes or len(path_points) < 2:
        return []

    coord_to_id: Dict[tuple[int, int], int] = {
        (node.x, node.y): nid for nid, node in graph.nodes.items()
    }
    path_ids: List[int] = []
    for p in path_points:
        nid = coord_to_id.get((int(round(p.x)), int(round(p.y))))
        if nid is not None:
            path_ids.append(nid)
    if len(path_ids) < 2:
        return []

    adj: Dict[int, set[int]] = {nid: set() for nid in graph.nodes.keys()}
    for e in graph.edges:
        adj[e.from_id].add(e.to_id)
        adj[e.to_id].add(e.from_id)

    def has_alt_path(start: int, goal: int) -> bool:
        stack = [start]
        visited: set[int] = set()
        while stack:
            cur = stack.pop()
            if cur == goal:
                return True
            visited.add(cur)
            for nb in adj.get(cur, set()):
                if nb not in visited:
                    stack.append(nb)
        return False

    removed_keys: set[tuple[int, int]] = set()

    for u, v in zip(path_ids, path_ids[1:]):
        if v not in adj.get(u, set()):
            continue
        # temporarily drop the main edge to see if a detour exists
        adj[u].discard(v)
        adj[v].discard(u)
        alt_exists = has_alt_path(u, v)
        adj[u].add(v)
        adj[v].add(u)
        if not alt_exists:
            continue

        candidates: List[tuple[int, int]] = []
        for nb in list(adj[u]):
            if nb != v:
                candidates.append((u, nb))
        for nb in list(adj[v]):
            if nb != u:
                candidates.append((v, nb))
        if not candidates:
            continue
        rm_u, rm_v = candidates[0]
        adj[rm_u].discard(rm_v)
        adj[rm_v].discard(rm_u)
        removed_keys.add(tuple(sorted((rm_u, rm_v))))

    # rebuild edges and degrees according to adj
    new_edges = []
    for e in graph.edges:
        key = tuple(sorted((e.from_id, e.to_id)))
        if key in removed_keys:
            continue
        if e.to_id in adj.get(e.from_id, set()):
            new_edges.append(e)
    graph.edges = new_edges
    for node in graph.nodes.values():
        node.degree = len(adj[node.id])

    if removed_keys:
        logger.debug("enforce_uniqueness removed_edges=%s", sorted(removed_keys))
    return [tuple(k) for k in removed_keys]

def generate_unicursal_maze(image: Image.Image, options: MazeOptions) -> MazeResult:
    """
    エッジ → スケルトン → グラフ → パス → SVG/PNG というパイプラインで
    一筆迷路を生成する。
    """
    maze_id = str(uuid.uuid4())
    debug_path_scoring = bool(getattr(options, "debug_path_scoring", False))


    width, height = _resolve_size(options)
    stroke_width = options.stroke_width or 6.0

    # フェーズ1: 線画抽出（輪郭線マスク）
    line_mode = getattr(options, "line_mode", None) or "default"
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

    # 顔領域マスク・顔ランドマーク線（あれば）を推定し、
    # 線画・特徴量に反映する。
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
    _min_edge_size = getattr(options, "min_edge_size", None) or 8
    skeleton = edges_to_skeleton(edges, min_edge_size=_min_edge_size)
    _spur_length = getattr(options, "spur_length", None)
    spur_in = _spur_length if _spur_length is not None else (getattr(options, "face_in_spur_length", None) or 4)
    spur_out = getattr(options, "face_out_spur_length", None) or 7
    comp_out = getattr(options, "face_out_min_component_size", None) or 14
    if face_mask is not None and isinstance(face_mask, np.ndarray) and face_mask.shape == skeleton.shape:
        try:
            from skimage import morphology  # type: ignore[import]

            inside = np.logical_and(skeleton, face_mask)
            outside = np.logical_and(skeleton, np.logical_not(face_mask))
            outside = morphology.remove_small_objects(outside, min_size=comp_out, connectivity=2)
            inside = remove_short_spurs(inside, max_length=spur_in)
            outside = remove_short_spurs(outside, max_length=spur_out)
            skeleton = np.logical_or(inside, outside)
        except Exception as e:
            logger.debug("face-aware pruning failed: %s", e)
            skeleton = remove_short_spurs(skeleton, max_length=spur_in)
    else:
        skeleton = remove_short_spurs(skeleton, max_length=spur_in)
    graph = skeleton_to_graph(skeleton)
    apply_feature_weights(graph, feature_summary)
    # 特徴量（シルエット / 顔 / ランドマーク）を反映した weight を使って、
    # 「絵としての良さ」をスコアリングしながら一筆パスを選ぶ。
    debug_path_png: bytes | None = None
    # Try to anchor start/goal using backbone (silhouette boundary + landmarks) diameter.
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

    path_points = find_unicursal_like_path(
        graph,
        features=feature_summary,
        debug=debug_path_scoring,
        start_candidates_override=start_candidates_override,
    )
    dummy_branches: List[List[PathPoint]] | None = None
    if len(path_points) >= 2:
        dummy_branches = generate_dummy_branches(graph, path_points)

    use_fallback = len(path_points) < 2

    if use_fallback:
        path_d = _edges_to_svg_path(skeleton, width=width, height=height, margin=20)
        png_bytes = _edges_to_png(skeleton, width=width, height=height, margin=20)
        solve_result: SolveResult | None = None
    else:
        h, w = skeleton.shape
        path_d = _paths_to_svg_path(
            path_points,
            dummy_branches,
            width=width,
            height=height,
            margin=20,
            src_width=w,
            src_height=h,
        )

        # 背景線は、シルエット境界と内部線を分けて描画する。
        background_edges = edges
        background_edges_internal = None
        try:
            if feature_summary.silhouette_mask is not None or feature_summary.internal_edges is not None:
                from skimage import segmentation  # type: ignore[import]

                background_edges = np.zeros_like(edges, dtype=bool)
                background_edges_internal = None
                sil_for_boundary = feature_summary.refined_silhouette_mask or feature_summary.silhouette_mask
                if sil_for_boundary is not None:
                    boundary = segmentation.find_boundaries(
                        sil_for_boundary,
                        mode="inner",
                    )
                    background_edges |= boundary
                if feature_summary.internal_edges is not None:
                    background_edges_internal = feature_summary.internal_edges.astype(bool)

                    # detail モードでは、内部線を「顔帯域」に絞る
                    if line_mode == "detail":
                        ys_internal, _ = np.where(background_edges_internal)
                        if ys_internal.size > 0:
                            y_min_i = int(ys_internal.min())
                            y_max_i = int(ys_internal.max())
                        elif feature_summary.silhouette_mask is not None:
                            ys_sil, _ = np.where(feature_summary.silhouette_mask)
                            if ys_sil.size > 0:
                                y_min_i = int(ys_sil.min())
                                y_max_i = int(ys_sil.max())
                            else:
                                y_min_i = 0
                                y_max_i = background_edges_internal.shape[0] - 1
                        else:
                            y_min_i = 0
                            y_max_i = background_edges_internal.shape[0] - 1

                        band_height = max(1, y_max_i - y_min_i + 1)
                        band_top = y_min_i + int(band_height * 0.2)
                        band_bottom = y_min_i + int(band_height * 0.7)
                        band_top = max(0, min(band_top, background_edges_internal.shape[0] - 1))
                        band_bottom = max(band_top + 1, min(band_bottom, background_edges_internal.shape[0]))

                        band_mask = np.zeros_like(background_edges_internal, dtype=bool)
                        band_mask[band_top:band_bottom, :] = True
                        background_edges_internal = np.logical_and(background_edges_internal, band_mask)
        except Exception:
            background_edges = edges
            background_edges_internal = None

        # 元画像のグレースケールを下絵として薄く敷く
        gray_bg = None
        if getattr(options, "use_overlay", True):
            try:
                gray_bg = image.convert("L")
            except Exception:
                gray_bg = None

        png_bytes = _paths_to_png(
            path_points,
            dummy_branches,
            width=width,
            height=height,
            margin=20,
            src_width=w,
            src_height=h,
            background_edges=background_edges,
            background_edges_internal=background_edges_internal,
            background_gray=gray_bg,
            landmark_edges=landmark_edges,
        )
        if debug_path_scoring:
            debug_path_png = _render_path_weight_debug_png(path_points, graph, width, height, margin=20, src_width=w, src_height=h) or debug_path_png

        # パスの両端ノードを MazeGraph 上で特定し、簡易な解数を推定する。
        coord_to_id = {(node.x, node.y): nid for nid, node in graph.nodes.items()}
        start_coord = (int(round(path_points[0].x)), int(round(path_points[0].y)))
        goal_coord = (int(round(path_points[-1].x)), int(round(path_points[-1].y)))
        start_id = coord_to_id.get(start_coord)
        goal_id = coord_to_id.get(goal_coord)
        num_solutions_hint = None
        if start_id is not None and goal_id is not None:
            num_solutions_hint = count_solutions_on_graph(
                graph,
                start_id,
                goal_id,
                max_solutions=2,
            )
        solve_result = solve_path(path_points, num_solutions_hint=num_solutions_hint)

    edge_count = int(skeleton.sum())
    segment_count = path_d.count("M ")

    # 顔らしさスコアを計算
    face_likeness = _compute_face_likeness_score(path_points, face_mask, landmark_edges)
    logger.info(
        "Face likeness score: maze_id=%s landmark_pass_rate=%.3f (%d/%d) "
        "face_mask_pass_rate=%.3f (%d/%d path_points)",
        maze_id,
        face_likeness["landmark_pass_rate"] or 0.0,
        face_likeness["landmark_pass_count"] or 0,
        face_likeness["landmark_total"] or 0,
        face_likeness["face_mask_pass_rate"] or 0.0,
        face_likeness["face_mask_pass_count"] or 0,
        face_likeness["path_total"] or 0,
    )

    if solve_result is not None:
        logger.info(
            "Generated unicursal maze: maze_id=%s width=%d height=%d stroke_width=%.2f "
            "edge_pixels=%d segments=%d has_solution=%s num_solutions=%s difficulty_score=%s "
            "centroid=%s bbox_count=%d",
            maze_id,
            width,
            height,
            stroke_width,
            edge_count,
            segment_count,
            solve_result.has_solution,
            solve_result.num_solutions,
            solve_result.difficulty_score,
            feature_summary.centroid,
            len(feature_summary.bounding_boxes),
        )
    else:
        logger.info(
            "Generated unicursal maze (fallback): maze_id=%s width=%d height=%d stroke_width=%.2f "
            "edge_pixels=%d segments=%d centroid=%s bbox_count=%d",
            maze_id,
            width,
            height,
            stroke_width,
            edge_count,
            segment_count,
            feature_summary.centroid,
            len(feature_summary.bounding_boxes),
        )

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="5" y="5" width="{width - 10}" height="{height - 10}"
        fill="white" stroke="black" stroke-width="{stroke_width}"/>
  <path d="{path_d}"
        fill="none" stroke="black" stroke-linecap="round"
        stroke-linejoin="round" stroke-width="{stroke_width}"/>
</svg>
"""

    # T-8: solver 評価結果を MazeResult に格納
    _num_sol = solve_result.num_solutions if solve_result is not None else None
    _diff_score = solve_result.difficulty_score if solve_result is not None else None
    return MazeResult(
        maze_id=maze_id,
        svg=svg,
        png_bytes=png_bytes,
        path_weight_debug_png=debug_path_png,
        num_solutions=_num_sol,
        difficulty_score=_diff_score,
    )
