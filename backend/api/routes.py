from __future__ import annotations

import base64
import io
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from ..core.models import MazeOptions
from ..core.maze_generator import generate_unicursal_maze
from ..core.staged_generator import generate_staged_maze
from ..core.density import generate_density_maze as generate_density_maze_core, MASTERPIECE_PRESET
from ..core.density import generate_dm5_maze, DM5Config

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@router.post("/generate_maze")
async def generate_maze(
    file: UploadFile = File(...),
    width: int | None = Form(None),
    height: int | None = Form(None),
    stroke_width: float | None = Form(None),
    line_mode: str | None = Form(None),
    face_band_top: float | None = Form(None),
    face_band_bottom: float | None = Form(None),
    face_band_left: float | None = Form(None),
    face_band_right: float | None = Form(None),
    use_overlay: bool | None = Form(True),
    use_face_canny_detail: bool | None = Form(True),
    stage: str | None = Form(None),
    debug_path_scoring: bool | None = Form(False),
    min_edge_size: int | None = Form(None),
    spur_length: int | None = Form(None),
    maze_weight: float | None = Form(None),   # T-10: 顔らしさ↔迷路性トレードオフ
):
    raw_bytes = await file.read()

    if len(raw_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 10MB.",
        )

    try:
        image_stream = io.BytesIO(raw_bytes)
        image = Image.open(image_stream)
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Unsupported or invalid image file.")

    # line_mode は "default" / "detail" のいずれかを期待するが、
    # 不正値の場合は None として扱い、バックエンド側のデフォルトに任せる。
    normalized_mode: str | None
    if line_mode in ("default", "detail"):
        normalized_mode = line_mode
    else:
        normalized_mode = None

    def _clamp_ratio(value: float | None) -> float | None:
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, v))

    band_top = _clamp_ratio(face_band_top)
    band_bottom = _clamp_ratio(face_band_bottom)
    band_left = _clamp_ratio(face_band_left)
    band_right = _clamp_ratio(face_band_right)

    use_overlay_bool = bool(use_overlay) if use_overlay is not None else True
    use_face_canny_detail_bool = (
        bool(use_face_canny_detail) if use_face_canny_detail is not None else True
    )
    debug_path_scoring_bool = bool(debug_path_scoring) if debug_path_scoring is not None else False

    options = MazeOptions(
        width=width,
        height=height,
        stroke_width=stroke_width,
        line_mode=normalized_mode,  # type: ignore[arg-type]
        face_band_top=band_top,
        face_band_bottom=band_bottom,
        face_band_left=band_left,
        face_band_right=band_right,
        use_overlay=use_overlay_bool,
        use_face_canny_detail=use_face_canny_detail_bool,
        stage=stage,
        debug_path_scoring=debug_path_scoring_bool,
        min_edge_size=min_edge_size,
        spur_length=spur_length,
        maze_weight=maze_weight,   # T-10
    )

    if stage in {"line", "unicursal", "maze"}:
        result = generate_staged_maze(image=image, options=options, stage=stage)
    else:
        result = generate_unicursal_maze(image=image, options=options)

    logger.info(
        "maze generated: maze_id=%s image=%dx%d num_solutions=%s difficulty_score=%s",
        result.maze_id,
        image.width,
        image.height,
        result.num_solutions,
        result.difficulty_score,
    )

    png_base64 = base64.b64encode(result.png_bytes).decode("utf-8")
    debug_png_base64 = None
    if result.path_weight_debug_png is not None:
        debug_png_base64 = base64.b64encode(result.path_weight_debug_png).decode("utf-8")

    return {
        "maze_id": result.maze_id,
        "svg": result.svg,
        "png_base64": png_base64,
        "timings": result.timings or {},
        "num_solutions": result.num_solutions,
        "difficulty_score": result.difficulty_score,
        "turn_count": result.turn_count,          # T-9
        "path_length": result.path_length,        # T-9
        "dead_end_count": result.dead_end_count,  # T-9
        "path_weight_debug_base64": debug_png_base64,
    }


@router.post("/generate_density_maze")
async def generate_density_maze(
    file: UploadFile = File(...),
    grid_size: int | None = Form(50),
    width: int | None = Form(800),
    height: int | None = Form(600),
    stroke_width: float | None = Form(2.0),
    show_solution: bool | None = Form(True),
    density_factor: float | None = Form(1.0),
    max_side: int | None = Form(512),
    # Phase 2: A3 エッジ強調 (Canny)
    edge_weight: float | None = Form(0.0),
    edge_sigma: float | None = Form(1.0),
    edge_low_threshold: float | None = Form(0.05),
    edge_high_threshold: float | None = Form(0.20),
    # Phase 2: A2 CLAHEコントラスト
    contrast_boost: float | None = Form(1.0),
    # Phase 2: テクスチャ
    use_texture: bool | None = Form(False),
    use_heuristic: bool | None = Form(False),
    bias_strength: float | None = Form(0.5),
    preset: str | None = Form("generic"),
    n_segments: int | None = Form(4),
    # Phase 2b: 密度制御（ループ許容）
    extra_removal_rate: float | None = Form(0.0),
    dark_threshold: float | None = Form(0.3),
    light_threshold: float | None = Form(0.7),
    # Masterpiece 3本柱
    thickness_range: float | None = Form(1.5),
    use_image_guided: bool | None = Form(False),
    solution_highlight: bool | None = Form(False),
    # --masterpiece: MASTERPIECE_PRESET を一括適用（True のとき個別パラメータを上書き）
    masterpiece: bool | None = Form(False),
):
    """密度迷路 Phase 1/2: 濃度マップ→グリッド→Kruskal→解経路。Phase2パラメータ対応。"""
    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
    try:
        image_stream = io.BytesIO(raw_bytes)
        image = Image.open(image_stream)
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Unsupported or invalid image file.")

    gs = max(2, min(200, grid_size)) if grid_size is not None else 50
    w = width or 800
    h = height or 600
    sw = float(stroke_width) if stroke_width is not None else 2.0
    df = float(density_factor) if density_factor is not None else 1.0
    ms = max_side or 512
    ew = max(0.0, min(2.0, float(edge_weight))) if edge_weight is not None else 0.0
    e_sigma = max(0.3, min(5.0, float(edge_sigma))) if edge_sigma is not None else 1.0
    e_low = max(0.01, min(0.5, float(edge_low_threshold))) if edge_low_threshold is not None else 0.05
    e_high = max(0.05, min(0.8, float(edge_high_threshold))) if edge_high_threshold is not None else 0.20
    cb = max(0.0, min(3.0, float(contrast_boost))) if contrast_boost is not None else 1.0
    bs = max(0.0, min(1.0, float(bias_strength))) if bias_strength is not None else 0.5
    ns = max(1, min(8, int(n_segments))) if n_segments is not None else 4
    err = max(0.0, min(1.0, float(extra_removal_rate))) if extra_removal_rate is not None else 0.0
    dt = max(0.0, min(1.0, float(dark_threshold))) if dark_threshold is not None else 0.3
    lt = max(0.0, min(1.0, float(light_threshold))) if light_threshold is not None else 0.7
    tr = max(0.0, min(3.0, float(thickness_range))) if thickness_range is not None else 1.5
    ig = bool(use_image_guided) if use_image_guided is not None else False
    sh = bool(solution_highlight) if solution_highlight is not None else False

    # masterpiece=True のとき MASTERPIECE_PRESET で一括上書き
    if masterpiece:
        p = MASTERPIECE_PRESET
        gs = p["grid_size"]
        sw = p["stroke_width"]
        tr = p["thickness_range"]
        ew = p["edge_weight"]
        err = p["extra_removal_rate"]
        dt = p["dark_threshold"]
        lt = p["light_threshold"]
        ig = p["use_image_guided"]
        sh = p["solution_highlight"]
        show_solution = p["show_solution"]

    result = generate_density_maze_core(
        image,
        grid_size=gs,
        max_side=ms,
        width=w,
        height=h,
        stroke_width=sw,
        show_solution=bool(show_solution) if show_solution is not None else True,
        density_factor=df,
        edge_weight=ew,
        edge_sigma=e_sigma,
        edge_low_threshold=e_low,
        edge_high_threshold=e_high,
        contrast_boost=cb,
        use_texture=bool(use_texture) if use_texture is not None else False,
        use_heuristic=bool(use_heuristic) if use_heuristic is not None else False,
        bias_strength=bs,
        preset=preset or "generic",
        n_segments=ns,
        extra_removal_rate=err,
        dark_threshold=dt,
        light_threshold=lt,
        thickness_range=tr,
        use_image_guided=ig,
        solution_highlight=sh,
    )

    r, c = result.grid_rows, result.grid_cols
    entrance_rc = (result.entrance // c, result.entrance % c)
    exit_rc = (result.exit_cell // c, result.exit_cell % c)

    return {
        "maze_id": result.maze_id,
        "maze_svg": result.svg,
        "maze_png_base64": base64.b64encode(result.png_bytes).decode("utf-8"),
        "entrance": {"cell_id": result.entrance, "row": entrance_rc[0], "col": entrance_rc[1]},
        "exit": {"cell_id": result.exit_cell, "row": exit_rc[0], "col": exit_rc[1]},
        "solution_path": result.solution_path,
        "grid_rows": r,
        "grid_cols": c,
    }


@router.post("/dm5/generate")
async def generate_dm5(
    file: UploadFile = File(...),
    print_format: str | None = Form("A4"),
    viewing_distance: str | None = Form("desk"),
    output_format: str | None = Form("png"),
    dpi: int | None = Form(300),
):
    """DM-5 印刷最適化迷路: A4/A3 @ 300DPI, viewing_distance 別壁厚, CMYK/PDF 対応。"""
    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Unsupported or invalid image file.")

    valid_formats = {"A4", "A3"}
    valid_distances = {"desk", "poster", "large"}
    valid_outputs = {"png", "cmyk_png", "pdf"}

    pf = print_format if print_format in valid_formats else "A4"
    vd = viewing_distance if viewing_distance in valid_distances else "desk"
    of = output_format if output_format in valid_outputs else "png"
    d = max(72, min(600, int(dpi))) if dpi is not None else 300

    config = DM5Config(
        print_format=pf,
        viewing_distance=vd,
        output_format=of,
        dpi=d,
    )

    result = generate_dm5_maze(image, config)

    response: dict = {
        "print_format": result.print_format,
        "viewing_distance": result.viewing_distance,
        "dpi": result.dpi,
        "wall_thickness_px": result.wall_thickness_px,
        "grid_rows": result.grid_rows,
        "grid_cols": result.grid_cols,
        "ssim_score": result.ssim_score,
        "solution_path_length": len(result.solution_path),
        "maze_svg": result.svg,
    }

    if of == "pdf" and result.pdf_bytes is not None:
        response["pdf_base64"] = base64.b64encode(result.pdf_bytes).decode("utf-8")
    elif of == "cmyk_png" and result.cmyk_png_bytes is not None:
        response["cmyk_png_base64"] = base64.b64encode(result.cmyk_png_bytes).decode("utf-8")
    else:
        response["png_base64"] = base64.b64encode(result.png_bytes).decode("utf-8")

    return response
