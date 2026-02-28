from __future__ import annotations

import base64
import io
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from ..core.models import MazeOptions
from ..core.maze_generator import generate_unicursal_maze
from ..core.staged_generator import generate_staged_maze

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
        "path_weight_debug_base64": debug_png_base64,
    }
