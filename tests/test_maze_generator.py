from __future__ import annotations

import re

from PIL import Image, ImageDraw

from backend.core.maze_generator import generate_unicursal_maze
from backend.core.models import MazeOptions


def test_generate_unicursal_maze_basic() -> None:
    image = Image.new("RGB", (200, 200), "white")
    options = MazeOptions()

    result = generate_unicursal_maze(image, options)

    assert getattr(result, "maze_id", None)
    assert isinstance(result.svg, str) and result.svg.strip()
    assert "<svg" in result.svg
    assert isinstance(result.png_bytes, (bytes, bytearray))
    assert len(result.png_bytes) > 0


def test_generate_unicursal_maze_polyline_has_enough_points() -> None:
    image = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((20, 40, 180, 160), fill="black")

    options = MazeOptions(width=400, height=300)
    result = generate_unicursal_maze(image, options)

    svg = result.svg
    match = re.search(r'<path[^>]* d="([^"]+)"', svg)
    assert match is not None, "SVG の path d 属性が見つかりません"
    d = match.group(1).strip()

    tokens = d.split()
    xs: list[float] = []
    i = 0
    while i < len(tokens):
        cmd = tokens[i]
        if cmd in ("M", "L") and i + 2 < len(tokens):
            try:
                x = float(tokens[i + 1])
            except ValueError:
                break
            xs.append(x)
            i += 3
        else:
            i += 1

    assert len(xs) >= 8, f"折れ線の代表点が少なすぎます: {len(xs)}"

