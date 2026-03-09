"""
I4 Reverse Pipeline — standalone maze generation interface (DM-1 Part A).

generate_i4_maze() wraps the existing Path-First Masterpiece pipeline
(I4+F3+G1+H2+K1) with a clean, renderer-agnostic interface that returns
I4MazeResult — consumed independently by:
  - A7 (renderer): uses walls, density_map, cell_size_px for variable-width rendering
  - A8 (solver):   uses walls, entrance, exit_pos, solution_path for verification

Pipeline:
  image_path → preprocess (CLAHE) → cell grid → edge detection (K1) →
  classify DARK/MID/BRIGHT → dark blobs → entrance/exit →
  F3 serpentine fill → I4 wall mold (anti-shortcut Kruskal) → I4MazeResult

Usage::

    from backend.core.density.i4_pipeline import generate_i4_maze

    r = generate_i4_maze("data/input/test3.jpg", grid_width=200, grid_height=300)
    print(f"grid={r.grid_width}x{r.grid_height}, path_len={len(r.solution_path)}")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

from .edge_enhancer import detect_edge_map, extract_edge_waypoints
from .grid_builder import build_cell_grid
from .path_designer import (
    build_walls_around_path,
    classify_cells,
    design_masterpiece_path,
    find_dark_blobs,
    find_entrance_exit_path_first,
    order_blobs_for_path,
)
from .preprocess import preprocess_image


# ---------------------------------------------------------------------------
# I4MazeResult — A7 / A8 shared interface
# ---------------------------------------------------------------------------

@dataclass
class I4MazeResult:
    """
    Output of the I4 reverse pipeline.

    Attributes:
        grid_width:    Number of columns in the cell grid.
        grid_height:   Number of rows in the cell grid.
        cell_size_px:  Pixel size per cell (renderer hint for A7).
        walls:         Set of walls as pairs of adjacent cell coordinates.
                       Each wall is ``((r1, c1), (r2, c2))`` where
                       ``r1 * cols + c1 < r2 * cols + c2`` (smaller-ID cell first).
                       Absence of a wall entry means a passage exists.
        solution_path: Non-revisiting path from entrance to exit as a list of
                       ``(row, col)`` tuples. Every consecutive pair is
                       4-neighbor adjacent.
        density_map:   ``(grid_height, grid_width)`` float32 array of cell
                       luminance values [0, 1]. Used by A7 for G1 variable
                       path width control.
        entrance:      ``(row, col)`` of entrance cell (border, darkest).
        exit_pos:      ``(row, col)`` of exit cell (opposite border, darkest).
    """
    grid_width: int
    grid_height: int
    cell_size_px: int
    walls: Set[Tuple[Tuple[int, int], Tuple[int, int]]]
    solution_path: List[Tuple[int, int]]
    density_map: np.ndarray
    entrance: Tuple[int, int]
    exit_pos: Tuple[int, int]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _adj_to_walls(
    adj: Dict[int, List[int]],
    grid_rows: int,
    grid_cols: int,
) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Convert adjacency list (open passages) to walls set (blocked passages).

    For every pair of 4-neighbor adjacent cells that are NOT connected in
    *adj*, a wall tuple ``((r1, c1), (r2, c2))`` is added to the result.
    By construction: ``r1 * grid_cols + c1 < r2 * grid_cols + c2``
    (smaller cell-ID coordinate is always placed first).

    Args:
        adj:        Dict[cell_id, List[cell_id]] adjacency list from
                    build_walls_around_path().
        grid_rows:  Number of grid rows.
        grid_cols:  Number of grid columns.

    Returns:
        Set of wall coordinate tuples.
    """
    # Convert to sets for O(1) membership test
    adj_sets: Dict[int, Set[int]] = {i: set(v) for i, v in adj.items()}

    walls: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    for r in range(grid_rows):
        for c in range(grid_cols):
            cid = r * grid_cols + c
            # Right neighbor — cid < cid2 always (same row, next col)
            if c + 1 < grid_cols:
                cid2 = r * grid_cols + (c + 1)
                if cid2 not in adj_sets[cid]:
                    walls.add(((r, c), (r, c + 1)))
            # Down neighbor — cid < cid2 always (next row, same col)
            if r + 1 < grid_rows:
                cid2 = (r + 1) * grid_cols + c
                if cid2 not in adj_sets[cid]:
                    walls.add(((r, c), (r + 1, c)))
    return walls


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_i4_maze(
    image_path: str,
    grid_width: int = 200,
    grid_height: int = 300,
    cell_size_px: int = 3,
    *,
    dark_threshold: float = 0.3,
    bright_threshold: float = 0.7,
    extra_removal_rate: float = 0.3,
    max_side: int = 512,
    contrast_boost: float = 1.0,
    edge_sigma: float = 1.0,
    edge_low_threshold: float = 0.05,
    edge_high_threshold: float = 0.20,
    edge_waypoint_threshold: float = 0.3,
) -> I4MazeResult:
    """
    Generate a maze using the I4 reverse pipeline (Path-First Masterpiece).

    Implements **DM-1 Part A**: the path-first pipeline that designs the
    solution path first (F3 serpentine fill tracing the image's dark regions)
    and then places walls around it (I4 anti-shortcut Kruskal mold).

    Steps
    -----
    1. Load + preprocess image → grayscale density map (with optional CLAHE).
    2. Build cell grid from density map (H2: each cell carries a luminance value).
    3. Edge detection (K1: Canny) → edge waypoints for path bias.
    4. Classify cells: DARK (< dark_threshold), MID, BRIGHT (> bright_threshold).
    5. BFS flood-fill → connected components of DARK cells (blobs).
    6. Select entrance (darkest border cell) and exit (darkest opposite-side cell).
    7. Order blobs greedily from entrance to exit (nearest-neighbor).
    8. F3 serpentine fill: visit each blob in boustrophedon order, bridging
       gaps through unvisited cells via dark-biased Dijkstra (P1).
    9. I4 mold: anti-shortcut Kruskal builds dead-end branches around the
       solution path without creating shortcuts across it.
    10. Convert internal cell-ID format to coordinate-based I4MazeResult.

    Args:
        image_path:              Path to input image (JPEG / PNG / etc.).
        grid_width:              Number of columns in the cell grid.
        grid_height:             Number of rows in the cell grid.
        cell_size_px:            Pixel size per cell (passed through to result).
        dark_threshold:          Luminance below this → DARK (F3 fill target).
        bright_threshold:        Luminance above this → BRIGHT (transit region).
        extra_removal_rate:      Dead-end branch density [0, 1]. Higher = more
                                 branches hanging off the solution path.
        max_side:                Max image side for resize during preprocessing.
        contrast_boost:          CLAHE intensity [0, 1]. 0 disables CLAHE.
        edge_sigma:              Gaussian sigma for Canny edge detection.
        edge_low_threshold:      Canny lower hysteresis threshold.
        edge_high_threshold:     Canny upper hysteresis threshold.
        edge_waypoint_threshold: Edge strength above which a cell is a K1 waypoint.

    Returns:
        I4MazeResult with walls, solution_path, density_map, entrance, exit_pos.
    """
    # Step 1: Load + preprocess
    image = Image.open(image_path)
    gray = preprocess_image(image, max_side=max_side, contrast_boost=contrast_boost)

    # Step 2: Build cell grid (grid_height = rows, grid_width = cols)
    grid = build_cell_grid(gray, grid_height, grid_width)

    # Step 3: Edge detection → K1 waypoints
    edge_map = detect_edge_map(
        gray, grid_height, grid_width,
        sigma=edge_sigma,
        low_threshold=edge_low_threshold,
        high_threshold=edge_high_threshold,
    )
    edge_waypoints = extract_edge_waypoints(
        edge_map, grid_width, threshold=edge_waypoint_threshold
    )

    # Step 4: Classify cells
    cell_classes = classify_cells(
        grid.luminance,
        dark_thresh=dark_threshold,
        bright_thresh=bright_threshold,
    )

    # Step 5: Find dark blobs
    blobs = find_dark_blobs(cell_classes, grid)

    # Step 6: Entrance / exit selection
    entrance_id, exit_id = find_entrance_exit_path_first(grid, cell_classes)

    # Step 7: Order blobs for path traversal
    ordered_blobs = order_blobs_for_path(blobs, entrance_id, exit_id, grid)

    # Step 8: F3 serpentine fill → solution path + path edges
    solution_ids, path_edges = design_masterpiece_path(
        grid, cell_classes, ordered_blobs,
        entrance_id, exit_id,
        edge_waypoints,
    )

    # Step 9: I4 mold — anti-shortcut Kruskal around path
    adj = build_walls_around_path(
        grid, path_edges, cell_classes,
        extra_removal_rate=extra_removal_rate,
        solution_cells=set(solution_ids),
    )

    # Step 10: Convert to I4MazeResult coordinate format
    walls = _adj_to_walls(adj, grid_height, grid_width)
    solution_path: List[Tuple[int, int]] = [grid.cell_rc(cid) for cid in solution_ids]
    entrance_rc: Tuple[int, int] = grid.cell_rc(entrance_id)
    exit_rc: Tuple[int, int] = grid.cell_rc(exit_id)

    return I4MazeResult(
        grid_width=grid_width,
        grid_height=grid_height,
        cell_size_px=cell_size_px,
        walls=walls,
        solution_path=solution_path,
        density_map=grid.luminance,
        entrance=entrance_rc,
        exit_pos=exit_rc,
    )
