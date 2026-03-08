"""
Path-First Masterpiece Pipeline (I4+F3+G1+H2+K1).

Reverses the standard maze pipeline: instead of building walls first and
finding a path second, this module designs the solution path first (to
trace the input image) and then builds walls around it.

Pipeline:
  Input image -> edge detection (K1) -> cell grid (H2) ->
  path design (F3 serpentine fill) -> wall placement (I4 mold) ->
  rendering (G1 variable width)
"""
from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .grid_builder import CellGrid
from .maze_builder import UnionFind


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DarkBlob:
    """A connected component of DARK cells."""
    cells: List[int]
    centroid: Tuple[float, float]  # (row, col) in grid coordinates
    area: int


# Cell classification constants
DARK = 0
MID = 1
BRIGHT = 2


# ---------------------------------------------------------------------------
# 1-1. classify_cells
# ---------------------------------------------------------------------------

def classify_cells(
    luminance: np.ndarray,
    dark_thresh: float = 0.3,
    bright_thresh: float = 0.7,
) -> np.ndarray:
    """
    Classify each cell as DARK(0), MID(1), or BRIGHT(2) based on luminance.

    Args:
        luminance: (rows, cols) float array of cell luminance values (0-1).
        dark_thresh: Luminance below this is DARK.
        bright_thresh: Luminance above this is BRIGHT.

    Returns:
        (rows, cols) int array with values DARK(0), MID(1), BRIGHT(2).
    """
    result = np.full(luminance.shape, MID, dtype=np.int32)
    result[luminance < dark_thresh] = DARK
    result[luminance > bright_thresh] = BRIGHT
    return result


# ---------------------------------------------------------------------------
# 1-2. find_dark_blobs
# ---------------------------------------------------------------------------

def find_dark_blobs(cell_classes: np.ndarray, grid: CellGrid) -> List[DarkBlob]:
    """
    BFS flood-fill to find connected components of DARK cells.

    Uses 4-neighbor connectivity (up/down/left/right).

    Args:
        cell_classes: (rows, cols) int array from classify_cells().
        grid: CellGrid for dimension info and cell_id/cell_rc conversion.

    Returns:
        List of DarkBlob sorted by area descending.
    """
    rows, cols = grid.rows, grid.cols
    visited = np.zeros((rows, cols), dtype=bool)
    blobs: List[DarkBlob] = []

    for r in range(rows):
        for c in range(cols):
            if cell_classes[r, c] != DARK or visited[r, c]:
                continue
            # BFS flood-fill
            cells: List[int] = []
            queue: deque[Tuple[int, int]] = deque([(r, c)])
            visited[r, c] = True
            sum_r, sum_c = 0.0, 0.0
            while queue:
                cr, cc = queue.popleft()
                cid = grid.cell_id(cr, cc)
                cells.append(cid)
                sum_r += cr
                sum_c += cc
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and cell_classes[nr, nc] == DARK:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            area = len(cells)
            centroid = (sum_r / area, sum_c / area)
            blobs.append(DarkBlob(cells=cells, centroid=centroid, area=area))

    blobs.sort(key=lambda b: b.area, reverse=True)
    return blobs


# ---------------------------------------------------------------------------
# 1-3. find_entrance_exit_path_first
# ---------------------------------------------------------------------------

def find_entrance_exit_path_first(
    grid: CellGrid,
    cell_classes: np.ndarray,
) -> Tuple[int, int]:
    """
    Select entrance and exit for path-first mode.

    Entrance: darkest border cell.
    Exit: darkest border cell on the opposite side.

    Args:
        grid: CellGrid for dimensions.
        cell_classes: (rows, cols) from classify_cells().

    Returns:
        (entrance_cell_id, exit_cell_id)
    """
    from .entrance_exit import _border_cells_by_side

    sides = _border_cells_by_side(grid.rows, grid.cols)
    flat_lum = grid.luminance.flatten()

    _OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}

    # Find darkest border cell across all sides
    best_entrance = 0
    best_side = "top"
    best_lum = float("inf")
    for side_name, cells in sides.items():
        for c in cells:
            lum = float(flat_lum[c])
            if lum < best_lum:
                best_lum = lum
                best_entrance = c
                best_side = side_name

    # Exit: darkest cell on opposite side
    opp_side = _OPPOSITE[best_side]
    opp_cells = sides[opp_side]
    if opp_cells:
        best_exit = min(opp_cells, key=lambda c: float(flat_lum[c]))
    else:
        # Fallback: farthest border cell from entrance
        best_exit = max(sides[best_side], key=lambda c: abs(c - best_entrance))

    return best_entrance, best_exit


# ---------------------------------------------------------------------------
# 1-4. order_blobs_for_path
# ---------------------------------------------------------------------------

def _cell_distance(cid1: int, cid2: int, grid: CellGrid) -> float:
    """Euclidean distance between two cells in grid coordinates."""
    r1, c1 = grid.cell_rc(cid1)
    r2, c2 = grid.cell_rc(cid2)
    return ((r1 - r2) ** 2 + (c1 - c2) ** 2) ** 0.5


def order_blobs_for_path(
    blobs: List[DarkBlob],
    entrance: int,
    exit_cell: int,
    grid: CellGrid,
) -> List[DarkBlob]:
    """
    Order blobs for path traversal using nearest-neighbor greedy.

    Starts from the blob closest to entrance, ends with blob closest to exit.

    Args:
        blobs: List of DarkBlob from find_dark_blobs().
        entrance: Entrance cell ID.
        exit_cell: Exit cell ID.
        grid: CellGrid for coordinate conversion.

    Returns:
        Ordered list of DarkBlob.
    """
    if len(blobs) <= 1:
        return list(blobs)

    def _blob_center_cell(blob: DarkBlob) -> int:
        """Find the cell in blob closest to its centroid."""
        cr, cc = blob.centroid
        return min(blob.cells, key=lambda cid: (
            (grid.cell_rc(cid)[0] - cr) ** 2 + (grid.cell_rc(cid)[1] - cc) ** 2
        ))

    remaining = list(range(len(blobs)))
    ordered: List[DarkBlob] = []

    # Start with blob closest to entrance
    start_idx = min(remaining, key=lambda i: _cell_distance(
        _blob_center_cell(blobs[i]), entrance, grid
    ))
    remaining.remove(start_idx)
    ordered.append(blobs[start_idx])

    # Greedy nearest-neighbor
    while remaining:
        last_center = _blob_center_cell(ordered[-1])
        next_idx = min(remaining, key=lambda i: _cell_distance(
            _blob_center_cell(blobs[i]), last_center, grid
        ))
        remaining.remove(next_idx)
        ordered.append(blobs[next_idx])

    # Move blob closest to exit to the end (if not already)
    if len(ordered) > 2:
        exit_blob_idx = max(range(len(ordered)), key=lambda i: (
            -_cell_distance(_blob_center_cell(ordered[i]), exit_cell, grid)
            if i == len(ordered) - 1 else 0.0
        ))
        # Check if the last blob is already the closest to exit
        closest_to_exit = min(range(len(ordered)), key=lambda i: _cell_distance(
            _blob_center_cell(ordered[i]), exit_cell, grid
        ))
        if closest_to_exit != len(ordered) - 1 and closest_to_exit > 0:
            blob_to_move = ordered.pop(closest_to_exit)
            ordered.append(blob_to_move)

    return ordered


# ---------------------------------------------------------------------------
# 1-5. serpentine_fill_blob
# ---------------------------------------------------------------------------

def _bfs_shortest(grid: CellGrid, start: int, end: int, allowed: Optional[set] = None) -> List[int]:
    """BFS shortest path on grid 4-neighbors. If allowed is set, only use those cells."""
    if start == end:
        return [start]
    rows, cols = grid.rows, grid.cols
    visited = {start}
    prev: Dict[int, int] = {start: -1}
    q: deque[int] = deque([start])
    while q:
        u = q.popleft()
        if u == end:
            break
        r, c = grid.cell_rc(u)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                v = grid.cell_id(nr, nc)
                if v not in visited and (allowed is None or v in allowed):
                    visited.add(v)
                    prev[v] = u
                    q.append(v)
    if end not in prev:
        # No path within allowed; fall back to unrestricted BFS
        if allowed is not None:
            return _bfs_shortest(grid, start, end, allowed=None)
        return [start, end]  # last resort
    path = []
    cur = end
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def serpentine_fill_blob(
    blob: DarkBlob,
    grid: CellGrid,
    entry_cell: int,
    exit_cell: int,
) -> List[int]:
    """
    Fill a blob with a boustrophedon (serpentine) pattern.

    Scans rows alternately left-to-right and right-to-left. Between any
    two consecutive waypoints that are not grid-adjacent, inserts a BFS
    shortest path to ensure every step is a valid 4-neighbor move.

    Args:
        blob: DarkBlob to fill.
        grid: CellGrid for coordinate conversion.
        entry_cell: Desired entry point (must be in blob).
        exit_cell: Desired exit point (must be in blob).

    Returns:
        List of cell IDs forming a path through all blob cells.
        Every consecutive pair is 4-neighbor adjacent.
    """
    if not blob.cells:
        return []
    if len(blob.cells) == 1:
        return list(blob.cells)

    blob_set = set(blob.cells)

    # Group cells by row
    rows_dict: Dict[int, List[int]] = {}
    for cid in blob.cells:
        r, c = grid.cell_rc(cid)
        rows_dict.setdefault(r, []).append((c, cid))

    # Sort rows
    sorted_rows = sorted(rows_dict.keys())

    # Determine scan direction: start from entry_cell's row side
    entry_r = grid.cell_rc(entry_cell)[0] if entry_cell in blob_set else grid.cell_rc(blob.cells[0])[0]
    if sorted_rows:
        if abs(entry_r - sorted_rows[0]) > abs(entry_r - sorted_rows[-1]):
            sorted_rows = sorted_rows[::-1]

    # Build waypoint sequence (serpentine order)
    waypoints: List[int] = []
    for i, row_idx in enumerate(sorted_rows):
        cells_in_row = rows_dict[row_idx]
        cells_in_row.sort(key=lambda x: x[0])
        if i % 2 == 1:
            cells_in_row = cells_in_row[::-1]
        for _, cid in cells_in_row:
            waypoints.append(cid)

    # Try to start from entry_cell
    if entry_cell in blob_set and entry_cell in waypoints:
        idx = waypoints.index(entry_cell)
        if idx > 0:
            waypoints = waypoints[idx:] + waypoints[:idx][::-1]

    # Stitch waypoints with BFS to guarantee 4-neighbor adjacency
    path: List[int] = [waypoints[0]]
    for i in range(1, len(waypoints)):
        prev_cell = path[-1]
        next_cell = waypoints[i]
        if next_cell == prev_cell:
            continue
        # Check if 4-neighbor adjacent
        rp, cp = grid.cell_rc(prev_cell)
        rn, cn = grid.cell_rc(next_cell)
        if abs(rp - rn) + abs(cp - cn) == 1:
            path.append(next_cell)
        else:
            # Insert BFS bridge (prefer staying in blob)
            bridge = _bfs_shortest(grid, prev_cell, next_cell, allowed=blob_set)
            path.extend(bridge[1:])  # skip first (=prev_cell, already in path)

    return path


# ---------------------------------------------------------------------------
# 1-6. connect_through_bright
# ---------------------------------------------------------------------------

def connect_through_bright(
    grid: CellGrid,
    cell_classes: np.ndarray,
    from_cell: int,
    to_cell: int,
    edge_waypoints: Optional[Set[int]] = None,
) -> List[int]:
    """
    Connect two cells via Dijkstra through the grid, preferring BRIGHT cells.

    Cost model:
      - BRIGHT cells: low cost (1.0) - preferred for transit
      - DARK cells: high cost (10.0) - avoid re-entering dark regions
      - Edge waypoints: medium cost (3.0) - somewhat preferred
      - MID cells: medium cost (5.0)

    Uses 4-neighbor grid connectivity (not maze adjacency).

    Args:
        grid: CellGrid for dimensions and coordinate conversion.
        cell_classes: (rows, cols) from classify_cells().
        from_cell: Starting cell ID.
        to_cell: Target cell ID.
        edge_waypoints: Set of cell IDs on image edges (from K1).

    Returns:
        List of cell IDs from from_cell to to_cell (inclusive).
    """
    if from_cell == to_cell:
        return [from_cell]

    if edge_waypoints is None:
        edge_waypoints = set()

    rows, cols = grid.rows, grid.cols
    n = grid.num_cells

    # Cost per cell class
    flat_classes = cell_classes.flatten()

    def _cost(cid: int) -> float:
        cls = flat_classes[cid]
        if cls == BRIGHT:
            return 1.0
        elif cls == DARK:
            return 10.0
        else:  # MID
            if cid in edge_waypoints:
                return 3.0
            return 5.0

    INF = float("inf")
    dist = [INF] * n
    prev = [-1] * n
    dist[from_cell] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, from_cell)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == to_cell:
            break
        r, c = grid.cell_rc(u)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                v = grid.cell_id(nr, nc)
                edge_cost = (_cost(u) + _cost(v)) / 2.0
                nd = d + edge_cost
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

    # Reconstruct path
    if dist[to_cell] == INF:
        # No path found - return direct line (should not happen in connected grid)
        return [from_cell, to_cell]

    path: List[int] = []
    cur = to_cell
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# 1-7. design_masterpiece_path
# ---------------------------------------------------------------------------

def _greedy_dark_walk(
    grid: CellGrid,
    cell_classes_flat: np.ndarray,
    dark_cells: Set[int],
    start: int,
    exit_cell: int,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """
    Greedy walk that visits dark cells, producing a simple (non-revisiting) path.

    Uses Warnsdorff-like heuristic: among unvisited neighbors, prefer
    dark cells, breaking ties by choosing the neighbor with fewest
    unvisited dark neighbors.

    When stuck (no unvisited dark neighbors), tries ANY unvisited neighbor
    first. If completely stuck (no unvisited neighbors at all), uses BFS
    through unvisited cells only to reach the nearest unvisited dark cell.
    This avoids creating loops that would be erased later.

    Args:
        grid: CellGrid for neighbor computation.
        cell_classes_flat: Flattened cell classes array.
        dark_cells: Set of DARK cell IDs to visit.
        start: Starting cell ID.
        exit_cell: Exit cell ID.
        rng: Random number generator (for direction shuffle).

    Returns:
        List of cell IDs — a simple path (no cell appears twice).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    rows, cols = grid.rows, grid.cols
    visited: Set[int] = {start}
    path: List[int] = [start]
    remaining_dark = dark_cells - {start}

    _DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def _count_dark_unvisited_neighbors(cid: int) -> int:
        count = 0
        r, c = grid.cell_rc(cid)
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                nb = grid.cell_id(nr, nc)
                if nb not in visited and nb in dark_cells:
                    count += 1
        return count

    def _get_unvisited_neighbors(cid: int) -> Tuple[List[int], List[int]]:
        """Returns (dark_neighbors, all_unvisited_neighbors)."""
        r, c = grid.cell_rc(cid)
        dirs = list(_DIRS)
        rng.shuffle(dirs)
        dark_nbs: List[int] = []
        all_nbs: List[int] = []
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                nb = grid.cell_id(nr, nc)
                if nb not in visited:
                    all_nbs.append(nb)
                    if nb in dark_cells:
                        dark_nbs.append(nb)
        return dark_nbs, all_nbs

    while remaining_dark:
        cur = path[-1]
        dark_nbs, all_nbs = _get_unvisited_neighbors(cur)

        if dark_nbs:
            # Warnsdorff: pick dark neighbor with fewest unvisited dark neighbors
            chosen = min(dark_nbs, key=lambda n: (
                _count_dark_unvisited_neighbors(n) + rng.random() * 0.1
            ))
            visited.add(chosen)
            remaining_dark.discard(chosen)
            path.append(chosen)
        elif all_nbs:
            # No dark neighbors, but unvisited non-dark neighbors exist.
            # Pick the one closest to remaining dark (Manhattan distance).
            if remaining_dark:
                chosen = min(all_nbs, key=lambda nb: _min_dist_to_dark(
                    nb, remaining_dark, grid) + rng.random() * 0.5)
            else:
                chosen = all_nbs[0]
            visited.add(chosen)
            remaining_dark.discard(chosen)
            path.append(chosen)
        else:
            # Completely stuck: no unvisited neighbors at all.
            # BFS through unvisited cells to nearest dark cell.
            if not remaining_dark:
                break
            bridge = _bfs_unvisited_to_dark(grid, cur, remaining_dark, visited, rng)
            if bridge is None:
                break
            for cid in bridge[1:]:
                visited.add(cid)
                remaining_dark.discard(cid)
                path.append(cid)

    return path


def _bfs_unvisited_to_target(
    grid: CellGrid,
    start: int,
    target: int,
    visited: Set[int],
    rng: np.random.Generator,
) -> Optional[List[int]]:
    """
    BFS from start to target through unvisited cells only.
    Target is allowed even if in visited. Start is always the path origin.
    Returns path from start to target, or None if unreachable.
    """
    if start == target:
        return [start]
    rows, cols = grid.rows, grid.cols
    _DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    seen = {start}
    prev: Dict[int, int] = {start: -1}
    q: deque[int] = deque([start])
    while q:
        u = q.popleft()
        if u == target:
            path: List[int] = []
            cur = u
            while cur != -1:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path
        r, c = grid.cell_rc(u)
        dirs = list(_DIRS)
        rng.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                nb = grid.cell_id(nr, nc)
                if nb not in seen and (nb not in visited or nb == target):
                    seen.add(nb)
                    prev[nb] = u
                    q.append(nb)
    return None


def _walk_dark_blob(
    grid: CellGrid,
    flat_classes: np.ndarray,
    blob_remaining: Set[int],
    visited: Set[int],
    remaining_dark: Set[int],
    solution: List[int],
    rng: np.random.Generator,
) -> None:
    """
    Greedy Warnsdorff walk within a single dark blob.
    Modifies solution, visited, remaining_dark, blob_remaining in-place.
    Only steps on unvisited cells adjacent to current position.
    Prefers dark cells, uses Warnsdorff tie-breaking.
    """
    rows, cols = grid.rows, grid.cols
    _DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def _count_dark_unvisited(cid: int) -> int:
        count = 0
        r, c = grid.cell_rc(cid)
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                nb = grid.cell_id(nr, nc)
                if nb not in visited and nb in blob_remaining:
                    count += 1
        return count

    while blob_remaining:
        cur = solution[-1]
        r_cur, c_cur = grid.cell_rc(cur)

        # Find unvisited dark neighbors in this blob
        dirs = list(_DIRS)
        rng.shuffle(dirs)
        dark_nbs: List[int] = []
        for dr, dc in dirs:
            nr, nc = r_cur + dr, c_cur + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                nb = grid.cell_id(nr, nc)
                if nb not in visited and nb in blob_remaining:
                    dark_nbs.append(nb)

        if dark_nbs:
            chosen = min(dark_nbs, key=lambda n: (
                _count_dark_unvisited(n) + rng.random() * 0.1
            ))
            visited.add(chosen)
            blob_remaining.discard(chosen)
            remaining_dark.discard(chosen)
            solution.append(chosen)
        else:
            # Can't reach more blob cells from here without revisiting.
            # Try BFS through unvisited non-blob cells to reach blob cells.
            bridge = _bfs_unvisited_to_dark(grid, cur, blob_remaining, visited, rng)
            if bridge is None:
                break
            for cid in bridge[1:]:
                visited.add(cid)
                blob_remaining.discard(cid)
                remaining_dark.discard(cid)
                solution.append(cid)


def _min_dist_to_dark(cid: int, dark_remaining: Set[int], grid: CellGrid) -> float:
    """Manhattan distance from cid to nearest cell in dark_remaining."""
    if not dark_remaining:
        return 0.0
    r, c = grid.cell_rc(cid)
    return min(abs(r - grid.cell_rc(d)[0]) + abs(c - grid.cell_rc(d)[1])
               for d in dark_remaining)


def _bfs_unvisited_to_dark(
    grid: CellGrid,
    start: int,
    dark_remaining: Set[int],
    visited: Set[int],
    rng: np.random.Generator,
) -> Optional[List[int]]:
    """
    BFS from start through UNVISITED cells only to reach nearest dark cell.

    Unlike _bfs_to_nearest_dark which allows revisits (creating loops),
    this version only traverses unvisited cells. If no path exists through
    unvisited cells, returns None.
    """
    rows, cols = grid.rows, grid.cols
    _DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    seen = {start}
    prev: Dict[int, int] = {start: -1}
    q: deque[int] = deque([start])
    while q:
        u = q.popleft()
        if u in dark_remaining and u != start:
            path: List[int] = []
            cur = u
            while cur != -1:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path
        r, c = grid.cell_rc(u)
        dirs = list(_DIRS)
        rng.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                nb = grid.cell_id(nr, nc)
                if nb not in seen and nb not in visited:
                    seen.add(nb)
                    prev[nb] = u
                    q.append(nb)
    return None


def _bfs_to_nearest_dark(
    grid: CellGrid,
    start: int,
    dark_remaining: Set[int],
    visited: Set[int],
) -> Optional[List[int]]:
    """BFS from start to nearest cell in dark_remaining, allowing revisits."""
    rows, cols = grid.rows, grid.cols
    seen = {start}
    prev: Dict[int, int] = {start: -1}
    q: deque[int] = deque([start])
    while q:
        u = q.popleft()
        if u in dark_remaining and u != start:
            # Found nearest dark cell
            path = []
            cur = u
            while cur != -1:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path
        r, c = grid.cell_rc(u)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                nb = grid.cell_id(nr, nc)
                if nb not in seen:
                    seen.add(nb)
                    prev[nb] = u
                    q.append(nb)
    return None


def _bfs_to_nearest_dark_jittered(
    grid: CellGrid,
    start: int,
    dark_remaining: Set[int],
    visited: Set[int],
    rng: np.random.Generator,
) -> Optional[List[int]]:
    """
    BFS to nearest unvisited dark cell with direction jitter.

    Unlike plain BFS (which produces axis-aligned shortest paths creating □
    artifacts), this version shuffles neighbor exploration order at each node.
    The result is a slightly longer but visually irregular bridge path that
    breaks up rectangular blank areas.
    """
    rows, cols = grid.rows, grid.cols
    _DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    seen = {start}
    prev: Dict[int, int] = {start: -1}
    q: deque[int] = deque([start])
    while q:
        u = q.popleft()
        if u in dark_remaining and u != start:
            path: List[int] = []
            cur = u
            while cur != -1:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path
        r, c = grid.cell_rc(u)
        dirs = list(_DIRS)
        rng.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                nb = grid.cell_id(nr, nc)
                if nb not in seen:
                    seen.add(nb)
                    prev[nb] = u
                    q.append(nb)
    return None


def _dijkstra_dark_biased(
    grid: CellGrid,
    cell_classes_flat: np.ndarray,
    dark_cells: Set[int],
    start: int,
    end: int,
) -> List[int]:
    """
    Dijkstra shortest path biased to prefer dark cells.

    Cost: DARK=1, MID=3, BRIGHT=8. This produces a short path that
    preferentially routes through dark areas (the image's features).
    """
    rows, cols = grid.rows, grid.cols
    n = grid.num_cells
    INF = float("inf")
    dist = [INF] * n
    prev = [-1] * n
    dist[start] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == end:
            break
        r, c = grid.cell_rc(u)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                v = grid.cell_id(nr, nc)
                cls_v = cell_classes_flat[v]
                if cls_v == DARK:
                    cost = 1.0
                elif cls_v == MID:
                    cost = 3.0
                else:
                    cost = 8.0
                nd = d + cost
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

    if dist[end] == INF:
        return _bfs_shortest(grid, start, end)

    path: List[int] = []
    cur = end
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _dijkstra_avoiding(
    grid: CellGrid,
    cell_classes_flat: np.ndarray,
    dark_cells: Set[int],
    start: int,
    end: int,
    avoid: Set[int],
) -> List[int]:
    """
    Dijkstra like _dijkstra_dark_biased but avoids cells in `avoid` set.
    The `end` cell is allowed even if in `avoid`.
    Falls back to unrestricted Dijkstra if no path exists.
    """
    rows, cols = grid.rows, grid.cols
    n = grid.num_cells
    INF = float("inf")
    dist = [INF] * n
    prev = [-1] * n
    dist[start] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == end:
            break
        r, c = grid.cell_rc(u)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                v = grid.cell_id(nr, nc)
                if v in avoid and v != end:
                    continue  # skip visited cells
                cls_v = cell_classes_flat[v]
                if cls_v == DARK:
                    cost = 1.0
                elif cls_v == MID:
                    cost = 3.0
                else:
                    cost = 8.0
                nd = d + cost
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

    if dist[end] == INF:
        # Fall back to unrestricted
        return _dijkstra_dark_biased(grid, cell_classes_flat, dark_cells, start, end)

    path: List[int] = []
    cur = end
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def design_masterpiece_path(
    grid: CellGrid,
    cell_classes: np.ndarray,
    blobs: List[DarkBlob],
    entrance: int,
    exit_cell: int,
    edge_waypoints: Optional[Set[int]] = None,
) -> Tuple[List[int], Set[Tuple[int, int]]]:
    """
    Design the masterpiece solution path that IS the image trace.

    Architecture (V6 — no-shortcut spanning tree):
      1. Greedy dark walk creates a long path that visits all dark cells.
         This path IS the solution — the unique maze path from entrance
         to exit.
      2. Deduplicate the walk into a simple (non-revisiting) path.
         Where the walk revisits a cell, detour the revisit as a
         dead-end branch instead.
      3. Return the deduplicated solution + all path edges.
      4. build_walls_around_path will connect remaining cells as
         dead-end branches only (never creating shortcuts across the
         solution path).

    The key invariant: the spanning tree's unique entrance→exit path
    IS the designed walk. No shortcuts exist because Kruskal edges
    never bridge two solution-path cells.

    Returns:
        (solution_path, path_edges) where:
        - solution_path: Non-revisiting path from entrance to exit
          (visits all reachable dark cells).
        - path_edges: Edges on the solution path (cycle-free).
    """
    dark_cells: Set[int] = set()
    for blob in blobs:
        dark_cells.update(blob.cells)

    flat_classes = cell_classes.flatten()

    if not dark_cells:
        path = _bfs_shortest(grid, entrance, exit_cell)
        edges = set()
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            edges.add((min(a, b), max(a, b)))
        return path, edges

    # === Step 1: Greedy walk through all dark cells ===
    # Single continuous walk. Prefers dark cells (Warnsdorff heuristic),
    # but will step on any unvisited neighbor when stuck. Never revisits
    # (simple path guaranteed).
    rng = np.random.default_rng(42)
    raw_walk = _greedy_dark_walk(grid, flat_classes, dark_cells, entrance, exit_cell, rng)

    # Reach exit if not already there
    if raw_walk[-1] != exit_cell:
        visited_walk = set(raw_walk)
        bridge = _bfs_unvisited_to_target(
            grid, raw_walk[-1], exit_cell, visited_walk, rng)
        if bridge is not None:
            for cid in bridge[1:]:
                raw_walk.append(cid)
        else:
            bridge = _bfs_shortest(grid, raw_walk[-1], exit_cell)
            for cid in bridge[1:]:
                raw_walk.append(cid)

    # Loop-erase any revisits (from fallback bridge)
    solution: List[int] = []
    cell_idx: Dict[int, int] = {}
    for cid in raw_walk:
        if cid in cell_idx:
            erase_from = cell_idx[cid]
            for erased in solution[erase_from + 1:]:
                if erased in cell_idx and cell_idx[erased] > erase_from:
                    del cell_idx[erased]
            solution = solution[:erase_from + 1]
        else:
            solution.append(cid)
            cell_idx[cid] = len(solution) - 1
    solution_set = set(solution)

    # === Step 3: Build solution edges ===
    path_edges: Set[Tuple[int, int]] = set()
    for i in range(len(solution) - 1):
        a, b = solution[i], solution[i + 1]
        edge_key = (min(a, b), max(a, b))
        path_edges.add(edge_key)

    return solution, path_edges


def _closest_cell_to(cells: List[int], target: int, grid: CellGrid) -> int:
    """Find the cell in `cells` closest to `target`."""
    tr, tc = grid.cell_rc(target)
    return min(cells, key=lambda cid: (
        (grid.cell_rc(cid)[0] - tr) ** 2 + (grid.cell_rc(cid)[1] - tc) ** 2
    ))


# ---------------------------------------------------------------------------
# 1-8. build_walls_around_path
# ---------------------------------------------------------------------------

def build_walls_around_path(
    grid: CellGrid,
    path_edges: Set[Tuple[int, int]],
    cell_classes: np.ndarray,
    extra_removal_rate: float = 0.3,
    rng: Optional[np.random.Generator] = None,
    solution_cells: Optional[Set[int]] = None,
) -> Dict[int, List[int]]:
    """
    Build maze walls using I4 mold technique (V6 no-shortcut).

    Key invariant: Kruskal edges NEVER connect two solution-path cells
    unless they are already adjacent on the path. This prevents shortcuts
    that would allow the solver to bypass the designed solution.

    Non-path cells are connected as dead-end branches hanging off the
    solution path (or off other branches).

    1. Pre-connect all path edges in UnionFind.
    2. Kruskal on remaining edges, but SKIP any edge where both endpoints
       are solution-path cells (anti-shortcut rule).
    3. If some cells remain disconnected after step 2, connect them via
       edges to the nearest non-solution neighbor.

    Args:
        grid: CellGrid with walls list.
        path_edges: Set of (cell_a, cell_b) edges from the designed path.
        cell_classes: (rows, cols) from classify_cells().
        extra_removal_rate: Accepted for backward compat (ignored).
        rng: Random number generator.
        solution_cells: Set of cell IDs on the solution path. If provided,
            edges between two solution cells (not in path_edges) are blocked.

    Returns:
        Adjacency list Dict[int, List[int]] representing the maze.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if solution_cells is None:
        solution_cells = set()

    n = grid.num_cells
    uf = UnionFind(n)
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}

    # Step 1: Pre-connect path edges (these MUST be open passages)
    for a, b in path_edges:
        uf.union(a, b)
        if b not in adj[a]:
            adj[a].append(b)
            adj[b].append(a)

    # Step 2: Kruskal on remaining edges (anti-shortcut)
    flat_classes = cell_classes.flatten()

    def _biased_weight(c1: int, c2: int, w: float) -> float:
        cls1 = flat_classes[c1]
        cls2 = flat_classes[c2]
        avg_cls = (cls1 + cls2) / 2.0
        bias = (avg_cls / 2.0) * 0.3
        return w - bias

    biased_walls = []
    for c1, c2, w in grid.walls:
        edge = (min(c1, c2), max(c1, c2))
        if edge in path_edges:
            continue
        # ANTI-SHORTCUT: skip edges where both endpoints are on solution path
        if c1 in solution_cells and c2 in solution_cells:
            continue
        bw = _biased_weight(c1, c2, w)
        biased_walls.append((c1, c2, bw))

    biased_walls.sort(key=lambda x: x[2])

    for c1, c2, _ in biased_walls:
        if uf.union(c1, c2):
            adj[c1].append(c2)
            adj[c2].append(c1)

    # Step 3: Connect any remaining disconnected cells
    # (may happen if anti-shortcut blocks the only path to some cells)
    # Allow solution-solution edges as last resort for connectivity
    for c1, c2, w in grid.walls:
        if uf.union(c1, c2):
            adj[c1].append(c2)
            adj[c2].append(c1)

    return adj
