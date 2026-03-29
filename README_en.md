# maze-artisan

[![PyPI version](https://img.shields.io/pypi/v/maze-artisan.svg)](https://pypi.org/project/maze-artisan/)
[![Python versions](https://img.shields.io/pypi/pyversions/maze-artisan.svg)](https://pypi.org/project/maze-artisan/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![tests](https://img.shields.io/badge/tests-822%20passed-brightgreen)

**A unicursal maze generator where solving reveals an image.**

Upload any photo → maze-artisan traces a single Hamiltonian path through your image → solving the maze from entrance to exit gradually reconstructs the original picture.

---

## Features

- **Image-faithful unicursal mazes** — every maze has exactly one solution; the solved path reveals the source image via SSIM-optimized routing
- **DM-7 passage_ratio control** — parametric passage width (default `0.5`, use `0.15` for photo-accurate results, SSIM 0.60–0.65)
- **Bayesian parameter search** — Optuna-powered optimizer finds the best grid size, passage ratio, and tonal parameters for your image automatically
- **4 difficulty levels** — easy (6×6) / medium (10×10) / hard (14×14) / extreme (16×16)
- **4 image presets** — portrait / landscape / logo / anime, tuned for each category
- **Multiple output formats** — PNG, SVG, PDF (A4/A3 print-ready at 300 DPI)
- **CLI + Python API + FastAPI backend** — use as a command-line tool, import as a library, or run as a local web app

---

## Quick Start

```bash
pip install maze-artisan
```

```python
from PIL import Image
from backend.core.density.dm7 import generate_dm7_maze, DM7Config

image = Image.open("photo.jpg")

# High-fidelity maze: passage_ratio=0.15 breaks the SSIM ceiling
result = generate_dm7_maze(image, DM7Config(passage_ratio=0.15, difficulty="hard"))

with open("maze.png", "wb") as f:
    f.write(result.png_bytes)

print(f"SSIM: {result.ssim_score:.4f}")  # expect 0.60–0.65
```

---

## Gallery

| Category | Description | Expected SSIM |
|---|---|---|
| Portrait | Face / hair detail preserved | 0.58–0.65 |
| Landscape | Horizon lines, sky gradients | 0.55–0.62 |
| Logo | High-contrast edges, clean lines | 0.62–0.70 |
| Anime | Outline emphasis, flat fill | 0.60–0.68 |
| Gradient | Smooth tone transitions | 0.60–0.65 |

> Gallery images coming soon. Run `maze-artisan optimize --image your_photo.jpg` to generate your own.

---

## How It Works

### 1. Preprocessing
The input image is converted to grayscale and analyzed for tonal distribution. An edge map is extracted to guide path routing.

### 2. Hamiltonian Path Construction
A grid is overlaid on the image. The algorithm constructs a Hamiltonian path — visiting every cell exactly once — using a density-weighted graph where darker regions attract more path segments. This ensures the solved path visually reconstructs the image.

### 3. passage_ratio (DM-7 key parameter)
Each grid cell contains one passage. `passage_ratio` controls the ratio of passage width to cell size:

```
passage_width = int(cell_size * passage_ratio)
```

| passage_ratio | Effect |
|---|---|
| `0.5` (default) | Standard width; 21% of image is forced white → SSIM ceiling ≈ 0.49 |
| `0.15` | Narrow passages; forced white coverage drops to ~5.5% → SSIM 0.60–0.65 |
| `0.10` | Masterpiece preset; maximum image fidelity |

### 4. Bayesian Optimization (optional)
`maze-artisan optimize` runs Optuna trials to find the best `(grid_size, passage_ratio, difficulty)` combination for your specific image, maximizing SSIM score.

---

## Installation

### From PyPI (recommended)
```bash
pip install maze-artisan
```

### From source
```bash
git clone https://github.com/Majiro-ns/ai-unicursal-maze
cd ai-unicursal-maze
pip install -e ".[dev]"
```

### With API server
```bash
pip install "maze-artisan[api]"
uvicorn backend.app:app --reload
```

---

## CLI Reference

### `generate` — Create a maze from an image
```bash
# Basic usage
maze-artisan generate --image photo.jpg --difficulty hard --output maze.png

# DM-7 high-fidelity mode
maze-artisan generate --image photo.jpg --difficulty hard --passage-ratio 0.15

# With category preset
maze-artisan generate --image photo.jpg --preset portrait --difficulty medium

# Dry-run (validate inputs only)
maze-artisan generate --image photo.jpg --difficulty extreme --dry-run
```

### `optimize` — Bayesian parameter search
```bash
# Find best parameters for a portrait (100 trials)
maze-artisan optimize --image photo.jpg --trials 100 --category portrait --output preset.json

# Dry-run
maze-artisan optimize --image photo.jpg --category logo --dry-run
```

### Difficulty levels

| Level | Grid | Extra removal | Character |
|---|---|---|---|
| `easy` | 6×6 | 40% | Many loops, forgiving |
| `medium` | 10×10 | 15% | Balanced |
| `hard` | 14×14 | 5% | Few loops, challenging |
| `extreme` | 16×16 | 0% | Pure spanning tree |

---

## Python API Reference

### `generate_dm7_maze(image, config) → MazeResult`

```python
from backend.core.density.dm7 import generate_dm7_maze, DM7Config

config = DM7Config(
    difficulty="hard",       # "easy" | "medium" | "hard" | "extreme"
    passage_ratio=0.15,      # float 0.05–0.50; lower = higher SSIM
    grid_size=14,            # override grid size (optional)
    category="portrait",     # "portrait" | "landscape" | "logo" | "anime"
)

result = generate_dm7_maze(image, config)
print(result.ssim_score)    # float: SSIM against source image
print(result.grid_rows)     # int
print(result.grid_cols)     # int
result.png_bytes            # bytes: PNG output
result.svg_content          # str: SVG output
```

### `MazeResult` fields

| Field | Type | Description |
|---|---|---|
| `png_bytes` | `bytes` | PNG-encoded maze image |
| `svg_content` | `str` | SVG markup |
| `ssim_score` | `float` | Structural similarity to source (0–1) |
| `grid_rows` | `int` | Number of grid rows |
| `grid_cols` | `int` | Number of grid columns |
| `passage_ratio` | `float` | Passage ratio used |
| `difficulty` | `str` | Difficulty level used |

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests (822 tests)
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=term-missing
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Links

- **PyPI**: https://pypi.org/project/maze-artisan/
- **GitHub**: https://github.com/Majiro-ns/ai-unicursal-maze
- **Issues**: https://github.com/Majiro-ns/ai-unicursal-maze/issues
