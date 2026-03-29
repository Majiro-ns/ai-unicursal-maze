# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.6.0] - 2026-03-31

### Added
- **DM-7: `passage_ratio` parameter** — controls the ratio of passage width to cell size (default `0.5`). Setting `passage_ratio=0.15` reduces forced white coverage from 21% to ~5.5%, enabling SSIM scores of 0.60–0.65 (up from 0.4884 ceiling in DM-6).
- `DM7Config` dataclass with `passage_ratio: float = 0.5` field.
- `generate_dm7_maze()` function in `backend.core.density.dm7`.
- CLI subcommand `maze-artisan generate-dm7` with `--passage-ratio` option.
- **PyPI public release** — package now available via `pip install maze-artisan`.

### Changed
- `pyproject.toml`: `build-backend` updated from `setuptools.backends.legacy:build` to `setuptools.build_meta` for PyPI compatibility.
- Test badge updated to reflect 583 passing tests.

### Fixed
- SSIM ceiling issue caused by fixed `passage_width = int(cell_size * 0.5)` — parameterizing this value breaks the 21% white coverage hard limit.

---

## [0.5.0] - 2026-02-28

### Added
- DM-6: density-aware maze generation with Hamiltonian path optimization.
- Bayesian optimization via Optuna for parameter search (`maze-artisan optimize`).
- SSIM scoring against source image for quality evaluation.
- FastAPI backend with `/generate`, `/optimize` endpoints.
- Streamlit frontend for local web UI.
- 387 tests covering core generation pipeline.

### Changed
- Switched from grid-fill approach to line-drawing approach for SVG output.
- Improved portrait/landscape category detection.

---

## [0.4.0] - 2026-01-31

### Added
- DM-5: multi-scale density sampling.
- PDF export via reportlab.
- CLI: `maze-artisan generate`, `maze-artisan optimize`.

---

## [0.1.0] - 2025-12-01

### Added
- Initial release: unicursal maze generation from uploaded images.
- Basic grid-based Hamiltonian path algorithm.
- PNG/SVG output.
