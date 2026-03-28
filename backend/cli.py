"""
maze-artisan CLI — DM-6 Bayesian最適化 + 難易度制御 コマンドラインインターフェース

使用例:
    maze-artisan optimize --image input.jpg --trials 100 --output preset.json
    maze-artisan generate --image input.jpg --preset portrait --difficulty hard
    maze-artisan generate --image input.jpg --difficulty-score 0.8 --dry-run
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError:  # pragma: no cover
    sys.exit("typer not installed: pip install typer")

from PIL import Image

app = typer.Typer(
    name="maze-artisan",
    help="AI unicursal maze generator — DM-6 Bayesian optimization + difficulty control.",
    add_completion=False,
)


def _load_image(image_path: Path) -> Image.Image:
    """画像ファイルを PIL Image として読み込む。"""
    if not image_path.exists():
        typer.echo(f"ERROR: Image file not found: {image_path}", err=True)
        raise typer.Exit(code=1)
    try:
        img = Image.open(image_path)
        img.load()
        return img
    except Exception as e:
        typer.echo(f"ERROR: Failed to load image: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("optimize")
def cmd_optimize(
    image: Path = typer.Option(..., "--image", "-i", help="Input image file path."),
    trials: int = typer.Option(100, "--trials", "-n", help="Number of optuna trials."),
    category: str = typer.Option(
        "portrait",
        "--category", "-c",
        help="Image category: portrait / landscape / logo / anime.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output JSON preset file path. If omitted, prints to stdout.",
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducibility."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate inputs without running optimization."),
):
    """
    Bayesian最適化で SSIM 最大化パラメータを探索し、JSON プリセットを出力する。
    """
    from .core.density.dm6_optimizer import optimize_for_image, VALID_CATEGORIES

    valid_cats = sorted(VALID_CATEGORIES)
    if category not in VALID_CATEGORIES:
        typer.echo(f"ERROR: Invalid category '{category}'. Choose from: {valid_cats}", err=True)
        raise typer.Exit(code=1)

    img = _load_image(image)

    typer.echo(f"Image: {image} ({img.size[0]}x{img.size[1]})")
    typer.echo(f"Category: {category}, Trials: {trials}")

    if dry_run:
        typer.echo("[dry-run] Validation passed. Skipping optimization.")
        return

    typer.echo("Running Bayesian optimization...")
    result = optimize_for_image(img, n_trials=trials, category=category, seed=seed)

    out_json = json.dumps(result, indent=2, ensure_ascii=False)

    if output is not None:
        output.write_text(out_json, encoding="utf-8")
        typer.echo(f"Preset saved to: {output}")
        typer.echo(f"Best SSIM: {result['best_value']:.4f}")
    else:
        typer.echo(out_json)


@app.command("generate")
def cmd_generate(
    image: Path = typer.Option(..., "--image", "-i", help="Input image file path."),
    difficulty: str = typer.Option(
        "medium",
        "--difficulty", "-d",
        help="Difficulty level: easy / medium / hard / extreme.",
    ),
    difficulty_score: Optional[float] = typer.Option(
        None,
        "--difficulty-score",
        help="Difficulty score 0.0-1.0 (overrides --difficulty).",
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset", "-p",
        help="Category preset name: portrait / landscape / logo / anime.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output PNG file path. Defaults to <image_stem>_dm6_<difficulty>.png.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate inputs without generating."),
):
    """
    DM-6 難易度制御迷路を生成し PNG として保存する。
    """
    from .core.density.dm6 import generate_dm6_maze, DM6Config, VALID_DIFFICULTIES
    from .core.density.dm6_optimizer import VALID_CATEGORIES

    if difficulty not in VALID_DIFFICULTIES:
        typer.echo(
            f"ERROR: Invalid difficulty '{difficulty}'. "
            f"Choose from: {sorted(VALID_DIFFICULTIES)}",
            err=True,
        )
        raise typer.Exit(code=1)

    if preset is not None and preset not in VALID_CATEGORIES:
        typer.echo(
            f"ERROR: Invalid preset '{preset}'. "
            f"Choose from: {sorted(VALID_CATEGORIES)}",
            err=True,
        )
        raise typer.Exit(code=1)

    img = _load_image(image)

    typer.echo(f"Image: {image} ({img.size[0]}x{img.size[1]})")
    typer.echo(f"Difficulty: {difficulty}" + (f" (score={difficulty_score})" if difficulty_score is not None else ""))
    if preset:
        typer.echo(f"Preset: {preset}")

    if dry_run:
        typer.echo("[dry-run] Validation passed. Skipping generation.")
        return

    config = DM6Config(
        difficulty=difficulty,
        difficulty_score=difficulty_score,
        preset_name=preset,
    )

    typer.echo("Generating maze...")
    result = generate_dm6_maze(img, config)

    out_path = output or Path(f"{image.stem}_dm6_{result.difficulty}.png")
    out_path.write_bytes(result.png_bytes)

    typer.echo(f"Saved: {out_path}")
    typer.echo(f"Grid: {result.grid_rows}×{result.grid_cols}, SSIM: {result.ssim_score:.4f}")
    typer.echo(f"Difficulty: {result.difficulty} (score={result.difficulty_score:.3f})")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
