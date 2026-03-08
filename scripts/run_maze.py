#!/usr/bin/env python3
"""
maze-artisan density maze CLI runner.

使い方:
  # 通常モード
  python scripts/run_maze.py --input photo.jpg --output maze.png

  # masterpiece モード（黄金設定 grid_size=8 を一括適用）
  python scripts/run_maze.py --masterpiece --input photo.jpg --output maze.png
  python scripts/run_maze.py --masterpiece --input photo.jpg --output maze.svg

  # オプション確認
  python scripts/run_maze.py --help
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加（backend パッケージをインポートするため）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PIL import Image

from backend.core.density import generate_density_maze, MASTERPIECE_PRESET


def build_parser() -> argparse.ArgumentParser:
    """引数パーサーを構築する（テスト用に分離）。"""
    parser = argparse.ArgumentParser(
        description="maze-artisan: 画像から密度迷路を生成する CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="入力画像パス（JPEG, PNG, BMP 等）",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="出力ファイルパス（.png または .svg）",
    )
    parser.add_argument(
        "--format",
        choices=["png", "svg", "auto"],
        default="auto",
        help="出力フォーマット（auto: 拡張子から自動判定）",
    )
    parser.add_argument(
        "--masterpiece",
        action="store_true",
        help=(
            "masterpiece 黄金設定を一括適用（grid_size=8, thickness_range=1.5, "
            "edge_weight=0.5, use_image_guided=True 等）。"
            "個別パラメータで上書き可能。"
        ),
    )
    # 個別パラメータ（--masterpiece より優先度低い、明示指定で上書き可能）
    parser.add_argument("--grid-size", type=int, default=None,
                        help="グリッドサイズ（--masterpiece 時デフォルト: 8）")
    parser.add_argument("--width", type=int, default=800, help="出力幅 (px)")
    parser.add_argument("--height", type=int, default=600, help="出力高さ (px)")
    parser.add_argument("--stroke-width", type=float, default=None,
                        help="壁の基本線幅（--masterpiece 時デフォルト: 2.0）")
    parser.add_argument("--thickness-range", type=float, default=None,
                        help="可変壁厚の範囲（--masterpiece 時デフォルト: 1.5）")
    parser.add_argument("--dpi", type=int, default=None,
                        help="PNG DPI（300=印刷用, 96=Web用, 省略=メタデータなし）")
    parser.add_argument("--show-solution", action="store_true", default=None,
                        help="解経路を表示する（--masterpiece 時デフォルト: False）")
    parser.add_argument("--max-side", type=int, default=512,
                        help="前処理時の最大辺長")
    return parser


def build_params(args: argparse.Namespace) -> dict:
    """引数から generate_density_maze 用パラメータ dict を構築する。"""
    if args.masterpiece:
        # MASTERPIECE_PRESET を展開してベースに使用
        params = dict(MASTERPIECE_PRESET)
        # 個別指定があれば上書き
        if args.grid_size is not None:
            params["grid_size"] = args.grid_size
        if args.stroke_width is not None:
            params["stroke_width"] = args.stroke_width
        if args.thickness_range is not None:
            params["thickness_range"] = args.thickness_range
        if args.show_solution:
            params["show_solution"] = True
    else:
        params = {
            "grid_size": args.grid_size if args.grid_size is not None else 50,
            "stroke_width": args.stroke_width if args.stroke_width is not None else 2.0,
            "thickness_range": args.thickness_range if args.thickness_range is not None else 1.5,
            "show_solution": bool(args.show_solution) if args.show_solution is not None else True,
        }

    # 共通パラメータ
    params["width"] = args.width
    params["height"] = args.height
    params["max_side"] = args.max_side
    if args.dpi is not None:
        params["png_dpi"] = args.dpi

    return params


def detect_format(args: argparse.Namespace) -> str:
    """出力フォーマット（'png' または 'svg'）を決定する。"""
    if args.format != "auto":
        return args.format
    ext = Path(args.output).suffix.lower()
    return "svg" if ext == ".svg" else "png"


def main(argv: list[str] | None = None) -> int:
    """CLI エントリポイント。argv=None で sys.argv を使用。テストから argv を渡せる。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    # 入力画像読み込み
    try:
        img = Image.open(args.input)
        img.load()
    except FileNotFoundError:
        print(f"エラー: 入力ファイルが見つかりません: {args.input}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"エラー: 画像の読み込みに失敗しました: {e}", file=sys.stderr)
        return 1

    params = build_params(args)
    fmt = detect_format(args)

    # 出力ディレクトリが存在しない場合は作成
    output_dir = Path(args.output).parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # 迷路生成
    result = generate_density_maze(img, **params)

    # 出力
    if fmt == "svg":
        Path(args.output).write_text(result.svg, encoding="utf-8")
    else:
        Path(args.output).write_bytes(result.png_bytes)

    mode_label = "[masterpiece]" if args.masterpiece else "[standard]"
    print(
        f"生成完了 {mode_label}: {args.output} "
        f"(maze_id={result.maze_id}, "
        f"grid={result.grid_rows}x{result.grid_cols})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
