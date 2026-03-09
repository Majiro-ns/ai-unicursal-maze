#!/usr/bin/env python3
"""
SSIMベンチマーク: i4_renderer.py 各アプローチの SSIM 計測 (cmd_380k_a7)

i4_renderer.py の線幅制御改善アプローチを比較:
  - baseline: 現状（線形補間 density→thickness）
  - approach1: 輝度→線幅の冪乗マッピング（暗部強調）
  - approach2: アンチエイリアシング（ガウシアンブラー）
  - approach3: セル全体塗りつぶし（density に比例した面積充填）

使い方:
  cd /mnt/c/Users/owner/Desktop/llama3_wallthinker/ai-unicursal-maze
  python3 scripts/ssim_i4_benchmark.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from backend.core.density.i4_renderer import (
    I4MazeResult,
    _norm_key,
    _build_walls_normalized,
    _build_solution_set,
    _g1_thickness,
    _G1_THICK_DARK,
    _G1_THICK_BRIGHT,
    _SOL_COLOR_PNG,
    _PATH_COLOR_PNG,
    _BG_COLOR_PNG,
)
from evaluate_quality import compute_ssim, preprocess_for_ssim


# ============================================================
# i4_pipeline → i4_renderer 変換ヘルパー
# ============================================================

def pipeline_result_to_renderer_result(pipeline_result) -> I4MazeResult:
    """i4_pipeline.I4MazeResult を i4_renderer.I4MazeResult に変換。

    フィールド差異:
      - exit_pos (pipeline) → exit (renderer)
      - density_map: pipeline=luminance(0=暗部), renderer=density(1=暗部)
        → 1 - luminance を渡す
    """
    return I4MazeResult(
        grid_width=pipeline_result.grid_width,
        grid_height=pipeline_result.grid_height,
        cell_size_px=pipeline_result.cell_size_px,
        walls=pipeline_result.walls,
        solution_path=pipeline_result.solution_path,
        density_map=1.0 - pipeline_result.density_map,  # luminance→density変換
        entrance=pipeline_result.entrance,
        exit=pipeline_result.exit_pos,
    )


# ============================================================
# ベースライン: 現状の _render_png（線形補間）
# ============================================================

def render_baseline(result: I4MazeResult) -> bytes:
    """現状の線形補間レンダリング（i4_renderer._render_png と同一）。"""
    cs = result.cell_size_px
    img_w = result.grid_width * cs
    img_h = result.grid_height * cs

    img = Image.new("RGB", (img_w, img_h), _BG_COLOR_PNG)
    draw = ImageDraw.Draw(img)

    walls_norm = _build_walls_normalized(result.walls)
    sol_set = _build_solution_set(result.solution_path)
    density = result.density_map

    for r in range(result.grid_height):
        for c in range(result.grid_width):
            cx1 = int((c + 0.5) * cs)
            cy1 = int((r + 0.5) * cs)

            if c + 1 < result.grid_width:
                key = _norm_key((r, c), (r, c + 1))
                if key not in walls_norm:
                    cx2 = int((c + 1.5) * cs)
                    cy2 = cy1
                    if key in sol_set:
                        avg_d = (float(density[r, c]) + float(density[r, c + 1])) / 2.0
                        w = max(1, round(_g1_thickness(avg_d)))
                        draw.line([(cx1, cy1), (cx2, cy2)], fill=_SOL_COLOR_PNG, width=w)
                    else:
                        draw.line([(cx1, cy1), (cx2, cy2)], fill=_PATH_COLOR_PNG, width=1)

            if r + 1 < result.grid_height:
                key = _norm_key((r, c), (r + 1, c))
                if key not in walls_norm:
                    cx2 = cx1
                    cy2 = int((r + 1.5) * cs)
                    if key in sol_set:
                        avg_d = (float(density[r, c]) + float(density[r + 1, c])) / 2.0
                        w = max(1, round(_g1_thickness(avg_d)))
                        draw.line([(cx1, cy1), (cx2, cy2)], fill=_SOL_COLOR_PNG, width=w)
                    else:
                        draw.line([(cx1, cy1), (cx2, cy2)], fill=_PATH_COLOR_PNG, width=1)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================
# アプローチ1: 冪乗マッピング（暗部強調 + 細かい中間調）
# ============================================================

def _g1_thickness_power(avg_density: float, gamma: float = 2.0) -> float:
    """冪乗マッピング: density^gamma で暗部を強調。
    gamma=1.0: 線形（ベースライン）
    gamma=2.0: 暗部寄りにシフト
    gamma=0.5: 明部寄りにシフト
    境界値: density=0→1px, density=1→8px （TI-6互換）
    """
    clamped = max(0.0, min(1.0, avg_density))
    powered = clamped ** gamma
    return _G1_THICK_BRIGHT + (_G1_THICK_DARK - _G1_THICK_BRIGHT) * powered


def render_approach1(result: I4MazeResult, gamma: float = 2.0) -> bytes:
    """アプローチ1: 冪乗マッピングで暗部に太線・明部に細線をより強調。"""
    cs = result.cell_size_px
    img_w = result.grid_width * cs
    img_h = result.grid_height * cs

    img = Image.new("RGB", (img_w, img_h), _BG_COLOR_PNG)
    draw = ImageDraw.Draw(img)

    walls_norm = _build_walls_normalized(result.walls)
    sol_set = _build_solution_set(result.solution_path)
    density = result.density_map

    for r in range(result.grid_height):
        for c in range(result.grid_width):
            cx1 = int((c + 0.5) * cs)
            cy1 = int((r + 0.5) * cs)

            if c + 1 < result.grid_width:
                key = _norm_key((r, c), (r, c + 1))
                if key not in walls_norm:
                    cx2 = int((c + 1.5) * cs)
                    cy2 = cy1
                    if key in sol_set:
                        avg_d = (float(density[r, c]) + float(density[r, c + 1])) / 2.0
                        w = max(1, round(_g1_thickness_power(avg_d, gamma)))
                        draw.line([(cx1, cy1), (cx2, cy2)], fill=_SOL_COLOR_PNG, width=w)
                    else:
                        draw.line([(cx1, cy1), (cx2, cy2)], fill=_PATH_COLOR_PNG, width=1)

            if r + 1 < result.grid_height:
                key = _norm_key((r, c), (r + 1, c))
                if key not in walls_norm:
                    cx2 = cx1
                    cy2 = int((r + 1.5) * cs)
                    if key in sol_set:
                        avg_d = (float(density[r, c]) + float(density[r + 1, c])) / 2.0
                        w = max(1, round(_g1_thickness_power(avg_d, gamma)))
                        draw.line([(cx1, cy1), (cx2, cy2)], fill=_SOL_COLOR_PNG, width=w)
                    else:
                        draw.line([(cx1, cy1), (cx2, cy2)], fill=_PATH_COLOR_PNG, width=1)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================
# アプローチ2: アンチエイリアシング（高解像度描画→ダウンスケール）
# ============================================================

def render_approach2(result: I4MazeResult, scale: int = 4) -> bytes:
    """アプローチ2: 高解像度で描画してダウンスケール（アンチエイリアシング効果）。

    scale=4: 4倍解像度で描画 → LANCZOS でダウンスケール
    中間輝度がなめらかなグレーとして表現され、元画像の輝度勾配に近づく。
    """
    cs = result.cell_size_px
    img_w = result.grid_width * cs
    img_h = result.grid_height * cs

    # scale倍の高解像度でまず描画
    hi_w, hi_h = img_w * scale, img_h * scale
    cs_hi = cs * scale

    img_hi = Image.new("RGB", (hi_w, hi_h), _BG_COLOR_PNG)
    draw = ImageDraw.Draw(img_hi)

    walls_norm = _build_walls_normalized(result.walls)
    sol_set = _build_solution_set(result.solution_path)
    density = result.density_map

    for r in range(result.grid_height):
        for c in range(result.grid_width):
            cx1 = int((c + 0.5) * cs_hi)
            cy1 = int((r + 0.5) * cs_hi)

            if c + 1 < result.grid_width:
                key = _norm_key((r, c), (r, c + 1))
                if key not in walls_norm:
                    cx2 = int((c + 1.5) * cs_hi)
                    cy2 = cy1
                    if key in sol_set:
                        avg_d = (float(density[r, c]) + float(density[r, c + 1])) / 2.0
                        w = max(scale, round(_g1_thickness(avg_d) * scale))
                        draw.line([(cx1, cy1), (cx2, cy2)], fill=_SOL_COLOR_PNG, width=w)
                    else:
                        draw.line([(cx1, cy1), (cx2, cy2)], fill=_PATH_COLOR_PNG, width=scale)

            if r + 1 < result.grid_height:
                key = _norm_key((r, c), (r + 1, c))
                if key not in walls_norm:
                    cx2 = cx1
                    cy2 = int((r + 1.5) * cs_hi)
                    if key in sol_set:
                        avg_d = (float(density[r, c]) + float(density[r + 1, c])) / 2.0
                        w = max(scale, round(_g1_thickness(avg_d) * scale))
                        draw.line([(cx1, cy1), (cx2, cy2)], fill=_SOL_COLOR_PNG, width=w)
                    else:
                        draw.line([(cx1, cy1), (cx2, cy2)], fill=_PATH_COLOR_PNG, width=scale)

    # ダウンスケール（LANCZOS でアンチエイリアシング）
    img = img_hi.resize((img_w, img_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================
# アプローチ3: セル全体塗りつぶし（面積充填）
# ============================================================

def render_approach3(result: I4MazeResult) -> bytes:
    """アプローチ3: 解経路セルを輝度に比例した面積で塗りつぶす。

    各解経路セルを density に応じた輝度（グレー）で矩形塗りつぶし。
    線幅調整ではなく「セル全体の輝度」で元画像を再現。
    密度 1.0（暗部）→ 黒(0), 密度 0.0（明部）→ 白(255) に対応。
    """
    cs = result.cell_size_px
    img_w = result.grid_width * cs
    img_h = result.grid_height * cs

    img = Image.new("RGB", (img_w, img_h), _BG_COLOR_PNG)
    draw = ImageDraw.Draw(img)

    walls_norm = _build_walls_normalized(result.walls)
    sol_set = _build_solution_set(result.solution_path)
    density = result.density_map

    # 解経路セルを density に応じたグレーで塗りつぶす
    sol_cells = set()
    for cell in result.solution_path:
        sol_cells.add(cell)

    for r in range(result.grid_height):
        for c in range(result.grid_width):
            if (r, c) in sol_cells:
                # density=1→黒(0), density=0→白(255)
                d = float(density[r, c])
                gray_val = int(255 * (1.0 - d))
                color = (gray_val, gray_val, gray_val)
                x0 = c * cs
                y0 = r * cs
                x1 = x0 + cs - 1
                y1 = y0 + cs - 1
                draw.rectangle([(x0, y0), (x1, y1)], fill=color)

    # 非解経路の通路も淡く描画（迷路構造を残す）
    for r in range(result.grid_height):
        for c in range(result.grid_width):
            cx1 = int((c + 0.5) * cs)
            cy1 = int((r + 0.5) * cs)

            if c + 1 < result.grid_width:
                key = _norm_key((r, c), (r, c + 1))
                if key not in walls_norm and key not in sol_set:
                    cx2 = int((c + 1.5) * cs)
                    draw.line([(cx1, cy1), (cx2, cy1)], fill=_PATH_COLOR_PNG, width=1)

            if r + 1 < result.grid_height:
                key = _norm_key((r, c), (r + 1, c))
                if key not in walls_norm and key not in sol_set:
                    cy2 = int((r + 1.5) * cs)
                    draw.line([(cx1, cy1), (cx1, cy2)], fill=_PATH_COLOR_PNG, width=1)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================
# メイン: ベンチマーク実行
# ============================================================

def run_benchmark(image_path: str, grid_width: int = 50, grid_height: int = 50,
                  cell_size_px: int = 3) -> dict:
    """ベンチマーク実行。各アプローチの SSIM を計測して比較表を返す。"""
    from backend.core.density.i4_pipeline import generate_i4_maze

    print(f"[1/5] 画像読み込み: {image_path}")
    orig_img = Image.open(image_path)

    print(f"[2/5] i4_pipeline で迷路生成中 (grid={grid_width}x{grid_height})...")
    pipeline_result = generate_i4_maze(
        image_path,
        grid_width=grid_width,
        grid_height=grid_height,
        cell_size_px=cell_size_px,
    )
    renderer_result = pipeline_result_to_renderer_result(pipeline_result)
    print(f"  解経路長: {len(pipeline_result.solution_path)} セル")

    print("[3/5] 各アプローチでPNG生成 + SSIM計測中...")

    target_size = (256, 256)
    arr_in = preprocess_for_ssim(orig_img, target_size)

    approaches = {}

    # ベースライン
    print("  ▶ baseline (線形補間)...", end="", flush=True)
    png = render_baseline(renderer_result)
    arr_out = preprocess_for_ssim(Image.open(io.BytesIO(png)), target_size)
    ssim_b = compute_ssim(arr_in, arr_out)
    approaches["baseline (線形補間)"] = {"ssim": ssim_b, "png": png}
    print(f" SSIM={ssim_b:.4f}")

    # アプローチ1a: gamma=1.5
    print("  ▶ approach1a (冪乗 gamma=1.5)...", end="", flush=True)
    png = render_approach1(renderer_result, gamma=1.5)
    arr_out = preprocess_for_ssim(Image.open(io.BytesIO(png)), target_size)
    ssim = compute_ssim(arr_in, arr_out)
    approaches["approach1a (gamma=1.5)"] = {"ssim": ssim, "png": png}
    print(f" SSIM={ssim:.4f}")

    # アプローチ1b: gamma=2.0
    print("  ▶ approach1b (冪乗 gamma=2.0)...", end="", flush=True)
    png = render_approach1(renderer_result, gamma=2.0)
    arr_out = preprocess_for_ssim(Image.open(io.BytesIO(png)), target_size)
    ssim = compute_ssim(arr_in, arr_out)
    approaches["approach1b (gamma=2.0)"] = {"ssim": ssim, "png": png}
    print(f" SSIM={ssim:.4f}")

    # アプローチ1c: gamma=0.5（明部強調）
    print("  ▶ approach1c (冪乗 gamma=0.5)...", end="", flush=True)
    png = render_approach1(renderer_result, gamma=0.5)
    arr_out = preprocess_for_ssim(Image.open(io.BytesIO(png)), target_size)
    ssim = compute_ssim(arr_in, arr_out)
    approaches["approach1c (gamma=0.5)"] = {"ssim": ssim, "png": png}
    print(f" SSIM={ssim:.4f}")

    # アプローチ2: アンチエイリアシング
    print("  ▶ approach2 (アンチエイリアシング scale=4)...", end="", flush=True)
    png = render_approach2(renderer_result, scale=4)
    arr_out = preprocess_for_ssim(Image.open(io.BytesIO(png)), target_size)
    ssim = compute_ssim(arr_in, arr_out)
    approaches["approach2 (アンチエイリアシング)"] = {"ssim": ssim, "png": png}
    print(f" SSIM={ssim:.4f}")

    # アプローチ3: セル塗りつぶし
    print("  ▶ approach3 (セル塗りつぶし)...", end="", flush=True)
    png = render_approach3(renderer_result)
    arr_out = preprocess_for_ssim(Image.open(io.BytesIO(png)), target_size)
    ssim = compute_ssim(arr_in, arr_out)
    approaches["approach3 (セル塗りつぶし)"] = {"ssim": ssim, "png": png}
    print(f" SSIM={ssim:.4f}")

    print("[4/5] PNG保存 (outputs/cmd380k_a7/)...")
    out_dir = PROJECT_ROOT / "outputs" / "cmd380k_a7"
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, data in approaches.items():
        safe = label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        path = out_dir / f"{safe}.png"
        path.write_bytes(data["png"])

    print("[5/5] 結果サマリー:")
    sorted_approaches = sorted(approaches.items(), key=lambda x: x[1]["ssim"], reverse=True)
    print(f"\n{'アプローチ':<40} {'SSIM':>8}")
    print("-" * 50)
    for label, data in sorted_approaches:
        mark = " ← BEST" if label == sorted_approaches[0][0] else ""
        print(f"{label:<40} {data['ssim']:>8.4f}{mark}")

    best_label, best_data = sorted_approaches[0]
    print(f"\n最良: {best_label}, SSIM={best_data['ssim']:.4f}")

    return {
        "approaches": {k: v["ssim"] for k, v in approaches.items()},
        "best": best_label,
        "best_ssim": best_data["ssim"],
        "baseline_ssim": approaches["baseline (線形補間)"]["ssim"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/input/test3.jpg")
    parser.add_argument("--grid-width", type=int, default=50)
    parser.add_argument("--grid-height", type=int, default=50)
    parser.add_argument("--cell-size", type=int, default=3)
    args = parser.parse_args()

    image_path = str(PROJECT_ROOT / args.image) if not Path(args.image).is_absolute() else args.image
    result = run_benchmark(
        image_path,
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        cell_size_px=args.cell_size,
    )
