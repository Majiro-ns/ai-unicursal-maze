#!/usr/bin/env python3
"""
maze-artisan masterpiece 品質定量評価スクリプト（cmd_357k_a8）

入力画像と出力迷路 PNG の構造類似度（SSIM）を計算する。

SSIM の解釈:
  1.0 = 完全一致（迷路が元画像そのもの）
  0.8+ = 高類似（迷路に元画像の構造が明確に反映）
  0.5-0.8 = 中程度（迷路らしさと元画像の面影が共存）
  0.5未満 = 低類似（迷路として機能するが元画像との関連は薄い）

目標値の考え方（masterpiece基準案）:
  SSIM >= 0.30: 最低ライン（元画像の輪郭が迷路に反映されている）
  SSIM >= 0.50: 良品（「誰の迷路か」がわかるレベル）
  SSIM >= 0.70: 傑作候補（元画像の細部まで迷路に宿っている）

使い方:
  python3 scripts/evaluate_quality.py --input photo.jpg --output maze.png
  python3 scripts/evaluate_quality.py --input photo.jpg  # 自動生成して評価
  python3 scripts/evaluate_quality.py --input photo.jpg --grid 50 --preset face
  python3 scripts/evaluate_quality.py --benchmark       # サンプル画像で複数プリセット比較

出力例:
  SSIM スコア: 0.4821 (中程度: 迷路に元画像の面影あり)
  Edge-SSIM:  0.6103 (輪郭保持スコア)
  評価: ✅ 良品基準クリア
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# プロジェクトルートを sys.path に追加
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# SSIM 計算（skimage に依存しない実装 + skimage 高精度版の両対応）
# ============================================================

def _ssim_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    シンプルな SSIM 実装（skimage 非依存）。
    img1, img2: float64, shape (H, W), 値域 0.0-1.0
    """
    K1, K2 = 0.01, 0.03
    L = 1.0  # 輝度範囲
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return float(numerator / denominator)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    SSIM を計算する（skimage 利用可能なら高精度版を使用）。
    img1, img2: float64, shape (H, W), 値域 0.0-1.0
    """
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(img1, img2, data_range=1.0))
    except ImportError:
        return _ssim_simple(img1, img2)


def compute_edge_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    エッジマップ同士の SSIM（輪郭保持スコア）。
    元画像の輪郭が迷路の壁に反映されているかを評価する。
    """
    try:
        from skimage.feature import canny
        edge1 = canny(img1, sigma=1.5).astype(np.float64)
        edge2 = canny(img2, sigma=1.5).astype(np.float64)
        return compute_ssim(edge1, edge2)
    except ImportError:
        # Canny 非利用: 輝度差による近似
        grad1 = np.abs(np.gradient(img1)[0]) + np.abs(np.gradient(img1)[1])
        grad2 = np.abs(np.gradient(img2)[0]) + np.abs(np.gradient(img2)[1])
        grad1 = np.clip(grad1 / (grad1.max() + 1e-8), 0, 1)
        grad2 = np.clip(grad2 / (grad2.max() + 1e-8), 0, 1)
        return compute_ssim(grad1, grad2)


# ============================================================
# 前処理
# ============================================================

def preprocess_for_ssim(pil_img: Image.Image, target_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    PIL.Image → グレースケール float64 配列（0.0-1.0）。
    SSIM 計算のため同一サイズにリサイズする。
    """
    gray = pil_img.convert("L").resize(target_size, Image.LANCZOS)
    return np.asarray(gray, dtype=np.float64) / 255.0


# ============================================================
# 評価ロジック
# ============================================================

def evaluate_quality(
    input_img: Image.Image,
    output_png_bytes: bytes,
    target_size: tuple[int, int] = (256, 256),
) -> dict:
    """
    入力画像と出力迷路 PNG を比較し品質スコアを返す。

    Returns:
        dict:
            ssim: float          SSIM スコア (0-1)
            edge_ssim: float     エッジ SSIM スコア (0-1)
            rating: str          "excellent" / "good" / "fair" / "poor"
            message: str         日本語評価メッセージ
            target_size: tuple   評価時のリサイズ解像度
    """
    output_img = Image.open(io.BytesIO(output_png_bytes))

    arr_in = preprocess_for_ssim(input_img, target_size)
    arr_out = preprocess_for_ssim(output_img, target_size)

    ssim = compute_ssim(arr_in, arr_out)
    edge_ssim = compute_edge_ssim(arr_in, arr_out)

    if ssim >= 0.70:
        rating = "excellent"
        message = "傑作候補: 元画像の構造が迷路に深く宿っている"
    elif ssim >= 0.50:
        rating = "good"
        message = "良品: 「誰の迷路か」がわかるレベル"
    elif ssim >= 0.30:
        rating = "fair"
        message = "中程度: 迷路に元画像の面影あり"
    else:
        rating = "poor"
        message = "要改善: 元画像との構造類似度が低い"

    return {
        "ssim": round(ssim, 4),
        "edge_ssim": round(edge_ssim, 4),
        "rating": rating,
        "message": message,
        "target_size": target_size,
    }


def print_result(result: dict, input_path: Optional[str] = None, output_path: Optional[str] = None) -> None:
    """評価結果を標準出力に表示する。"""
    icon = {"excellent": "🏆", "good": "✅", "fair": "⚠️", "poor": "❌"}.get(result["rating"], "")
    if input_path:
        print(f"入力: {input_path}")
    if output_path:
        print(f"出力: {output_path}")
    print(f"評価サイズ: {result['target_size'][0]}×{result['target_size'][1]}px")
    print(f"SSIM スコア:  {result['ssim']:.4f}")
    print(f"Edge-SSIM:   {result['edge_ssim']:.4f}")
    print(f"評価: {icon} {result['message']}")


# ============================================================
# 密度迷路自動生成 + 評価
# ============================================================

def generate_and_evaluate(
    input_img: Image.Image,
    grid_size: int = 50,
    preset: str = "generic",
    use_texture: bool = True,
    use_heuristic: bool = True,
    n_segments: int = 4,
) -> dict:
    """
    入力画像から密度迷路を自動生成し、品質評価を行う。
    """
    from backend.core.density import generate_density_maze

    result = generate_density_maze(
        input_img,
        grid_size=grid_size,
        use_texture=use_texture,
        use_heuristic=use_heuristic,
        preset=preset,
        n_segments=n_segments,
    )
    quality = evaluate_quality(input_img, result.png_bytes)
    quality["maze_id"] = result.maze_id
    quality["grid_size"] = grid_size
    quality["preset"] = preset
    quality["solution_path_length"] = len(result.solution_path)
    return quality


# ============================================================
# masterpiece 3本柱設定での品質評価（cmd_358k_a2 発見）
# ============================================================

#: masterpiece 3本柱パラメータ（最適化済み）
#: 可変壁厚 / ループ密度 / 解法ルーティング を全有効にした「黄金設定」。
#:
#: SSIM 実験結果（cmd_358k_a2, 2026-03-08）:
#:   grid_size=5  : gradient=0.5969 / circle=0.5441  [good]
#:   grid_size=8  : gradient=0.5566 / circle=0.5072  [good]
#:   grid_size=10 : gradient=0.5314 / circle=0.4856  [good/fair]
#:   grid_size=30 : gradient=0.4506 / circle=0.2849  [fair/poor]
#:   grid_size=100+: 0.16–0.45  [fair/poor]
#:
#: 知見:
#:   - 小グリッド（5〜10）+ 大出力 → 各セルが大きく輝度寄与が高い → SSIM↑
#:   - 大グリッド（100+）は細壁になり SSIM↓（Edge-SSIM は高い）
#:   - "excellent" (≥0.70) は二値壁レンダリングの構造的限界で未達
#:   - グラデーション・円形画像は "good" 到達。高コントラスト画像は "fair" が上限
MASTERPIECE_OPTIMAL_PARAMS = {
    "grid_size": 8,          # 5〜10 が SSIM と視認性のバランス点
    "thickness_range": 1.5,
    "extra_removal_rate": 0.5,
    "dark_threshold": 0.3,
    "light_threshold": 0.7,
    "use_image_guided": True,
    "solution_highlight": False,
    "show_solution": False,
    "edge_weight": 0.5,
    "stroke_width": 2.0,
}


def generate_and_evaluate_masterpiece(
    input_img: Image.Image,
    grid_size: Optional[int] = None,
) -> dict:
    """
    masterpiece 3本柱設定で迷路を生成し、品質評価を行う。

    MASTERPIECE_OPTIMAL_PARAMS をベースに生成。
    grid_size を省略すると 8 を使用（SSIM と視認性のバランス点）。

    Returns:
        dict: evaluate_quality() の結果 + maze_id / grid_size / solution_path_length
    """
    from backend.core.density import generate_density_maze

    params = dict(MASTERPIECE_OPTIMAL_PARAMS)
    if grid_size is not None:
        params["grid_size"] = grid_size

    result = generate_density_maze(input_img, **params)
    quality = evaluate_quality(input_img, result.png_bytes)
    quality["maze_id"] = result.maze_id
    quality["grid_size"] = params["grid_size"]
    quality["preset"] = "masterpiece"
    quality["solution_path_length"] = len(result.solution_path)
    return quality


def benchmark(input_img: Image.Image) -> list[dict]:
    """
    複数プリセットで迷路を生成し、SSIM スコアを比較する。
    """
    results = []
    configs = [
        {"preset": "generic", "grid_size": 30, "use_texture": False},
        {"preset": "generic", "grid_size": 30, "use_texture": True},
        {"preset": "face",    "grid_size": 40, "use_texture": True},
        {"preset": "landscape", "grid_size": 40, "use_texture": True},
    ]
    for cfg in configs:
        label = f"{cfg['preset']} grid={cfg['grid_size']} texture={cfg['use_texture']}"
        print(f"  評価中: {label} ...", end="", flush=True)
        try:
            q = generate_and_evaluate(input_img, **cfg)
            q["label"] = label
            results.append(q)
            print(f" SSIM={q['ssim']:.4f}, Edge={q['edge_ssim']:.4f} [{q['rating']}]")
        except Exception as e:
            print(f" ERROR: {e}")

    # masterpiece 3本柱設定も比較に追加
    label_mp = "masterpiece grid=8 (3本柱最適設定)"
    print(f"  評価中: {label_mp} ...", end="", flush=True)
    try:
        q = generate_and_evaluate_masterpiece(input_img)
        q["label"] = label_mp
        results.append(q)
        print(f" SSIM={q['ssim']:.4f}, Edge={q['edge_ssim']:.4f} [{q['rating']}]")
    except Exception as e:
        print(f" ERROR: {e}")

    return results


# ============================================================
# CLI エントリポイント
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="maze-artisan masterpiece 品質定量評価（SSIM）"
    )
    parser.add_argument("--input", "-i", type=str, help="入力画像パス")
    parser.add_argument("--output", "-o", type=str, help="出力 PNG パス（省略時: 自動生成）")
    parser.add_argument("--grid", type=int, default=50, help="グリッドサイズ（default: 50）")
    parser.add_argument("--preset", type=str, default="generic",
                        choices=["generic", "face", "landscape"],
                        help="テクスチャプリセット（default: generic）")
    parser.add_argument("--size", type=int, default=256,
                        help="SSIM 評価時のリサイズ解像度（default: 256）")
    parser.add_argument("--benchmark", action="store_true",
                        help="複数プリセットで比較評価を実行")
    args = parser.parse_args()

    if not args.input and not args.benchmark:
        # サンプル画像で動作確認
        print("[サンプルモード] 合成グラジエント画像で動作確認します。")
        arr = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
        img = Image.fromarray(arr, mode="L")
    else:
        if not args.input:
            print("[ERROR] --input が必要です。", file=sys.stderr)
            sys.exit(1)
        img = Image.open(args.input)

    target_size = (args.size, args.size)

    if args.benchmark:
        print("=== Benchmark: 複数プリセット SSIM 比較 ===")
        results = benchmark(img)
        print("\n=== 結果サマリー ===")
        results_sorted = sorted(results, key=lambda r: r["ssim"], reverse=True)
        for r in results_sorted:
            print(f"  {r['ssim']:.4f} (Edge: {r['edge_ssim']:.4f}) [{r['rating']:9s}] {r['label']}")
        best = results_sorted[0] if results_sorted else None
        if best:
            print(f"\n最高スコア: {best['ssim']:.4f} — {best['label']}")
        return

    if args.output:
        # 既存の PNG ファイルを読み込んで評価
        with open(args.output, "rb") as f:
            png_bytes = f.read()
        result = evaluate_quality(img, png_bytes, target_size=target_size)
        print_result(result, args.input, args.output)
    else:
        # 自動生成して評価
        print(f"[自動生成] grid={args.grid}, preset={args.preset} で密度迷路を生成中...")
        result = generate_and_evaluate(
            img, grid_size=args.grid, preset=args.preset,
        )
        print(f"生成完了: maze_id={result['maze_id']}, 解経路長={result['solution_path_length']}")
        print_result(result, args.input)


if __name__ == "__main__":
    main()
