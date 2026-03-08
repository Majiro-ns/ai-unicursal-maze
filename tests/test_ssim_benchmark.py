# -*- coding: utf-8 -*-
"""
SSIM ベンチマーク: 多画像での masterpiece 品質計測（maze_ssim_a6）。

目的:
  A2(壁色改善)・A3(セルサイズ可変化) 実装前の「現状ベースライン」を記録する。
  改善後に同ベンチマークを再実行して効果を定量比較する。

テスト画像 (7種):
  - gradient        : 水平グラデーション（左→右 0→255）
  - diagonal_grad   : 対角グラデーション（左上=黒、右下=白）
  - circle          : 同心円グラデーション（中心=白、周辺=黒）
  - checkerboard    : チェッカーボード（白黒市松）
  - concentric_rings: 同心円リング（circle より線的）
  - face_silhouette : 簡易顔シルエット（楕円+目+口）
  - text_pattern    : 大文字「A」パターン（太線白字・暗背景）

ベースライン実測値（maze_ssim_a6 計測 / masterpiece 3本柱 grid_size=8）:
  - gradient        : 0.5649  [good]  ← スムーズ輝度勾配が masterpiece と相性◎
  - diagonal_grad   : 0.6135  [good]  ← 同上
  - circle          : 0.2608  [poor]  ← 放射状パターンはアルゴリズム構造的に低SSIM
  - checkerboard    : 0.3803  [fair]  ← 高周波パターンは密度マップに粗く反映
  - concentric_rings: 0.1905  [poor]  ← 周期的リングは SSIM 低
  - face_silhouette : 0.2826  [poor]  ← 疎なシルエットはSSIM低
  - text_pattern    : 0.0537  [poor]  ← 極めて疎な線画はSSIM最低

知見: masterpiece アルゴリズムはスムーズな輝度勾配画像と相性が良い。
  高周波・疎なパターン画像では SSIM は低くなる（Edge-SSIM は高い場合あり）。
  SSIM だけが唯一の品質指標ではなく、Edge-SSIM との組み合わせで評価する。

出力:
  - outputs/ssim_benchmark_results.md（test_ssim_benchmark_report で生成）
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# プロジェクトルートを sys.path に追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_quality import generate_and_evaluate_masterpiece
from tests.fixtures import (
    make_circular_gradient,
    make_checkerboard,
    make_diagonal_gradient,
    make_concentric_rings,
    make_face_silhouette,
    make_text_pattern,
)


# ============================================================
# ヘルパー: 水平グラデーション（gradient）
# ============================================================

def _make_gradient(w: int = 64, h: int = 64) -> Image.Image:
    """水平グラデーション（0→255）。evaluate_quality 既存テストと同一。"""
    arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return Image.fromarray(arr, mode="L")


# ============================================================
# ベンチマーク画像カタログ
# ============================================================

#: (label, image_factory, ssim_min)
#: ssim_min は実測値（ベースライン）の -0.05 マージンを取った閾値。
#: A2/A3 改善後に閾値を引き上げること。
BENCHMARK_IMAGES = [
    # スムーズ勾配系: good 基準（≥0.50）
    ("gradient",         lambda: _make_gradient(64, 64),              0.50),
    ("diagonal_grad",    lambda: make_diagonal_gradient(64, 64),       0.55),
    # 放射状・周期系: 現状構造的限界（実測値ベース）
    ("circle",           lambda: make_circular_gradient(64, 64),       0.20),
    ("concentric_rings", lambda: make_concentric_rings(64, 64),        0.14),
    # 高周波パターン
    ("checkerboard",     lambda: make_checkerboard(64, 64, n_tiles=4), 0.30),
    # 複雑形状
    ("face_silhouette",  lambda: make_face_silhouette(64, 64),         0.20),
    # 極疎パターン（最低保証: マスタピース生成が完走すること）
    ("text_pattern",     lambda: make_text_pattern(64, 64),            0.01),
]


# ============================================================
# 個別 SSIM 閾値テスト（全画像 × 実測ベースライン保証）
# ============================================================

@pytest.mark.parametrize("label,img_factory,ssim_min", BENCHMARK_IMAGES)
def test_ssim_minimum_threshold(label: str, img_factory, ssim_min: float) -> None:
    """各画像で masterpiece 生成 → SSIM が実測ベースライン閾値を満たすこと。

    ★閾値は実測値 −0.05 マージン。A2/A3 改善後に引き上げること。
    """
    img = img_factory()
    result = generate_and_evaluate_masterpiece(img)
    ssim = result["ssim"]
    assert ssim >= ssim_min, (
        f"[{label}] SSIM={ssim:.4f} < {ssim_min:.2f} (ベースライン閾値未達)"
    )


# ============================================================
# gradient / diagonal_grad: good 基準（SSIM ≥ 0.50）
# ============================================================

def test_ssim_gradient_good() -> None:
    """水平グラデーション: SSIM ≥ 0.50 (good 基準)。実測値: ~0.5649"""
    img = _make_gradient(64, 64)
    result = generate_and_evaluate_masterpiece(img)
    assert result["ssim"] >= 0.50, (
        f"gradient SSIM={result['ssim']:.4f} < 0.50 [good 未達]"
    )
    assert result["rating"] in ("good", "excellent"), (
        f"rating={result['rating']} (期待: good 以上)"
    )


def test_ssim_diagonal_good() -> None:
    """対角グラデーション: SSIM ≥ 0.55 (good 基準)。実測値: ~0.6135"""
    img = make_diagonal_gradient(64, 64)
    result = generate_and_evaluate_masterpiece(img)
    assert result["ssim"] >= 0.55, (
        f"diagonal_grad SSIM={result['ssim']:.4f} < 0.55 [good 未達]"
    )
    assert result["rating"] in ("good", "excellent"), (
        f"rating={result['rating']} (期待: good 以上)"
    )


# ============================================================
# Edge-SSIM: 全画像で輪郭保持を確認
# ============================================================

@pytest.mark.parametrize("label,img_factory,ssim_min", BENCHMARK_IMAGES)
def test_edge_ssim_reasonable(label: str, img_factory, ssim_min: float) -> None:
    """Edge-SSIM ≥ 0.40: 輪郭保持性は最低限保証される（疎パターンでも有効）。"""
    img = img_factory()
    result = generate_and_evaluate_masterpiece(img)
    edge_ssim = result["edge_ssim"]
    # text_pattern は極めて疎なため Edge-SSIM も低い傾向
    edge_min = 0.40 if label != "text_pattern" else 0.20
    assert edge_ssim >= edge_min, (
        f"[{label}] Edge-SSIM={edge_ssim:.4f} < {edge_min:.2f}"
    )


# ============================================================
# 平均 SSIM 計算・レポート生成
# ============================================================

def test_ssim_benchmark_report() -> None:
    """
    全画像の SSIM を一括計測して平均を計算し、
    outputs/ssim_benchmark_results.md にレポートを出力する。

    ★ベースライン記録テスト: A2/A3 実装後に再実行して比較する。
    """
    results = []
    for label, img_factory, ssim_min in BENCHMARK_IMAGES:
        img = img_factory()
        r = generate_and_evaluate_masterpiece(img)
        results.append({
            "label": label,
            "ssim": r["ssim"],
            "edge_ssim": r["edge_ssim"],
            "rating": r["rating"],
            "ssim_min": ssim_min,
            "pass": r["ssim"] >= ssim_min,
        })

    avg_ssim = sum(r["ssim"] for r in results) / len(results)
    all_pass = all(r["pass"] for r in results)

    # outputs/ssim_benchmark_results.md を書き出す（常に出力）
    _write_benchmark_report(results, avg_ssim)

    assert all_pass, (
        "ベースライン閾値未達の画像あり: "
        + ", ".join(f"{r['label']}({r['ssim']:.4f}<{r['ssim_min']:.2f})"
                    for r in results if not r["pass"])
    )
    # 全体平均の最低ライン（A2/A3 改善効果を定量比較するための基準）
    assert avg_ssim >= 0.25, f"平均SSIM={avg_ssim:.4f} < 0.25 (ベースライン平均)"


# ============================================================
# レポート書き出しヘルパー
# ============================================================

def _write_benchmark_report(results: list[dict], avg_ssim: float) -> None:
    """outputs/ssim_benchmark_results.md にベースライン結果を書き出す。"""
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    report_path = outputs_dir / "ssim_benchmark_results.md"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# SSIM ベンチマーク結果（maze_ssim_a6 ベースライン）",
        "",
        f"生成日時: {timestamp}  ",
        "目的: A2(壁色改善)・A3(セルサイズ可変化) 実装前のベースライン記録  ",
        "設定: masterpiece 3本柱 / grid_size=8 / SSIM評価サイズ 256×256 / 入力画像 64×64",
        "",
        "## 計測結果",
        "",
        "| 画像 | SSIM | Edge-SSIM | Rating | 閾値 | 判定 |",
        "|------|------|-----------|--------|------|------|",
    ]
    for r in results:
        verdict = "✅ PASS" if r["pass"] else "❌ FAIL"
        lines.append(
            f"| {r['label']} | {r['ssim']:.4f} | {r['edge_ssim']:.4f} "
            f"| {r['rating']} | ≥{r['ssim_min']:.2f} | {verdict} |"
        )

    lines += [
        "",
        f"**平均 SSIM: {avg_ssim:.4f}**  ",
        "",
        "## 画像種別ごとの知見",
        "",
        "| 画像種別 | SSIM傾向 | 理由 |",
        "|---------|---------|------|",
        "| スムーズ勾配（gradient/diagonal_grad） | 高（0.55+） | 密度マップに輝度構造が忠実に反映される |",
        "| 放射状/周期（circle/concentric_rings） | 低（0.15-0.27） | 放射状パターンは行走査ベースの迷路と相性が悪い |",
        "| 高周波（checkerboard） | 中（0.30-0.40） | 粗い密度分解能が高周波を平滑化 |",
        "| 複雑形状（face_silhouette） | 低（0.20-0.30） | 疎な構造は迷路全体への影響が小さい |",
        "| 極疎線画（text_pattern） | 極低（0.01-0.06） | 黒背景が支配的で輝度分散が迷路に反映されない |",
        "",
        "## Rating 定義",
        "",
        "| Rating | SSIM 範囲 | 説明 |",
        "|--------|-----------|------|",
        "| excellent | ≥ 0.70 | 傑作候補: 元画像の構造が迷路に深く宿っている |",
        "| good      | ≥ 0.50 | 良品: 「誰の迷路か」がわかるレベル |",
        "| fair      | ≥ 0.30 | 中程度: 迷路に元画像の面影あり |",
        "| poor      | < 0.30  | 要改善: 元画像との構造類似度が低い |",
        "",
        "## 次回比較予定",
        "",
        "- **A2（壁色改善）実装後**: 同ベンチマークを再実行。SSIM の変化量で効果を定量評価。",
        "- **A3（セルサイズ可変化）実装後**: 同ベンチマークを再実行。グリッドサイズ変化の効果を確認。",
        "- 期待: A2/A3 により circle・face_silhouette・text_pattern の SSIM が上昇すること。",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
