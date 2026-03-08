# maze-artisan Phase3 品質レポート

**作成日**: 2026-03-14
**作成者**: 足軽6（P9クロスレビュー含む）
**全テスト数**: 442 PASS（テストファイル30本）

---

## 1. テスト数内訳（ファイル別）

| テストファイル | 件数 | カテゴリ |
|---|---|---|
| test_cli_masterpiece.py | 30 | CLI・masterpieceフラグ |
| test_density_image_guided_luminance.py | 15 | 画像適応ルーティング |
| test_density_maze_phase1.py | 49 | Phase1/2 基本機能 |
| test_density_phase2_clahe.py | 8 | CLAHE コントラスト |
| test_density_phase2_edge.py | 37 | Canny エッジ強調 |
| test_density_phase2_variable_wall.py | 11 | 可変壁厚 |
| test_density_phase2b_density_control.py | 43 | ループ密度制御 |
| test_density_phase3_svg_quality.py | 25 | SVGグループ化・品質 |
| test_density_solver.py | 31 | ソルバー |
| test_dummy_decorator.py | 1 | デコレータ |
| test_e2e_density_maze.py | 18 | E2E統合 |
| test_euler_path.py | 6 | オイラー路 |
| test_face_regressions.py | 5 | 顔画像回帰 |
| test_features.py | 12 | 特徴量 |
| test_grayscale_walls.py | 10 | グレースケール壁（A2）|
| test_masterpiece_integration.py | 17 | masterpiece統合 |
| test_maze_generator.py | 2 | 基本生成器 |
| test_maze_v11.py | 4 | v11 回帰 |
| test_performance.py | 9 | パフォーマンス |
| test_phase3_masterpiece.py | 19 | Phase3実画像テスト |
| test_quality_evaluation.py | 20 | SSIM品質評価 |
| test_skeleton_stability.py | 9 | スケルトン安定性 |
| test_solver_paths.py | 3 | ソルバーパス |
| test_t10_tradeoff.py | 8 | T10トレードオフ |
| test_t11_path_aesthetics.py | 8 | T11美的経路 |
| test_t13_unique_solution.py | 8 | T13一意解 |
| test_t7_ui_options.py | 11 | T7 UIオプション |
| test_t8_solver_integration.py | 8 | T8ソルバー統合 |
| test_t9_difficulty_metrics.py | 12 | T9難易度指標 |
| test_v2_pipeline.py | 3 | v2 パイプライン |
| **合計** | **442** | |

---

## 2. 実装状態一覧（Phase1/2/3）

### Phase 1（基本迷路生成）✅ 完了
| 機能 | 状態 | ファイル |
|---|---|---|
| Kruskal + Union-Find spanning tree | ✅ | maze_builder.py |
| 輝度ベース壁重み | ✅ | grid_builder.py |
| BFS 入口・出口決定（木の直径） | ✅ | entrance_exit.py |
| SVG/PNG エクスポート | ✅ | exporter.py |
| 画像前処理（CLAHE 対応） | ✅ | preprocess.py |

### Phase 2（表現品質向上）✅ 完了
| 機能 | 状態 | ファイル |
|---|---|---|
| 可変壁厚（Xu-Kaplan `_wall_stroke()`） | ✅ | exporter.py |
| セグメンテーション + テクスチャ | ✅ | segment.py, texture.py |
| SPIRAL テクスチャ | ✅ | grid_builder.py |
| Canny エッジ強調（edge_weight） | ✅ | edge_enhancer.py |
| 解ヒューリスティクス（明部葉ノード優先） | ✅ | entrance_exit.py |
| ループ密度制御（post_process_density） | ✅ | maze_builder.py |
| 画像適応ルーティング（Dijkstra, 柱3） | ✅ | entrance_exit.py |
| masterpiece 白線描画（解経路塗りつぶし） | ✅ | exporter.py |

### Phase 3（高品質・高解像度）✅ 完了（一部残課題）
| 機能 | 状態 | ファイル |
|---|---|---|
| SVG グループ化 + path 最適化 | ✅ | exporter.py |
| 壁厚ヒストグラム可視化 | ✅ | exporter.py |
| PNG DPI メタデータ | ✅ | exporter.py |
| 高解像度最適化（400x600で1.5秒） | ✅ | grid_builder.py, maze_builder.py |
| SSIM 品質評価スクリプト | ✅ | scripts/visual_quality_test.py |
| masterpiece 黄金設定（MASTERPIECE_PRESET） | ✅ | __init__.py |
| グレースケール壁（`_wall_color()`） | ✅ | exporter.py（A2, cmd_360k_a2） |
| セルサイズ可変化 | 📋 設計書のみ | outputs/design_variable_cell_size.md |

---

## 3. 3本柱の状態

| 柱 | 機能 | パラメータ | SSIM 貢献度 |
|---|---|---|---|
| **柱1**: 可変壁厚 | `_wall_stroke(sw, lum, range)` | `thickness_range=1.5` | 中 |
| **柱2**: ループ密度 | `post_process_density()` | `extra_removal_rate=0.5` | 中 |
| **柱3**: 画像適応ルーティング | `find_image_guided_path()` | `use_image_guided=True` | 高 |
| **拡張**: グレースケール壁 | `_wall_color(lum)` | （常時有効） | 限定的 |

### MASTERPIECE_PRESET（黄金設定）
```python
{
    "grid_size": 8,
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
```

---

## 4. P9クロスレビュー: グレースケール壁（cmd_360k_a2）

### CR-1: CLI実行確認
```
python3 -m pytest tests/test_grayscale_walls.py -v
10 passed in 0.45s  ✅
```

### CR-2: 手計算検証（`_wall_color()` 公式 `v = int(avg_lum * 220)`）

| avg_lum | 公式計算 | 期待値 | 実際の返り値 | 判定 |
|---|---|---|---|---|
| 0.0 | int(0.0×220)=0 | rgb(0,0,0) | rgb(0,0,0) | ✅ |
| 0.5 | int(0.5×220)=110 | rgb(110,110,110) | rgb(110,110,110) | ✅ |
| 1.0 | int(1.0×220)=220 | rgb(220,220,220) | rgb(220,220,220) | ✅ |
| 0.25 | int(0.25×220)=55 | rgb(55,55,55) | rgb(55,55,55) | ✅ |
| 0.75 | int(0.75×220)=165 | rgb(165,165,165) | rgb(165,165,165) | ✅ |

### CR-3: 単調増加検証（独立計算）
```
lum: [0.0, 0.1, ..., 1.0]
v:   [0, 22, 44, 66, 88, 110, 132, 154, 176, 198, 220]
結果: 単調増加 ✅（等差数列: 公差22）
```

### CR-4: SSIM影響分析

コミット記載の SSIM 値を評価:

| 画像 | grid_size | 変更前 | 変更後 | 差分 | 評価 |
|---|---|---|---|---|---|
| gradient | 8 | 0.5566 | 0.5726 | +0.016 | ✅ 改善 |
| gradient | 5 | — | 0.6062 | — | ✅ good |
| gradient | 30 | 0.4476 | 0.4876 | +0.040 | ✅ 改善 |
| circle | 8 | 0.5072 | 0.2314 | -0.276 | ❌ 大幅低下 |

**所見**: circle 画像での SSIM 急落（-27.6pt）が懸念。明部の壁が淡灰（rgb(220,220,220)）になることで、
白背景との差が生じにくくなり、円形パターンの再現性が低下したと考えられる。
`show_solution=False`（masterpieceモード）では白線が白背景に溶け込む設計なので、
グレースケール壁との相性を要確認。

### CR-5: 改善提案（実装はしない・提案のみ）

1. **circle SSIM 低下の根本原因**: 明部（lum≈1.0）の壁色 rgb(220,220,220) が
   白背景(255)に近くなるため、SSIM 測定で「壁が見えない」判定になる可能性
2. **改善案**: `v_max` を画像ごとに調整する `adaptive_wall_color()` の導入
3. **閾値付き切り替え**: 平均輝度 < 0.7 のときのみグレースケール壁を適用する等

---

## 5. 残課題

| 課題 | 優先度 | 担当 | 参考情報 |
|---|---|---|---|
| グレースケール壁の circle SSIM 低下修正 | 高 | A2 | CR-4参照 |
| セルサイズ可変化（Xu-Kaplan S=W/(1-G)） | 中 | — | design_variable_cell_size.md |
| SSIM ≥ 0.70 の達成 | 高 | — | 現状最良: gradient grid=5 で 0.6062 |
| `test_time_decay_features.py` 13件 FAIL | 中 | keirin担当 | top3_rate列欠如（keirin既存問題） |

---

## 6. 全体評価

**maze-artisan Phase3 達成状況**: 🟢 目標クリア

- ✅ 400x600高解像度が1.5秒（目標60秒）
- ✅ 3本柱（可変壁厚・ループ密度・画像適応ルーティング）全実装
- ✅ MASTERPIECE_PRESET で一発呼び出し対応
- ✅ 442テスト PASS
- ⚠️ SSIM目標0.70未達（現状最良0.6062）
- ⚠️ グレースケール壁は circle 画像でSSIM低下（要調査）
