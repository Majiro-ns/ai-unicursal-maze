# Masterpiece V6: Path-First No-Shortcut Spanning Tree — 再開用ドキュメント

> 最終更新: 2026-03-15
> コミット: `b02ffd5` (feat(masterpiece): V6 no-shortcut spanning tree)
> テスト: 533 pass

---

## 1. ゴール（殿承認済み）

> 3m離れて見る → 解経路の軌跡が元画像に見える
> 近づいて見る → 迷路として解ける
> 解いてみる → 解いた軌跡が元画像を描いている

## 2. 試行錯誤の全履歴

### V1（既存パイプライン）
- **方式**: 壁構築(Kruskal) → 経路探索(Dijkstra)
- **結果**: SSIM=0.32。経路は壁構造の「おまけ」で形状制御不可
- **死因**: 壁が先、経路が後。経路の形状を制御する自由度がない

### V2（Greedy Warnsdorff Walk）— 殿が最も評価
- **方式**: Greedy walk で暗部セルを巡回 → 壁を後付け
- **突破点**: 経路が暗部を面的に塗りつぶし、遠目で画像が見える（SSIM=0.40）
- **問題**: 経路が再訪問あり → 一意解でない → 迷路として解けない
- **殿の評価**: 「test3_v2_greedy.png が一番殿のイメージには近かった」
- **画像**: `data/output/test3_v2_greedy.png`

### V3（Trunk + Branch アーキテクチャ）
- **方式**: 短い幹(Dijkstra) + 暗部を dead-end branch で充填
- **結果**: 幹が397歩の対角線。画像再現度が急落
- **殿の反応**: 「なんか明らかに元画像の再現度が急落しているが、何をしている？」
- **死因**: 解経路が短い幹だけ → 画像をなぞらない。Branch は解の一部でない

### V4（V2 Greedy + Spanning Tree）
- **方式**: V2 の greedy walk を復活 + UnionFind で cycle-free edges + Kruskal で spanning tree
- **結果**: Spanning tree = True, BFS解 = 927歩 vs greedy walk = 8823歩
- **死因**: BFS は spanning tree の最短経路を使う → greedy walk のルートとは別の対角線ショートカットを通る。**解いても画像が見えない**

### V5（Pure Path — 濃淡なし描画）
- **方式**: V4 の構造を均一壁・均一太さで描画（殿指示「濃淡を忘れろ」）
- **結果**: BFS 927歩の赤線が対角線を描く。画像と無関係
- **意義**: BFS解 ≠ 設計経路 の問題を視覚的に確定。V6 の設計動機

### V6（No-Shortcut Spanning Tree）— 現行版 ✅
- **方式**: Greedy walk（再訪問なし） + Loop erasure + Anti-shortcut rule
- **突破点**: **BFS解 = 設計経路 = 300歩（100%一致）**
- **Spanning tree**: True（一意解保証）
- **暗部カバー率**: 57%（145/253）— 改善余地あり
- **画像**: `data/output/test3_v6b_path_only.png`, `test3_v6b_solved.png`（セッション内生成、永続化要確認）

## 3. V6 のアーキテクチャ詳細

### パイプライン
```
入力画像 → preprocess → build_cell_grid(H2可変)
    → detect_edge_map(K1) → extract_edge_waypoints
    → classify_cells (DARK/MID/BRIGHT)
    → find_dark_blobs (BFS flood-fill)
    → find_entrance_exit_path_first (最暗境界セル)
    → design_masterpiece_path ★ V6コア
        1. _greedy_dark_walk: Warnsdorff walk（再訪問なし）
        2. _bfs_unvisited_to_target: exit到達
        3. Loop erasure: 万一の再訪問をカット
        4. path_edges 構築
    → build_walls_around_path ★ Anti-shortcut
        1. path_edges を UnionFind で先に接続
        2. Kruskal だが「解経路セル同士の辺」を禁止
        3. 残りの孤立セルを最終接続
    → maze_to_svg / maze_to_png (G1 per-segment variable width)
```

### Anti-Shortcut Rule（V6の核心）
```
build_walls_around_path で Kruskal 辺を追加する際:
  if c1 in solution_cells and c2 in solution_cells:
      continue  # この辺を追加しない

→ 非解経路セルは dead-end branch として接続される
→ 解経路セル同士の新しい辺は作られない
→ BFS は解経路を迂回できない
→ BFS解 = 設計経路
```

### Greedy Walk（再訪問なし版）
```python
_greedy_dark_walk():
    1. 暗部隣接セルがあれば → Warnsdorff（未訪問暗部隣接が少ない方を優先）
    2. 暗部なし・非暗部未訪問あれば → 最寄り暗部セルに向かう方を選択
    3. 未訪問隣接が全くない → _bfs_unvisited_to_dark（未訪問セルのみで暗部到達）
    4. BFS も失敗 → 残りの暗部は到達不能、walk 終了
```

## 4. ファイル構成

| ファイル | 行数 | 役割 |
|---------|------|------|
| `backend/core/density/path_designer.py` | 1123 | V6コア。8+関数 |
| `backend/core/density/__init__.py` | ~305 | `MASTERPIECE_V2_PRESET` + `use_path_first` 分岐 |
| `backend/core/density/exporter.py` | ~600 | G1 per-segment variable width (SVG/PNG) |
| `backend/core/density/edge_enhancer.py` | ~175 | K1 edge detection + `extract_edge_waypoints` |
| `tests/test_path_first_masterpiece.py` | 566 | 33テスト |

### path_designer.py 関数一覧

| 関数 | 用途 | V6で使用 |
|------|------|---------|
| `classify_cells` | DARK/MID/BRIGHT 分類 | ✅ |
| `find_dark_blobs` | BFS flood-fill 連結成分 | ✅ |
| `find_entrance_exit_path_first` | 最暗境界セルで入口/出口 | ✅ |
| `order_blobs_for_path` | Nearest-neighbor blob 訪問順 | ✅（__init__.py から呼出） |
| `serpentine_fill_blob` | Boustrophedon 蛇行充填 | ❌（V6未使用、将来候補） |
| `connect_through_bright` | Dijkstra 明部経由接続 | ❌（V6未使用） |
| `_greedy_dark_walk` | Warnsdorff walk（再訪問なし） | ✅ コア |
| `_bfs_unvisited_to_dark` | 未訪問セルのみで暗部到達 | ✅ |
| `_bfs_unvisited_to_target` | 未訪問セルのみで特定セル到達 | ✅ |
| `_walk_dark_blob` | blob 内 greedy walk | ❌（blob-by-blob 試行で使用、現行未使用） |
| `_min_dist_to_dark` | 最寄り暗部 Manhattan 距離 | ✅ |
| `design_masterpiece_path` | V6 メインロジック | ✅ |
| `build_walls_around_path` | Anti-shortcut Kruskal | ✅ |

## 5. 現在の問題と次の一手

### 問題: 暗部カバー率 57%（145/253）
- 原因: greedy walk が non-dark セルを消費して暗部への到達経路を塞ぐ
- walk が通った領域が「島」になり、未訪問領域が分断される
- `_bfs_unvisited_to_dark` が None を返して walk 終了

### 殿承認済みの方針: 「割り切り」戦略（選択肢3）
- **解経路で暗部の70-80%をカバー** → 解いた軌跡で画像が見える
- **残り20-30%は dead-end branch の壁密度で補完** → 解かなくても遠目で画像が見える
- V2 greedy が良く見えた理由: 経路だけでなく壁密度の濃淡も画像認識に寄与

### 具体的な実装タスク（未着手）

#### A. 暗部カバー率 57% → 75%+
1. walk の non-dark 消費を減らす: BFS bridge で暗部間を最短接続（non-dark セルの消費最小化）
2. `dark_threshold` を 0.3 → 0.4 に上げて暗部判定を広げる（要実験）
3. blob-by-blob walk の再試行（前回 31% だったが、bridge 最適化で改善可能）

#### B. 壁密度で残り暗部を補完
1. `build_walls_around_path` の Kruskal バイアスを強化: 暗部の壁をもっと残す
2. dead-end branch が暗部に集中するように Kruskal の weight 調整
3. 壁の太さ（`thickness_range`）を暗部でさらに太く

#### C. 解経路の視認性改善
1. 解経路の white line が暗部の太い壁に埋もれる問題
2. 解経路セグメントの太さを G1 で輝度連動させる（実装済みだが効果不足）
3. 代替案: 解経路を「壁がない通路」として視認（壁の隙間＝白い道）

## 6. テスト画像

- **入力**: `data/input/test3.jpg`（漫画キャラ、166x300、4x upscale で 664x1200）
- **出力画像**:
  - `data/output/test3_v2_greedy.png` — V2 greedy（殿が最も評価）
  - `data/output/test3_v2_path_only.png` — V2 経路のみ
  - `data/output/test3_v4_masterpiece.png` — V4 density rendering
  - `data/output/test3_v4_masterpiece_solved.png` — V4 解表示
  - V5/V6 画像はセッション内生成（再生成が必要な場合あり）

## 7. 再開時の手順

```bash
cd /mnt/c/Users/owner/Desktop/llama3_wallthinker/ai-unicursal-maze

# 1. テスト確認
python3 -m pytest tests/ --tb=line  # 533 pass を確認

# 2. V6 画像再生成（セッション画像が消えていた場合）
python3 -c "
import sys; sys.path.insert(0, '.')
from PIL import Image
from backend.core.density import generate_density_maze, MASTERPIECE_V2_PRESET
img = Image.open('data/input/test3.jpg')
preset = {**MASTERPIECE_V2_PRESET, 'show_solution': False}
result = generate_density_maze(img, width=800, height=1200, **preset)
with open('data/output/test3_v6_masterpiece.png', 'wb') as f:
    f.write(result.png_bytes)
print(f'Grid: {result.grid_rows}x{result.grid_cols}, Solution: {len(result.solution_path)} steps')
"

# 3. 暗部カバー率の確認
python3 -c "
import sys; sys.path.insert(0, '.')
from PIL import Image
import numpy as np
from backend.core.density.preprocess import preprocess_image
from backend.core.density.grid_builder import build_cell_grid
from backend.core.density.path_designer import (
    classify_cells, find_dark_blobs, find_entrance_exit_path_first,
    order_blobs_for_path, design_masterpiece_path, build_walls_around_path,
)
from backend.core.density.edge_enhancer import detect_edge_map, extract_edge_waypoints
from collections import deque

img = Image.open('data/input/test3.jpg')
gray = preprocess_image(img, max_side=512)
grid_rows = min(50, max(gray.shape[0]//4, 1))
grid_cols = min(50, max(gray.shape[1]//4, 1))
grid = build_cell_grid(gray, grid_rows, grid_cols, density_factor=1.0, variable_cell_size=True)
edge_map = detect_edge_map(gray, grid_rows, grid_cols)
edge_wp = extract_edge_waypoints(edge_map, grid_cols, threshold=0.3)
cell_classes = classify_cells(grid.luminance, dark_thresh=0.3, bright_thresh=0.7)
blobs = find_dark_blobs(cell_classes, grid)
entrance, exit_cell = find_entrance_exit_path_first(grid, cell_classes)
ordered_blobs = order_blobs_for_path(blobs, entrance, exit_cell, grid)
solution_path, path_edges = design_masterpiece_path(
    grid, cell_classes, ordered_blobs, entrance, exit_cell, edge_wp)
adj = build_walls_around_path(grid, path_edges, cell_classes, solution_cells=set(solution_path))
n = grid.num_cells
total_edges = sum(len(v) for v in adj.values()) // 2
flat = cell_classes.flatten()
dark_tot = sum(1 for c in range(n) if flat[c]==0)
sol_dark = sum(1 for c in set(solution_path) if flat[c]==0)

# BFS check
vis={entrance}; prev={entrance:-1}; q=deque([entrance])
while q:
    u=q.popleft()
    if u==exit_cell: break
    for v in adj[u]:
        if v not in vis: vis.add(v); prev[v]=u; q.append(v)
p=[]; c=exit_cell
while c!=-1: p.append(c); c=prev[c]
p.reverse()
bfs_len = len(p)

print(f'Grid: {grid_rows}x{grid_cols} = {n} cells')
print(f'Spanning tree: {total_edges == n-1} ({total_edges} edges)')
print(f'BFS solution: {bfs_len} == designed: {len(solution_path)} → match={bfs_len==len(solution_path)}')
print(f'Dark coverage: {sol_dark}/{dark_tot} ({100*sol_dark/max(dark_tot,1):.0f}%)')
"
```

## 8. 試行錯誤から得た教訓

1. **Spanning tree の一意解は常に短い** — entrance→exit の唯一経路が長くなるには、木の構造自体が迂回を強制する必要がある
2. **Anti-shortcut が鍵** — 解経路セル同士の Kruskal 辺を禁止すれば、BFS は設計経路を通るしかない
3. **Loop erasure は暗部を失う** — 再訪問部分を消すと暗部カバーが落ちる。最初から再訪問しない walk が必要
4. **壁密度は経路と独立した情報チャネル** — 解経路だけで画像を100%再現する必要はない。壁の太さ・密度でも画像を表現できる（V2 が好評だった理由）
5. **「濃淡を忘れろ」は有効な診断法** — 純粋に経路構造だけで見ることで、問題が明確になった
