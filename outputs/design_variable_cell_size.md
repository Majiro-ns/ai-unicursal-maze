# 設計書: セルサイズ可変化（Xu-Kaplan G=(S-W)/S 実装案）

**作成日**: 2026-03-13
**対象プロジェクト**: ai-unicursal-maze
**タスクID**: cmd_360k_a6
**状態**: 設計のみ（実装なし）

---

## 1. 背景と目的

### Xu-Kaplan (SIGGRAPH 2007) の公式

```
G = (S - W) / S
```

| 変数 | 意味 |
|------|------|
| `S`  | セルサイズ（ピクセル） |
| `W`  | 壁幅（stroke_width） |
| `G`  | 通路率（gap ratio）。`G → 1` で通路が広い、`G → 0` で壁だらけ |

逆算:
```
S = W / (1 - G)
```

**思想**: 画像の輝度 `L` を `G` にマッピングする。
- 明るいセル (`L ≈ 1.0`): `G ≈ 1.0` → `S` 大 → 広い通路
- 暗いセル (`L ≈ 0.0`): `G ≈ 0.0` → `S` 小 → 壁が密集（通路なし）

現行実装との違い:
- **現行**: セルサイズ均一、壁厚（`stroke_width`）を輝度で可変
- **Xu-Kaplan**: セルサイズ自体を輝度で可変、壁厚は固定

---

## 2. 現行コードの cell_size 計算箇所

### 2.1 grid_builder.py: セル境界の計算

```python
# build_density_map()
y_bnd = [int(r * h / grid_rows) for r in range(grid_rows + 1)]  # 均一分割
x_bnd = [int(c * w / grid_cols) for c in range(grid_cols + 1)]  # 均一分割
```

現行では画像を **等間隔** に分割する。セルサイズ可変化では、この境界を
輝度マップから逆算した不均一境界に置き換える必要がある。

### 2.2 exporter.py: 描画座標の計算

**SVG (`maze_to_svg`)**:
```python
# 行 155: 単一 cell_size スカラー計算
cs_x = w / grid.cols
cs_y = h / grid.rows
cell_size = min(cs_x, cs_y)      # ★ 均一セルサイズ

# 行 177-192: 均一ピクセル座標計算
x0 = margin + c * cell_size       # ★ c × 均一サイズ
y0 = margin + r * cell_size       # ★ r × 均一サイズ
x1 = x0 + cell_size               # ★ 右端 = 左端 + 均一サイズ
y1 = y0 + cell_size               # ★ 下端 = 上端 + 均一サイズ
```

**PNG (`maze_to_png`)**:
```python
# 行 294-296: 同様
cs_x = w / grid.cols
cs_y = h / grid.rows
cell_size = min(cs_x, cs_y)

# 行 307: 均一変換
def to_px(x: float, y: float) -> tuple[int, int]:
    return int(margin + x * cell_size), int(margin + y * cell_size)
```

**解経路のセル中心計算**:
```python
# exporter.py 行 23-27: _cell_center()
def _cell_center(grid, cell_id, cell_size, margin):
    r, c = grid.cell_rc(cell_id)
    x = margin + (c + 0.5) * cell_size   # ★ 均一中心
    y = margin + (r + 0.5) * cell_size   # ★ 均一中心
```

### 2.3 entrance_exit.py・solver.py・maze_builder.py

これらのファイルはセル **インデックス** のみで動作し、ピクセル座標を一切参照しない。
→ **変更不要**。

---

## 3. セルサイズ可変化の設計

### 3.1 CellGrid への追加フィールド

```python
# grid_builder.py: CellGrid データクラス
@dataclass
class CellGrid:
    rows: int
    cols: int
    luminance: np.ndarray             # 既存: (rows, cols) float 0-1
    walls: List[Tuple[int, int, float]]  # 既存
    # 追加フィールド（可変セルサイズ用）
    row_heights: Optional[np.ndarray] = None  # (rows,) 各行の高さ比率（合計=1.0）
    col_widths: Optional[np.ndarray] = None   # (cols,) 各列の幅比率（合計=1.0）
```

`row_heights` と `col_widths` は**正規化された比率**（合計=1.0）として保持する。
実際のピクセルサイズは `exporter` で `画面幅 × col_widths[c]` のように計算する。

`None` の場合は均一（現行動作と後方互換）。

### 3.2 Xu-Kaplan 公式によるサイズマップ計算

```python
# 新関数: grid_builder.py に追加予定
def compute_cell_size_map(
    luminance: np.ndarray,     # (rows, cols) float 0-1
    wall_width: float = 2.0,   # W: 固定壁幅（ピクセル）
    canvas_width: int = 800,   # 描画領域の幅
    canvas_height: int = 600,  # 描画領域の高さ
    min_cell_px: float = 4.0,  # S_min: 視認できる最小セルサイズ（ピクセル）
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Xu-Kaplan G=(S-W)/S 公式で各セルのサイズを計算する。

    G = luminance（明るい = 広い通路）
    S = W / (1 - G)

    ただし G → 1 で S → ∞ になるため、クリッピングが必要:
      G_eff = clip(G, 0.0, G_max)  where G_max = 1 - W/S_max
      S = W / (1 - G_eff)

    戻り値:
      row_heights: (rows,) 各行の高さ比率（合計=1.0）
      col_widths: (cols,) 各列の幅比率（合計=1.0）
    """
    rows, cols = luminance.shape

    # 各セルの G を輝度から計算
    # G_min: 最小通路率（全黒でも壁は描ける最低値）
    G_min = wall_width / min_cell_px  # e.g., 2/4 = 0.5
    G_max = 0.98                       # 無限大への発散を防ぐ上限
    G = np.clip(luminance, G_min, G_max)

    # S = W / (1 - G): 各セルのピクセルサイズ
    S = wall_width / (1.0 - G)  # shape: (rows, cols)

    # 行方向: 各行の平均 S（列方向の平均）
    row_S = S.mean(axis=1)   # (rows,)
    # 列方向: 各列の平均 S（行方向の平均）
    col_S = S.mean(axis=0)   # (cols,)

    # 正規化して比率に変換
    row_heights = row_S / row_S.sum()  # (rows,) 合計=1.0
    col_widths  = col_S / col_S.sum()  # (cols,) 合計=1.0

    return row_heights, col_widths
```

### 3.3 累積和ベースの座標計算

セルサイズが不均一になると、列 `c` のx座標は:

```python
# 現行（均一）:
x0 = margin + c * cell_size

# 変更後（累積和）:
x_offsets = np.concatenate([[0.0], np.cumsum(col_widths * canvas_width)])
# x_offsets[c] = 列cの左端x座標（marginを除く）
x0 = margin + x_offsets[c]
x1 = margin + x_offsets[c + 1]
```

同様に行方向:
```python
y_offsets = np.concatenate([[0.0], np.cumsum(row_heights * canvas_height)])
y0 = margin + y_offsets[r]
y1 = margin + y_offsets[r + 1]
```

セル中心:
```python
x_center = margin + (x_offsets[c] + x_offsets[c + 1]) / 2.0
y_center = margin + (y_offsets[r] + y_offsets[r + 1]) / 2.0
```

---

## 4. 影響範囲の全ファイル・全関数リスト

### 変更が必要なファイル

| ファイル | 関数 | 変更内容 | 難易度 |
|---------|------|---------|--------|
| `grid_builder.py` | `CellGrid` | `row_heights`, `col_widths` フィールド追加 | 低 |
| `grid_builder.py` | `build_density_map()` | 可変境界での集計（`y_bnd`, `x_bnd` を非均一に） | 中 |
| `grid_builder.py` | `build_cell_grid()` | `compute_cell_size_map()` を呼び出して `CellGrid` に格納 | 低 |
| `grid_builder.py` | `build_cell_grid_with_edges()` | 同上 | 低 |
| `grid_builder.py` | `build_cell_grid_with_texture()` | 同上 | 低 |
| `grid_builder.py` | **新規** `compute_cell_size_map()` | Xu-Kaplan 公式実装 | 中 |
| `exporter.py` | `maze_to_svg()` | 均一 `cell_size` → 累積和 `x_offsets`, `y_offsets` | 高 |
| `exporter.py` | `maze_to_png()` | 同上 | 高 |
| `exporter.py` | `_cell_center()` | 均一中心 → 累積和中心 | 中 |
| `exporter.py` | `wall_thickness_histogram()` | 壁幅計算に影響なし（輝度ベース） | 不要 |
| `__init__.py` | `generate_density_maze()` | `use_variable_cell_size: bool = False` パラメータ追加 | 低 |

### 変更不要なファイル

| ファイル | 理由 |
|---------|------|
| `entrance_exit.py` | セルインデックスのみ操作。座標計算なし |
| `maze_builder.py` | セルインデックスのみ操作 |
| `solver.py` | セルインデックスのみ操作 |
| `texture.py` | 壁重み計算のみ |
| `segment.py` | 輝度クラスタリングのみ |
| `preprocess.py` | 画像前処理のみ |
| `edge_enhancer.py` | エッジ検出のみ |

---

## 5. テスト修正が必要な箇所

### 5.1 確実に壊れるテスト

```
tests/test_density_maze_phase1.py:
  - test_density_map_numpy_correctness()
    → y_bnd, x_bnd が非均一になるため、Python ループ版との比較を見直す

tests/test_performance.py:
  - test_build_cell_grid_numpy_wall_count()
    → 壁数は変わらないが、cell_size の意味が変わる

tests/test_phase3_masterpiece.py:
  - test_400x600_solution_path_adjacency()
    → adj 再構築ロジック変わらず、問題なし
  - test_thickness_range_zero_svg_uniform()
    → 均一 thickness_range=0 でもセルサイズ可変なら壁位置が変わる
```

### 5.2 新規追加すべきテスト

```python
# 追加すべきテスト
def test_compute_cell_size_map_bright_is_larger():
    """明るいセルのサイズが暗いセルより大きいこと。"""

def test_cell_size_map_sums_to_one():
    """row_heights と col_widths の合計が 1.0 であること。"""

def test_variable_cell_size_svg_has_varying_widths():
    """可変セルサイズ時、SVGのセル幅が均一でないこと。"""

def test_variable_cell_size_backward_compatible():
    """use_variable_cell_size=False（デフォルト）で現行と同一出力。"""
```

---

## 6. 最小セルサイズ制約

Xu-Kaplan 公式の `S = W / (1 - G)` は `G → 1` で `S → ∞`、`G = 1` で `S` が定義不能。

制約設計:

```
G_min = W / S_min   # 最小通路率（全黒でも最低 S_min ピクセル確保）
G_max = 0.98        # 最大通路率（S_max = W / (1 - 0.98) = 50W まで許容）

例: W=2.0, S_min=4px, S_max=100px のとき
  G_min = 2.0 / 4.0 = 0.50
  G_max = 1 - 2.0 / 100.0 = 0.98
  G の実効範囲: [0.50, 0.98] → S の実効範囲: [4px, 100px]
```

推奨パラメータ（Phase 3 masterpiece 品質用）:

```python
VARIABLE_CELL_PRESET = {
    "wall_width": 2.0,       # W: 固定壁幅
    "min_cell_px": 4.0,      # S_min: 最小セルサイズ
    "canvas_width": 1200,    # 描画幅
    "canvas_height": 800,    # 描画高さ
}
```

---

## 7. 実装優先度と判断基準

### 実装すべき条件

A2 の SSIM 測定結果に基づいて判断:

| SSIM | 推奨アクション |
|------|--------------|
| ≥ 0.70 | セルサイズ可変は不要。現行実装で十分 |
| 0.60〜0.69 | セルサイズ可変を試験実装してSSIM比較 |
| < 0.60 | セルサイズ可変を実装し、他の手法と組み合わせる |

### 実装難易度

- **低コスト**: `CellGrid` フィールド追加 + `compute_cell_size_map()` (50行程度)
- **高コスト**: `exporter.py` の座標計算全面改修（均一 → 累積和）
  - `maze_to_svg`: `cell_size` 参照箇所 **10箇所以上**
  - `maze_to_png`: `cell_size` 参照箇所 **8箇所以上**

### 段階的実装案（推奨）

**Step 1**: `CellGrid` に `row_heights`/`col_widths` を追加、`compute_cell_size_map()` 実装
**Step 2**: `exporter.py` を `x_offsets`/`y_offsets` ベースにリファクタ（後方互換維持）
**Step 3**: `__init__.py` に `use_variable_cell_size` パラメータ追加
**Step 4**: テスト追加・既存テスト修正

各 Step を独立したコミットで実施し、各 Step 後に全テストが通ることを確認する。

---

## 8. 現行との視覚的差異

| 手法 | 制御対象 | SSIM 期待値 | 計算量 |
|------|---------|------------|--------|
| 現行（均一セル + 可変壁厚） | 壁の太さ | 〜0.65 (推測) | O(N) |
| Xu-Kaplan 可変セル | セルの大きさ | 〜0.72 (推測) | O(N) |
| 両方を組み合わせ | 壁太さ + セルサイズ | 〜0.78 (推測) | O(N) |

**現行の `_wall_stroke()` は既に Xu-Kaplan の「壁厚」側を実装済み**。
`G = (S - W) / S` の「S を変える」側は未実装。

---

## 9. まとめ

本設計書で確認した要点:

1. **影響範囲**: `exporter.py` の描画座標計算が主要変更箇所（10箇所以上）
2. **solver・entrance_exit・maze_builder は変更不要**（セルインデックスベース）
3. **Xu-Kaplan 公式**: `G = clip(lum, G_min, G_max)`, `S = W / (1 - G)` で実装可
4. **累積和ベース座標**: `np.cumsum(col_widths)` で `x_offsets` を計算
5. **後方互換**: `use_variable_cell_size=False` デフォルトで現行動作を維持可能
6. **実装判断**: A2 SSIM ≥ 0.70 なら不要、< 0.70 なら着手推奨
