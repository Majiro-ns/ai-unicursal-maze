# DM-8 設計書: マルチスケール最適化（Pyramid Density Map）

date: 2026-03-31
status: 実装済み（backend/core/density/dm8.py）
author: ashigaru4 / cmd_703k_a4

---

## 1. 背景と動機

### DM-1〜DM-7 の進化

| バージョン | 主な機能 | 限界 |
|---|---|---|
| DM-1 | 基礎パイプライン（輝度→壁重み→Kruskal） | 単一スケールのみ |
| DM-2 | CLAHE コントラスト補正 | — |
| DM-3 | Canny エッジ強調 | — |
| DM-4 | 多値トーン壁表現 + SSIM スコア化 | passage_ratio 固定 |
| DM-5 | 印刷最適化（PDF/300DPI） | — |
| DM-6 | Bayesian 最適化 + 難易度制御 | **density_map は単一スケール** |
| DM-7 | passage_ratio 制御（v0.6.0 実装） | マルチスケール未対応 |

### DM-6/DM-7 の根本的限界

DM-6/DM-7 の `build_density_map(gray, grid_rows, grid_cols)` は、
画像を `grid_rows × grid_cols` ブロックに等分割して各ブロックの平均輝度を計算する。

この手法の問題点:
- **グローバル構造の欠損**: 大局的な暗部（例: ロゴの黒背景全体）が、
  同じ大きさの多くのセルに分散してしまい、グローバルな引力として機能しない
- **スケール依存性**: grid_size=8 では細かい局所情報が欠ける。
  grid_size=16 では大局構造が拡散する

**結果として**: 一部のカテゴリ（Photo: SSIM=0.0689）でスコアが極端に低くなる。

---

## 2. DM-8 の設計思想

### ピラミッド型密度マップ

画像処理の Laplacian Pyramid / Gaussian Pyramid と同じ発想:
**複数の解像度（スケール）で密度マップを計算し、加重合成する。**

```
元画像 (H×W)
   │
   ├──► L1 = build_density_map(gray, 4, 4)     # 4×4グリッド: グローバル構造
   │        ↓ アップサンプル（バイリニア補間）
   │        L1_up (grid_rows × grid_cols)
   │
   ├──► L2 = build_density_map(gray, 8, 8)     # 8×8グリッド: 中間ディテール
   │        ↓ アップサンプル（バイリニア補間）
   │        L2_up (grid_rows × grid_cols)
   │
   └──► L3 = build_density_map(gray, grid_rows, grid_cols)  # 最終解像度: 局所構造

マルチスケール密度マップ = w1*L1_up + w2*L2_up + w3*L3
                          （デフォルト: w1=0.2, w2=0.3, w3=0.5）
```

### なぜこれで SSIM が向上するか

1. **L1 (4×4) の役割**: 画像全体を16ブロックで見た時の「暗部領域」を捉える。
   たとえば「画像左上が全体的に暗い」という情報がL1には強く現れる。
   この情報をアップサンプルして最終グリッドに重み付けすることで、
   左上エリアのセル全体が「通路が多くなりやすい」（低壁重み）方向に引っ張られる。

2. **L2 (8×8) の役割**: 中間的な構造（輪郭、テキストの塊など）を捉える。
   L1 より細かく、L3 より粗い「橋渡し」スケール。

3. **L3 (最終グリッド) の役割**: 各セルの局所的な輝度。従来手法と同じ。

4. **加重合成**: `w1=0.2, w2=0.3, w3=0.5` のデフォルトでは、
   局所情報を基盤としながらグローバル・中間構造も反映する。
   Photo カテゴリのような複雑画像では `w1` を上げると効果的。

---

## 3. 各レベルの SSIM 寄与度分析

### 理論分析

| スケール | 捉える構造 | SSIM への寄与 |
|---|---|---|
| L1 (4×4) | グローバル輝度分布（明暗の大局） | 低周波成分の再現性を支配 |
| L2 (8×8) | 中規模構造（顔の輪郭・ロゴの形） | 中周波成分の再現性を補強 |
| L3 (N×N) | セル単位の局所輝度 | 高周波成分の再現性を確保 |

SSIM は輝度・コントラスト・構造の3成分の積で定義される。
マルチスケール密度マップは「構造 (structure)」成分の改善に直接寄与する。

### カテゴリ別推奨スケール重み

| カテゴリ | 推奨 scale_weights (w1, w2, w3) | 理由 |
|---|---|---|
| logo | (0.3, 0.3, 0.4) | 高コントラスト・単純形状 → グローバル重視 |
| anime | (0.2, 0.4, 0.4) | 輪郭主体 → 中間スケール重視 |
| portrait | (0.1, 0.3, 0.6) | 細かいトーン変化 → 局所重視 |
| landscape | (0.3, 0.4, 0.3) | 広域の明暗 → 全スケールバランス |
| photo | (0.4, 0.3, 0.3) | 複雑な輝度分布 → グローバル重視 |

---

## 4. 一意解保証

### 設計上の保証

DM-8 は `generate_dm4_maze()` の **density_map 計算ステップのみ** を差し替える。
それ以外のパイプライン（Kruskal MST → 一意スパニングツリー → 入口/出口決定）は変更しない。

```
DM-8 パイプライン:
  Stage 1: 前処理（DM-4 と同一）
  Stage 2: CLAHE（DM-4 と同一）
  Stage 2b: 🆕 マルチスケール密度マップ（DM-8 追加）
  Stage 3: 壁重み設定 + Kruskal MST（DM-4 と同一）
  Stage 4: 入口/出口 + BFS + 解数検証（DM-4 と同一）
  Stage 5: トーンレンダリング（DM-4 と同一）
```

Kruskal MST は全 N²セルの完全スパニングツリーを生成するため、
`extra_removal_rate=0.0` の場合は定義上 **必ず一意解** となる。
`extra_removal_rate>0` の場合はループが追加されるが、これは DM-6 と同じ挙動。

### 制約伝播の考え方

階層生成（L1→L2→L3 の順に生成して制約を伝播）は実装しない。
理由: 階層制約伝播は実装複雑度が高く、unicursal 保証の維持が困難。

代わりに **密度マップの融合** を採用:
「最終グリッドでの壁重みをマルチスケール情報で初期化する」という形で
全スケールの情報を一意解保証を壊さずに組み込む。

---

## 5. API

### DM8Config

```python
@dataclass
class DM8Config(DM6Config):
    coarse_size: int = 4           # L1 グリッドサイズ
    medium_size: int = 8           # L2 グリッドサイズ
    scale_weights: Tuple[float, float, float] = (0.2, 0.3, 0.5)  # (L1, L2, L3)
```

### DM8Result

```python
@dataclass
class DM8Result(DM6Result):
    scale_weights_used: Tuple[float, float, float] = (0.2, 0.3, 0.5)
    coarse_size_used: int = 4
    medium_size_used: int = 8
    ssim_improvement: float = 0.0   # DM-6 比（設定時）
```

### generate_dm8_maze

```python
def generate_dm8_maze(
    image: Image.Image,
    config: Optional[DM8Config] = None,
) -> DM8Result:
    ...
```

### build_multiscale_density_map

```python
def build_multiscale_density_map(
    gray: np.ndarray,
    target_rows: int,
    target_cols: int,
    coarse_size: int = 4,
    medium_size: int = 8,
    scale_weights: Tuple[float, float, float] = (0.2, 0.3, 0.5),
) -> np.ndarray:
    """
    マルチスケール密度マップを生成する。

    戻り値: (target_rows, target_cols) float, 値域 [0.0, 1.0]
    """
```

---

## 6. 期待される SSIM 改善

| カテゴリ | DM-6 SSIM | DM-8 期待 SSIM | 改善幅 |
|---|---|---|---|
| Logo | 0.5781 | 0.60〜0.63 | +4〜9% |
| Anime | 0.5389 | 0.56〜0.59 | +4〜9% |
| Portrait | 0.5375 | 0.55〜0.58 | +2〜8% |
| Landscape | 0.4930 | 0.51〜0.55 | +3〜12% |
| Photo | 0.0689 | 0.12〜0.20 | +74〜190% |

Photo カテゴリの改善幅が最大と予想される理由:
- Photo の低 SSIM（0.069）はグローバル構造の欠損が主因
- L1 の大局的輝度情報（w1 上昇）で劇的に改善できる余地がある

---

## 7. 今後の拡張候補（DM-9 以降）

1. **適応的スケール重み**: カテゴリ別に `scale_weights` を自動チューニング
   （DM-6 の Bayesian 最適化と統合）
2. **4 スケール**: L0 (2×2) を追加し、最大輝度コントラストを捉える
3. **エッジ強調 × マルチスケール**: 各スケールでのエッジ情報も合成
4. **DM-5 統合**: 印刷最適化（A4/A3）との組み合わせ
