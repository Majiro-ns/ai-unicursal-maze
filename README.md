# AI Unicursal Maze Generator

![tests](https://img.shields.io/badge/tests-387%20passed-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

画像をアップロードして、一筆書き迷路風の SVG/PNG を生成するローカル Web アプリです。入力画像の輪郭・シルエットに沿った一筆パスから迷路を生成します。

- **バックエンド**: FastAPI  
- **フロントエンド**: Streamlit（ローカル利用前提）

---

## maze-artisan との対応

本リポジトリは **maze-artisan** の実装リポジトリです。

| 項目 | 内容 |
|------|------|
| 要件・タスク | maze-artisan（Obsidian）の `01_Requirements.md`, `03_Tasks.md`, `02_Design.md` を参照 |
| 参照先例 | `C:\Users\owner\Documents\Obsidian Vault\10_Projects\maze-artisan`（ローカル Obsidian Vault 内） |
| 現状 | **V1.1** 完了（Stage A〜D パイプライン、解数・難易度返却）。**V2** 中期タスク（T-5〜T-7）進行中。**Phase DM-3 達成 ✅**（387 テスト PASS・Masterpiece 品質評価実装済み） |
| Gap 分析 | 本リポジトリ `docs/GAP_ANALYSIS.md` に要件との差分・次タスクを記載 |

---

## セットアップ

### 環境

- **Python**: 3.10 以降を推奨
- **依存**: プロジェクトルートで以下を実行

```bash
pip install -r requirements.txt
```

既存の仮想環境がある場合は、先に有効化してから実行してください。

### オプション依存

- **mediapipe**（`requirements.txt` に含む）: 顔ランドマーク・顔マスク用。未導入でも動作しますが、顔らしさ強調が弱まります。
- 導入例: `pip install mediapipe==0.10.14`

---

## 起動手順

### 方法 1: 手動で 2 プロセス起動（推奨）

1. **FastAPI バックエンド**（プロジェクトルートで実行）

   ```bash
   uvicorn backend.app:app --reload
   ```

   `http://localhost:8000` で API が待ち受けます。

2. **Streamlit フロントエンド**（別ターミナルで、同じくプロジェクトルートから）

   ```bash
   streamlit run frontend/ui.py
   ```

   ブラウザが開き、AI 一筆迷路ジェネレーターの UI が表示されます。

### 方法 2: 二段階で起動（いちばん確実・推奨）

1. **`run_backend_only.bat`** をダブルクリック  
   → 「Uvicorn running on http://127.0.0.1:8001」と出るまで待つ。**このウィンドウは閉じない。**

2. **別のウィンドウで `run_frontend_only.bat`** をダブルクリック  
   → ブラウザで http://localhost:8501 が開く。

（バックエンド・フロントとも **port 8001** を使用するため、8000 が他で使われていてもそのまま使えます。）

### 方法 3: 一括起動（Windows）

プロジェクトルートで `run_app.bat` を実行すると、バックエンドとフロントエンドを別ウィンドウでまとめて起動します（`--reload` は付けていません。`.venv` があれば自動で有効化）。

```batch
run_app.bat
```

### 接続エラーが出る場合

「接続できませんでした」「Max retries exceeded」と出る場合は、**バックエンドが起動していない**可能性が高いです。  
先に **`run_backend_only.bat`** を実行し、「Uvicorn running on http://127.0.0.1:8001」と表示されたら、その窓を閉じずに **`run_frontend_only.bat`** でフロントを開いてください。  
（通常はバックエンド・フロントとも port **8001** を使用します。8000 が他で使われていても問題ありません。）

### その他のスクリプト

- **RUN.bat** / **RUN_EXPERIMENTS.bat**: バッチ実験用（`scripts/` 内の p11 系スクリプト実行）。Web アプリの起動には使いません。

---

## 使い方

1. Streamlit の UI で画像ファイル（PNG/JPEG）をアップロードする。
2. サイドバーで **出力幅・高さ**、**線の太さ**、**線画モード**、**迷路の粗さ**（スパー長・ノイズ除去閾値）、**maze_weight**（顔らしさ↔迷路性）などを調整する（任意）。
3. 処理ステップのボタン（①線画 → ②一筆書き → ③迷路 → ④ダミー迷路）のいずれかを押す。
4. 生成結果が表示され、SVG/PNG をダウンロードできる。

画像サイズは最大 10MB まで（超過時は HTTP 413）。

---

## Masterpiece 機能（Phase DM-3 達成 ✅）

高品質な密度迷路を生成する「Masterpiece」モードを実装。画像誘導ルーティング・可変壁厚・ループ密度制御の **3本柱**を同時有効化することで、視覚的に豊かな迷路を生成します（387 テスト全 PASS）。

### 黄金設定（Golden Setting）

| パラメータ | 値 | 説明 |
|---|---|---|
| `grid_size` | **8** | 小グリッドで高 SSIM を実現 |
| `thickness_range` | **1.5** | 可変壁厚（暗部: 太く / 明部: 細く） |
| `extra_removal_rate` | **0.5** | ループ密度制御（暗部にループ追加） |
| `use_image_guided` | **True** | 画像誘導ルーティング（暗→明の最短経路） |

### SSIM 実績（cmd_358k_a2 評価）

| 入力画像 | SSIM | 評価 |
|---|---|---|
| グラデーション | **0.5566** | good ✅ |
| サークル | **0.5072** | good ✅ |
| Edge-SSIM（グラデーション） | **0.8092** | 優秀 ✅ |

- 旧ベースライン（`grid_size=30`）: gradient SSIM = 0.4476 [fair]
- `excellent`（≥ 0.70）は二値壁レンダリングの構造的限界により未達（上限 ≈ 0.63）

---

## CI（GitHub Actions）

`push` / `pull_request` で `.github/workflows/ci.yml` が実行されます。

- **内容**: 依存インストールと `pytest tests/` のみ。外部APIキー・シークレットは一切使用しません（**費用は発生しません**）。
- 将来有料APIを組み込む場合も、CI にはシークレットを渡さない設計にしています。

---

## 今後の拡張

- **V2**: スケルトン安定化（T-5）、オイラー路エントリ設計（T-6）、UI オプション整理（T-7）。
- **顔らしさ向上**: 一筆パス美観（T-11）、顔・髪再現性強化（T-12）。要件・タスクは maze-artisan の 03_Tasks を参照。
- 現状の差分と次タスクは `docs/GAP_ANALYSIS.md` を参照。

---

## 顔らしさ向上 TODO（メモ）

- 顔マスク/ランドマークを try/except でフォールバックしているため、失敗時は顔形状の反映が弱い。
- Canny 依存の線画は髪・背景ノイズの影響を受けやすい。顔帯域の前処理・しきい値制御の強化が有効。
- グラフ重み（ランドマーク優先）とパススコアの改善、スケルトン品質維持、回帰テストの追加が今後の対応候補。
- 詳細は maze-artisan の要件と `docs/GAP_ANALYSIS.md` を参照。
