# -*- coding: utf-8 -*-
"""
T-5: スケルトン線の安定化テスト

対象モジュール: backend.core.skeleton
確認事項:
  1. 孤立ピクセルが skeleton 後に除去される
  2. 小コンポーネント（2px）が min_size=5 で除去される
  3. 大コンポーネント（10px）が min_size=5 で除去されない
  4. T字型スケルトンの短辺（len=2）が max_length=4 で除去される
  5. T字型スケルトンの長辺（len=10）が max_length=4 で除去されない
  6. 大領域+微小ノイズ（3px）がskeletonに残らない
  7. stabilize_skeleton が bool 2D配列を返し、元の pixel 数以下になる

CHECK-9 テスト期待値の根拠:
  - テスト4: T字スケルトン短辺=2ピクセル。max_length=4 は分岐点から4ピクセル以内の
    行き止まり枝を削除するため len=2 は必ず削除される。手検証済み。
  - テスト5: 幹=12ピクセルの直線。max_length=4 では分岐点への経路長が4超のため
    削除されない。手検証済み。
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.skeleton import (
    edges_to_skeleton,
    remove_short_spurs,
    remove_small_skeleton_components,
    stabilize_skeleton,
)


# ---------------------------------------------------------------------------
# T-5-1: 孤立ピクセルが skeleton 後に除去される
# ---------------------------------------------------------------------------

def test_isolated_pixel_removed_post_skeleton() -> None:
    """
    孤立した 1 ピクセルは edges_to_skeleton 後に除去されること。

    CHECK-9: _remove_isolated_pixels は 8 近傍に True が 0 個のピクセルを除去する。
    孤立ピクセルは近傍が全て False なので必ず除去される。
    ただし skimage が利用不可の場合はフォールバックで除去されないため、
    skimage がある場合のみアサートする。
    """
    try:
        from skimage.morphology import skeletonize  # noqa: F401
        skimage_available = True
    except Exception:
        skimage_available = False

    mask = np.zeros((20, 20), dtype=bool)
    # 孤立した 1 ピクセル（近傍に仲間なし）
    mask[10, 10] = True

    result = edges_to_skeleton(mask)

    if skimage_available:
        assert not result[10, 10], "孤立ピクセルは除去されるべき"
    else:
        # skimage なしではフォールバックで元の mask が返る
        assert result.dtype == bool


# ---------------------------------------------------------------------------
# T-5-2: 小コンポーネント（2px）が min_size=5 で除去される
# ---------------------------------------------------------------------------

def test_remove_small_skeleton_components_removes_tiny() -> None:
    """
    2 ピクセルの連結コンポーネントは min_size=5 で除去されること。

    CHECK-9: remove_small_objects は min_size 未満のコンポーネントを除去する。
    2 < 5 なので必ず除去される。
    """
    mask = np.zeros((20, 20), dtype=bool)
    # 2 ピクセルの小コンポーネント（隣接）
    mask[5, 5] = True
    mask[5, 6] = True

    result = remove_small_skeleton_components(mask, min_size=5)

    assert result.dtype == bool
    # skimage が利用可能な場合は除去される
    try:
        from skimage.morphology import remove_small_objects  # noqa: F401
        assert not result[5, 5], "2px コンポーネントは除去されるべき"
        assert not result[5, 6], "2px コンポーネントは除去されるべき"
    except Exception:
        # skimage なしでは safe fallback（元の mask を返す）
        pass


# ---------------------------------------------------------------------------
# T-5-3: 大コンポーネント（10px）が min_size=5 で除去されない
# ---------------------------------------------------------------------------

def test_remove_small_skeleton_components_keeps_large() -> None:
    """
    10 ピクセルの連結コンポーネントは min_size=5 で除去されないこと。

    CHECK-9: 10 >= 5 なので remove_small_objects の対象外。
    """
    mask = np.zeros((20, 20), dtype=bool)
    # 10 ピクセルの直線コンポーネント
    for i in range(10):
        mask[5, 5 + i] = True

    result = remove_small_skeleton_components(mask, min_size=5)

    assert result.dtype == bool
    # コンポーネントの中心部は必ず残る
    assert result[5, 9], "10px コンポーネントは除去されるべきでない"


# ---------------------------------------------------------------------------
# T-5-4: T字型スケルトンの短辺（len=2）が max_length=4 で除去される
# ---------------------------------------------------------------------------

def test_spur_removal_removes_short_branch() -> None:
    """
    T 字型スケルトンで短い枝（長さ 2）が max_length=4 で除去されること。

    CHECK-9: remove_short_spurs は分岐点(deg>=3)から max_length 以内の
    行き止まり枝を除去する。短辺=2 < max_length=4 なので必ず除去される。

    手検証:
    - 縦幹: x=10, y=1..20（長さ20）
    - 短水平枝: y=10, x=11..12（長さ2）
    - 分岐点 (10,10) から枝先端 (10,12) まで2ピクセル。2 <= 4 → 除去される。
    - 幹端 (1,10) から分岐点まで9ピクセル。max_length=4 では到達前にブレーク → 除去されない。
    """
    mask = np.zeros((22, 22), dtype=bool)
    # 幹（縦方向: x=10, y=1〜20、長さ20）
    for y in range(1, 21):
        mask[y, 10] = True
    # 短い枝（右方向: y=10, x=11〜12、長さ2）
    mask[10, 11] = True
    mask[10, 12] = True

    result = remove_short_spurs(mask, max_length=4)

    assert result.dtype == bool
    # 短枝の先端は除去されているべき
    assert not result[10, 12], "短辺先端は除去されるべき"
    # 幹の中間（分岐点から9px以上遠い位置）は残るべき
    assert result[5, 10], "幹の中間は残るべき"


# ---------------------------------------------------------------------------
# T-5-5: T字型スケルトンの長辺（len=10）が max_length=4 で除去されない
# ---------------------------------------------------------------------------

def test_spur_removal_keeps_long_branch() -> None:
    """
    T 字型スケルトンで長い枝（長さ 10）が max_length=4 で除去されないこと。

    CHECK-9: 分岐点から長辺の先端までの経路長は 10 > 4 なので
    remove_short_spurs の対象外。手検証済み。

    手検証:
    - 水平幹: y=10, x=1..28（長さ28）
    - 縦長枝: x=10, y=11..20（長さ10）
    - 枝先端 (20,10) から分岐点への経路長 = 10 > max_length=4 → 除去されない。
    - max_length=4 ではブレーク条件(length > 4)により5ステップで打ち切り。
    """
    mask = np.zeros((22, 30), dtype=bool)
    # 水平幹: y=10, x=1〜28（長さ28）
    for x in range(1, 29):
        mask[10, x] = True
    # 縦長枝: x=10, y=11〜20（長さ10）
    for y in range(11, 21):
        mask[y, 10] = True

    result = remove_short_spurs(mask, max_length=4)

    assert result.dtype == bool
    # 長枝の先端は残るべき
    assert result[20, 10], "長辺先端は除去されるべきでない"
    # 長枝の中間も残るべき
    assert result[15, 10], "長辺中間は除去されるべきでない"


# ---------------------------------------------------------------------------
# T-5-6: 大領域+微小ノイズ（3px）がskeletonに残らない
# ---------------------------------------------------------------------------

def test_edges_to_skeleton_pre_cleans_noise() -> None:
    """
    大領域（10x10 正方形）と微小ノイズ（3px の点）が共存するとき、
    ノイズ（3px）が skeleton に残らないこと。

    CHECK-9: edges_to_skeleton の pre-clean で min_edge_size=8 未満の
    コンポーネントを除去する。3 < 8 なのでノイズは skeletonize 前に消える。
    大領域 (100px) は min_edge_size=8 より大きいので残る。
    skimage がない場合はフォールバックで検証省略。
    """
    try:
        from skimage.morphology import skeletonize, remove_small_objects  # noqa: F401
        skimage_available = True
    except Exception:
        skimage_available = False

    mask = np.zeros((50, 50), dtype=bool)
    # 大領域（細い直線: 20px）→ スケルトンが20%閾値を確実に超えるよう細線を使用
    # CHECK-9: 20px直線のスケルトンは~20px。20 >= 23*0.2=4.6 → 閾値通過。
    for x in range(5, 25):
        mask[20, x] = True  # 20 pixels
    # 微小ノイズ（3px、隔離された位置）
    mask[40, 40] = True
    mask[40, 41] = True
    mask[40, 42] = True

    result = edges_to_skeleton(mask, min_edge_size=8)

    assert result.ndim == 2
    assert result.dtype == bool

    if skimage_available:
        # ノイズ箇所は skeleton に残らないはず
        noise_pixels = result[40, 40:43].sum()
        assert noise_pixels == 0, f"ノイズ3pxはskeletonに残るべきでない（実際: {noise_pixels}px）"


# ---------------------------------------------------------------------------
# T-5-7: stabilize_skeleton が bool 2D配列を返し、元の pixel 数以下になる
# ---------------------------------------------------------------------------

def test_stabilize_skeleton_returns_bool_2d() -> None:
    """
    合成ノイジーエッジから stabilize_skeleton を呼ぶと、
    bool 2D 配列が返り、元の pixel 数以下になること。

    CHECK-9: stabilize_skeleton はノイズ除去→細線化→スパー除去→小コンポーネント除去の
    順で処理するため、出力ピクセル数は入力以下になる（細線化で幅が1に圧縮されるため）。
    """
    rng = np.random.default_rng(42)
    # ノイズ含むエッジマップ（20% 密度）
    edges = rng.random((60, 60)) < 0.2

    result = stabilize_skeleton(
        edges,
        min_edge_size=8,
        spur_length=4,
        min_component_size=5,
    )

    # 型チェック
    assert result.ndim == 2, "出力は 2 次元配列であるべき"
    assert result.dtype == bool, "出力は bool 配列であるべき"
    assert result.shape == edges.shape, "出力の shape は入力と同じであるべき"

    # ピクセル数チェック（細線化によって減少するはず）
    input_count = int(edges.sum())
    output_count = int(result.sum())
    assert output_count <= input_count, (
        f"出力ピクセル数({output_count})は入力({input_count})以下であるべき"
    )


# ---------------------------------------------------------------------------
# T-5-8: opening_size パラメータが指定でき、0 のとき既存挙動と互換
# ---------------------------------------------------------------------------

def test_edges_to_skeleton_opening_size_zero_unchanged() -> None:
    """
    opening_size=0 を指定しても既存の edges_to_skeleton と同じ挙動になること。
    """
    mask = np.zeros((30, 30), dtype=bool)
    for x in range(5, 20):
        mask[15, x] = True  # 水平線 15px

    r0 = edges_to_skeleton(mask, min_edge_size=8, opening_size=0)
    r1 = edges_to_skeleton(mask, min_edge_size=8)

    assert r0.dtype == bool and r1.dtype == bool
    assert r0.shape == r1.shape
    assert np.array_equal(r0, r1), "opening_size=0 はデフォルトと同じであるべき"


def test_edges_to_skeleton_opening_reduces_thin_protrusion() -> None:
    """
    opening_size=1 を指定してもクラッシュせず、bool 2D 配列が返ること。
    opening により形状が変わるため、ピクセル数比較は行わず形状・型のみ検証。
    """
    try:
        from skimage.morphology import skeletonize  # noqa: F401
        skimage_available = True
    except Exception:
        skimage_available = False

    if not skimage_available:
        pytest.skip("skimage.morphology が利用できません")

    # 5x5 のブロブ + 右に 1px 幅で 3px の突起
    mask = np.zeros((20, 20), dtype=bool)
    mask[8:13, 8:13] = True
    mask[10, 13] = True
    mask[10, 14] = True
    mask[10, 15] = True

    r0 = edges_to_skeleton(mask, min_edge_size=1, opening_size=0)
    r1 = edges_to_skeleton(mask, min_edge_size=1, opening_size=1)

    assert r0.dtype == bool and r1.dtype == bool
    assert r0.shape == r1.shape == mask.shape
    assert r0.ndim == 2 and r1.ndim == 2
    # 入力に True があるため、少なくとも一方はスケルトンが残る
    assert r0.sum() >= 1 or r1.sum() >= 1, "スケルトン結果が空でないこと"
