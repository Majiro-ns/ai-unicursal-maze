# -*- coding: utf-8 -*-
"""
T-3: maze-artisan generate_unicursal_maze 最小限テスト

対象関数: backend.core.maze_generator.generate_unicursal_maze
確認事項:
  1. MazeResult が返ること（型チェック）
  2. maze_id が空でないこと
  3. svg フィールドが '<svg' で始まること
  4. png_bytes が非None かつ非空であること

設計書: 02_Design.md
"""

from __future__ import annotations

import pytest
from PIL import Image

from backend.core.maze_generator import generate_unicursal_maze
from backend.core.models import MazeOptions, MazeResult


@pytest.fixture
def basic_image() -> Image.Image:
    """最小限のテスト用入力画像（白背景 200×200）"""
    return Image.new("RGB", (200, 200), "white")


@pytest.fixture
def basic_result(basic_image: Image.Image) -> MazeResult:
    """generate_unicursal_maze の基本実行結果（各テストで共有）"""
    return generate_unicursal_maze(basic_image, MazeOptions())


# ---------------------------------------------------------------------------
# T-3-1: 返り値の型チェック
# ---------------------------------------------------------------------------

def test_returns_maze_result(basic_result: MazeResult) -> None:
    """
    generate_unicursal_maze は MazeResult を返すこと。
    CHECK-9: MazeResult は dataclass。isinstance で型確認。
    """
    assert isinstance(basic_result, MazeResult)


# ---------------------------------------------------------------------------
# T-3-2: maze_id が空でないこと
# ---------------------------------------------------------------------------

def test_maze_id_not_empty(basic_result: MazeResult) -> None:
    """
    maze_id フィールドが非None・非空文字列であること。
    CHECK-9: maze_id はユニーク識別子として使われる。空なら下流処理が壊れる。
    """
    assert basic_result.maze_id is not None
    assert isinstance(basic_result.maze_id, str)
    assert len(basic_result.maze_id) > 0


# ---------------------------------------------------------------------------
# T-3-3: svg フィールドが '<svg' で始まること
# ---------------------------------------------------------------------------

def test_svg_starts_with_svg_tag(basic_result: MazeResult) -> None:
    """
    svg フィールドは有効な SVG 文字列であること。
    '<svg' タグを含むことを確認。
    実際の出力は XML 宣言（<?xml ...?>）が先頭に付く場合があるため
    startswith ではなく 'in' でチェックする。
    CHECK-9: '<svg' が含まれることで SVG として有効であることを確認。
    """
    assert isinstance(basic_result.svg, str)
    assert "<svg" in basic_result.svg, \
        f"svg フィールドに '<svg' が含まれない: {basic_result.svg[:80]!r}"


# ---------------------------------------------------------------------------
# T-3-4: png_bytes が非None・非空であること
# ---------------------------------------------------------------------------

def test_png_bytes_non_empty(basic_result: MazeResult) -> None:
    """
    png_bytes フィールドが非None かつ非空（バイト列）であること。
    CHECK-9: PNG バイト列がないと画像表示・ダウンロードが機能しない。
    """
    assert basic_result.png_bytes is not None
    assert isinstance(basic_result.png_bytes, (bytes, bytearray))
    assert len(basic_result.png_bytes) > 0
