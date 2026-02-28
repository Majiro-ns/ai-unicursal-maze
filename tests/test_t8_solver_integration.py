# -*- coding: utf-8 -*-
"""
T-8: 一意解評価ソルバ導入 統合テスト

対象: backend.core.models.MazeResult, backend.core.maze_generator.generate_unicursal_maze,
      backend.api.routes (ソース検査)

確認事項:
  1. MazeResult に num_solutions フィールドが存在する
  2. MazeResult に difficulty_score フィールドが存在する
  3. MazeResult のデフォルト値が None である
  4. generate_unicursal_maze の戻り値に num_solutions が含まれる（None でない）
  5. num_solutions が int 型である
  6. num_solutions が 0 以上の整数である
  7. routes.py ソースに num_solutions の参照あり（inspect）

CHECK-9 テスト期待値の根拠:
  - テスト1/2: dataclass フィールドを直接確認。Optional[int] = None はデータクラス定義から明白。
  - テスト3: デフォルト値 None はモデル定義から明白。
  - テスト4-6: 小サイズ白画像（200×200）で generate_unicursal_maze を実行。
    スケルトンが得られれば solver が走り int が返る。0 または 1 以上のいずれかになる。
    None になるのは path_points < 2 の fallback 時のみ（白画像は sparse graph の可能性あり）。
    白画像は edges が少ない／ゼロの場合、fallback になり None が返ることもある。
    そのため test4-6 は 楕円が描かれた画像を使用してスケルトンが確実に生成されるようにする。
  - テスト7: routes.py ソースコードに num_solutions が含まれることをソース解析で確認。
"""

from __future__ import annotations

import inspect

import numpy as np
from PIL import Image, ImageDraw

from backend.core.models import MazeResult, MazeOptions


# ---------------------------------------------------------------------------
# T-8-1: MazeResult に num_solutions フィールドが存在する
# ---------------------------------------------------------------------------
def test_maze_result_has_num_solutions_field():
    result = MazeResult(maze_id="x", svg="<svg/>", png_bytes=b"")
    assert hasattr(result, "num_solutions"), "MazeResult に num_solutions フィールドがない"


# ---------------------------------------------------------------------------
# T-8-2: MazeResult に difficulty_score フィールドが存在する
# ---------------------------------------------------------------------------
def test_maze_result_has_difficulty_score_field():
    result = MazeResult(maze_id="x", svg="<svg/>", png_bytes=b"")
    assert hasattr(result, "difficulty_score"), "MazeResult に difficulty_score フィールドがない"


# ---------------------------------------------------------------------------
# T-8-3: MazeResult のデフォルト値は None
# ---------------------------------------------------------------------------
def test_maze_result_num_solutions_defaults_none():
    result = MazeResult(maze_id="x", svg="<svg/>", png_bytes=b"")
    assert result.num_solutions is None, f"num_solutions のデフォルトが None でない: {result.num_solutions}"
    assert result.difficulty_score is None, f"difficulty_score のデフォルトが None でない: {result.difficulty_score}"


# ---------------------------------------------------------------------------
# T-8-4〜6: generate_unicursal_maze の戻り値に solver 結果が含まれる
# ---------------------------------------------------------------------------
def _make_ellipse_image(size: int = 200) -> Image.Image:
    """スケルトンが確実に生成される楕円画像を作成する。"""
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    margin = size // 5
    draw.ellipse(
        (margin, margin, size - margin, size - margin),
        outline="black",
        width=3,
    )
    return img


def test_generate_unicursal_maze_returns_num_solutions():
    """
    generate_unicursal_maze は solver を実行し、num_solutions を MazeResult に格納する。

    根拠: maze_generator.py L929 の solve_path() 呼び出し結果を L989 以降の return に反映した（T-8修正）。
    楕円画像はスケルトン生成が確実なので solve_result が None にならない。
    """
    from backend.core.maze_generator import generate_unicursal_maze

    image = _make_ellipse_image(200)
    options = MazeOptions(width=400, height=300)
    result = generate_unicursal_maze(image, options)

    # num_solutions は int または None（fallback 時）
    # 楕円画像ではスケルトン生成が確実なため int が返るはず
    assert result.num_solutions is not None, (
        "generate_unicursal_maze の戻り値に num_solutions が含まれていない（None）。"
        "maze_generator.py の return MazeResult に num_solutions を追加したか確認せよ。"
    )


def test_generate_unicursal_maze_num_solutions_is_int():
    """
    num_solutions は int 型でなければならない。

    根拠: SolveResult.num_solutions は int として定義されている。
    """
    from backend.core.maze_generator import generate_unicursal_maze

    image = _make_ellipse_image(200)
    options = MazeOptions(width=400, height=300)
    result = generate_unicursal_maze(image, options)

    if result.num_solutions is not None:
        assert isinstance(result.num_solutions, int), (
            f"num_solutions が int でない: {type(result.num_solutions)}"
        )


def test_generate_unicursal_maze_num_solutions_range():
    """
    num_solutions は 0 以上の整数でなければならない。

    根拠: 解の個数は非負整数。solver は max_solutions=2 で最大 2 を返す（0, 1, 2 のいずれか）。
    """
    from backend.core.maze_generator import generate_unicursal_maze

    image = _make_ellipse_image(200)
    options = MazeOptions(width=400, height=300)
    result = generate_unicursal_maze(image, options)

    if result.num_solutions is not None:
        assert result.num_solutions >= 0, (
            f"num_solutions が 0 未満: {result.num_solutions}"
        )


# ---------------------------------------------------------------------------
# T-8-7: routes.py ソースに num_solutions の参照あり
# ---------------------------------------------------------------------------
def test_routes_response_includes_num_solutions():
    """
    routes.py のレスポンスに num_solutions が含まれていることをソース解析で確認する。

    根拠: routes.py は result.num_solutions をレスポンス dict に格納している必要がある。
    """
    import backend.api.routes as rt
    src = inspect.getsource(rt)
    assert "num_solutions" in src, "routes.py が num_solutions をレスポンスに含めていない"
    assert "difficulty_score" in src, "routes.py が difficulty_score をレスポンスに含めていない"
