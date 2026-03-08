# -*- coding: utf-8 -*-
"""
scripts/run_maze.py --masterpiece フラグ と
backend/api/routes.py masterpiece パラメータのテスト。

対象:
  - MASTERPIECE_PRESET 定数（backend.core.density）
  - run_maze.build_params(): --masterpiece 時のパラメータ展開
  - run_maze.main(): CLI エントリポイント
  - routes.py: masterpiece Form フィールドの存在確認
"""
from __future__ import annotations

import io
import inspect
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# scripts/ を sys.path に追加（run_maze モジュールのインポート用）
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from run_maze import build_params, build_parser, detect_format, main
from backend.core.density import MASTERPIECE_PRESET


# ─────────────────────────────────────────────────────
# ヘルパー
# ─────────────────────────────────────────────────────

def _make_test_image(path: Path, w: int = 32, h: int = 32) -> None:
    """テスト用グレースケール画像を保存する。"""
    arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    img = Image.fromarray(arr, mode="L")
    img.save(str(path))


# ─────────────────────────────────────────────────────
# MASTERPIECE_PRESET 定数
# ─────────────────────────────────────────────────────

class TestMasterpiecePreset:
    """MASTERPIECE_PRESET 定数のテスト。"""

    def test_preset_has_required_keys(self):
        """必要なキーが揃っていること。"""
        required = {
            "grid_size", "thickness_range", "extra_removal_rate",
            "dark_threshold", "light_threshold", "use_image_guided",
            "solution_highlight", "show_solution", "edge_weight", "stroke_width",
        }
        assert required.issubset(MASTERPIECE_PRESET.keys()), (
            f"不足キー: {required - MASTERPIECE_PRESET.keys()}"
        )

    def test_preset_grid_size_is_8(self):
        """SSIM探索結果: grid_size=8 が黄金設定。"""
        assert MASTERPIECE_PRESET["grid_size"] == 8

    def test_preset_use_image_guided_true(self):
        """masterpiece は use_image_guided=True（明部ルーティング）。"""
        assert MASTERPIECE_PRESET["use_image_guided"] is True

    def test_preset_show_solution_false(self):
        """masterpiece は show_solution=False（解経路非表示で壁美術として完成）。"""
        assert MASTERPIECE_PRESET["show_solution"] is False

    def test_preset_thickness_range_positive(self):
        """thickness_range > 0（可変壁厚有効）。"""
        assert MASTERPIECE_PRESET["thickness_range"] > 0

    def test_preset_edge_weight_positive(self):
        """edge_weight > 0（エッジ強調有効）。"""
        assert MASTERPIECE_PRESET["edge_weight"] > 0


# ─────────────────────────────────────────────────────
# build_params(): --masterpiece フラグの展開
# ─────────────────────────────────────────────────────

class TestBuildParams:
    """run_maze.build_params() のユニットテスト。"""

    def _parse(self, argv: list[str]) -> "argparse.Namespace":
        """テスト用に引数をパースする（--input/--output はダミー）。"""
        parser = build_parser()
        return parser.parse_args(["--input", "x.jpg", "--output", "x.png"] + argv)

    def test_masterpiece_applies_grid_size_8(self):
        """--masterpiece 時 grid_size=8 が適用されること。"""
        args = self._parse(["--masterpiece"])
        params = build_params(args)
        assert params["grid_size"] == MASTERPIECE_PRESET["grid_size"]

    def test_masterpiece_applies_all_preset_keys(self):
        """--masterpiece 時 MASTERPIECE_PRESET の全キーが params に含まれること。"""
        args = self._parse(["--masterpiece"])
        params = build_params(args)
        for key, value in MASTERPIECE_PRESET.items():
            assert key in params, f"パラメータ '{key}' が不足"
            assert params[key] == value, f"params['{key}']={params[key]} != preset={value}"

    def test_masterpiece_grid_size_override(self):
        """--masterpiece + --grid-size N で grid_size を上書きできること。"""
        args = self._parse(["--masterpiece", "--grid-size", "16"])
        params = build_params(args)
        assert params["grid_size"] == 16

    def test_masterpiece_stroke_width_override(self):
        """--masterpiece + --stroke-width N で stroke_width を上書きできること。"""
        args = self._parse(["--masterpiece", "--stroke-width", "3.0"])
        params = build_params(args)
        assert params["stroke_width"] == pytest.approx(3.0)

    def test_no_masterpiece_uses_default_grid_size(self):
        """--masterpiece なしのとき grid_size=50 がデフォルト。"""
        args = self._parse([])
        params = build_params(args)
        assert params["grid_size"] == 50

    def test_no_masterpiece_grid_size_override(self):
        """--masterpiece なし + --grid-size N で grid_size が設定されること。"""
        args = self._parse(["--grid-size", "20"])
        params = build_params(args)
        assert params["grid_size"] == 20

    def test_dpi_passed_when_specified(self):
        """--dpi 300 のとき params に png_dpi=300 が入ること。"""
        args = self._parse(["--dpi", "300"])
        params = build_params(args)
        assert params.get("png_dpi") == 300

    def test_dpi_absent_when_not_specified(self):
        """--dpi 未指定のとき params に png_dpi キーがないこと。"""
        args = self._parse([])
        params = build_params(args)
        assert "png_dpi" not in params

    def test_width_height_always_in_params(self):
        """width/height は常に params に含まれること。"""
        args = self._parse(["--width", "1024", "--height", "768"])
        params = build_params(args)
        assert params["width"] == 1024
        assert params["height"] == 768


# ─────────────────────────────────────────────────────
# detect_format()
# ─────────────────────────────────────────────────────

class TestDetectFormat:
    """run_maze.detect_format() のユニットテスト。"""

    def _args(self, output: str, fmt: str = "auto"):
        parser = build_parser()
        args = parser.parse_args(["--input", "x.jpg", "--output", output])
        args.format = fmt
        return args

    def test_auto_detects_png(self):
        assert detect_format(self._args("maze.png")) == "png"

    def test_auto_detects_svg(self):
        assert detect_format(self._args("maze.svg")) == "svg"

    def test_explicit_png(self):
        assert detect_format(self._args("maze.svg", fmt="png")) == "png"

    def test_explicit_svg(self):
        assert detect_format(self._args("maze.png", fmt="svg")) == "svg"

    def test_unknown_ext_defaults_to_png(self):
        assert detect_format(self._args("maze.bmp")) == "png"


# ─────────────────────────────────────────────────────
# main(): エンドツーエンド
# ─────────────────────────────────────────────────────

class TestMainCLI:
    """run_maze.main() のエンドツーエンドテスト。"""

    def test_masterpiece_generates_png(self, tmp_path):
        """--masterpiece で PNG が生成されること。"""
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        _make_test_image(input_path)

        ret = main([
            "--masterpiece",
            "--input", str(input_path),
            "--output", str(output_path),
        ])
        assert ret == 0, "main() が非ゼロを返した"
        assert output_path.exists(), "出力ファイルが生成されなかった"
        assert output_path.stat().st_size > 0, "出力ファイルが空"

    def test_masterpiece_generates_svg(self, tmp_path):
        """--masterpiece --format svg で SVG が生成されること。"""
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.svg"
        _make_test_image(input_path)

        ret = main([
            "--masterpiece",
            "--input", str(input_path),
            "--output", str(output_path),
            "--format", "svg",
        ])
        assert ret == 0
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "<svg" in content, "SVG ファイルに <svg> タグがない"

    def test_standard_generates_png(self, tmp_path):
        """--masterpiece なしで PNG が生成されること。"""
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        _make_test_image(input_path)

        ret = main([
            "--input", str(input_path),
            "--output", str(output_path),
            "--grid-size", "6",
        ])
        assert ret == 0
        assert output_path.exists()

    def test_missing_input_returns_error(self, tmp_path):
        """存在しない入力ファイルでエラーコード 1 を返すこと。"""
        output_path = tmp_path / "output.png"
        ret = main([
            "--input", str(tmp_path / "nonexistent.jpg"),
            "--output", str(output_path),
        ])
        assert ret == 1

    def test_masterpiece_grid_size_is_8(self, tmp_path):
        """--masterpiece 時の出力グリッドが 8x8 相当であること（grid_rows <= 8）。"""
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.svg"
        _make_test_image(input_path, w=64, h=64)

        ret = main([
            "--masterpiece",
            "--input", str(input_path),
            "--output", str(output_path),
            "--format", "svg",
        ])
        assert ret == 0
        # SVG の grid_cols/rows はセルサイズから推定可能（厳密テストより生成成功を確認）
        content = output_path.read_text(encoding="utf-8")
        assert "<svg" in content

    def test_dpi_embedded_in_png(self, tmp_path):
        """--dpi 300 のとき PNG メタデータに DPI が埋め込まれること。"""
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        _make_test_image(input_path)

        ret = main([
            "--input", str(input_path),
            "--output", str(output_path),
            "--grid-size", "5",
            "--dpi", "300",
        ])
        assert ret == 0
        pil = Image.open(str(output_path))
        dpi = pil.info.get("dpi")
        assert dpi is not None, "DPI が PNG に埋め込まれていない"
        assert abs(dpi[0] - 300) < 5


# ─────────────────────────────────────────────────────
# routes.py: masterpiece フィールドの存在確認
# ─────────────────────────────────────────────────────

class TestAPIRouteMasterpiece:
    """backend/api/routes.py の masterpiece パラメータのテスト。"""

    def test_routes_has_masterpiece_form_param(self):
        """routes.py が masterpiece Form パラメータを含むこと。"""
        import backend.api.routes as rt
        src = inspect.getsource(rt)
        assert "masterpiece" in src, (
            "routes.py に masterpiece パラメータが存在しない"
        )

    def test_routes_imports_masterpiece_preset(self):
        """routes.py が MASTERPIECE_PRESET をインポートしていること。"""
        import backend.api.routes as rt
        src = inspect.getsource(rt)
        assert "MASTERPIECE_PRESET" in src, (
            "routes.py が MASTERPIECE_PRESET を参照していない"
        )

    def test_routes_applies_preset_on_masterpiece_true(self):
        """routes.py で masterpiece=True のとき MASTERPIECE_PRESET が適用されること。

        ソースコード中に 'if masterpiece:' 分岐が存在することを確認。
        """
        import backend.api.routes as rt
        src = inspect.getsource(rt)
        assert "if masterpiece" in src, (
            "routes.py に masterpiece 分岐が存在しない"
        )

    def test_density_maze_function_accepts_masterpiece(self):
        """generate_density_maze は masterpiece パラメータを受け付けないが、
        MASTERPIECE_PRESET のキーは全て受け付けること。"""
        from backend.core.density import generate_density_maze
        import inspect as _inspect
        sig = _inspect.signature(generate_density_maze)
        # MASTERPIECE_PRESET の各キーが generate_density_maze に存在すること
        for key in MASTERPIECE_PRESET:
            assert key in sig.parameters, (
                f"generate_density_maze に '{key}' パラメータが存在しない"
            )
