import base64
import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image, ImageDraw

API_URL = os.environ.get("MAZE_API_URL", "http://localhost:8000/api/generate_maze")


def _call_api(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,  # type: ignore[attr-defined]
    width: int,
    height: int,
    stroke_width: float,
    line_mode: str,
    face_band_top: float,
    face_band_bottom: float,
    face_band_left: float,
    face_band_right: float,
    use_overlay: bool,
    use_face_canny_detail: bool,
    stage: str,
    debug_path_scoring: bool = False,
    spur_length: int = 4,
    min_edge_size: int = 8,
) -> dict:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    data = {
        "width": str(int(width)),
        "height": str(int(height)),
        "stroke_width": str(float(stroke_width)),
        "line_mode": line_mode,
        "face_band_top": f"{face_band_top:.2f}",
        "face_band_bottom": f"{face_band_bottom:.2f}",
        "face_band_left": f"{face_band_left:.2f}",
        "face_band_right": f"{face_band_right:.2f}",
        "use_overlay": "true" if use_overlay else "false",
        "use_face_canny_detail": "true" if use_face_canny_detail else "false",
        "stage": stage,
        "debug_path_scoring": "true" if debug_path_scoring else "false",
        "spur_length": str(int(spur_length)),
        "min_edge_size": str(int(min_edge_size)),
    }

    response = requests.post(API_URL, files=files, data=data, timeout=60)
    response.raise_for_status()
    return response.json()


def _show_face_band_preview(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,  # type: ignore[attr-defined]
    face_band_top: float,
    face_band_bottom: float,
    face_band_left: float,
    face_band_right: float,
) -> None:
    st.markdown("#### 顔帯域プレビュー")
    try:
        img = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
        preview_h = 256
        w0, h0 = img.size
        if h0 <= 0:
            return
        scale = preview_h / float(h0)
        preview_w = max(1, int(w0 * scale))
        img_resized = img.resize((preview_w, preview_h))

        draw = ImageDraw.Draw(img_resized, "RGBA")
        top_px = int(preview_h * face_band_top)
        bottom_px = int(preview_h * face_band_bottom)
        left_px = int(preview_w * face_band_left)
        right_px = int(preview_w * face_band_right)

        top_px = max(0, min(preview_h - 1, top_px))
        bottom_px = max(top_px + 1, min(preview_h, bottom_px))
        left_px = max(0, min(preview_w - 1, left_px))
        right_px = max(left_px + 1, min(preview_w, right_px))

        draw.rectangle(
            [(left_px, top_px), (right_px, bottom_px)],
            outline=(0, 255, 0, 255),
            fill=(0, 255, 0, 64),
            width=2,
        )

        st.image(
            img_resized,
            caption=(
                f"顔帯域: 上端 {face_band_top:.2f}, 下端 {face_band_bottom:.2f}, "
                f"左端 {face_band_left:.2f}, 右端 {face_band_right:.2f} （0.00–1.00 比率）"
            ),
            use_column_width=True,
        )
    except Exception as exc:  # noqa: BLE001
        st.info(f"顔帯域プレビューの生成に失敗しました: {exc}")


def main() -> None:
    st.title("AI 一筆迷路ジェネレーター")
    st.write(
        "画像から線画を抽出し、一筆書き→迷路→ダミー枝追加という 4 段階で確認できるビューです。"
    )

    st.sidebar.header("出力オプション")
    width = st.sidebar.number_input("出力幅 (px)", min_value=100, max_value=4000, value=800, step=50)
    height = st.sidebar.number_input("出力高さ (px)", min_value=100, max_value=4000, value=600, step=50)
    stroke_width = st.sidebar.slider("線の太さ", min_value=1.0, max_value=20.0, value=6.0, step=0.5)
    line_mode = st.sidebar.selectbox(
        "線画モード",
        options=["default", "detail"],
        index=0,
        format_func=lambda x: "標準" if x == "default" else "細部重視 (detail)",
    )

    st.sidebar.markdown("### 迷路の粗さ")
    spur_length = st.sidebar.slider(
        "スパー最大長（大きいほど迷路が粗い）",
        min_value=0,
        max_value=12,
        value=4,
        step=1,
        help="スケルトンの短い突起を除去する最大長。大きくするほど細かい枝が消え迷路が粗くなる。",
    )
    min_edge_size = st.sidebar.slider(
        "ノイズ除去閾値（大きいほど細かい線を除去）",
        min_value=1,
        max_value=32,
        value=8,
        step=1,
        help="スケルトン化前に除去する小領域の最大ピクセル数。大きくするほどノイズが除去される。",
    )

    with st.sidebar.expander("顔帯域 (detail モード用)", expanded=False):
        face_band_top = st.number_input(
            "顔帯域 上端 (0.00–1.00)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.01,
            format="%.2f",
        )
        face_band_bottom = st.number_input(
            "顔帯域 下端 (0.00–1.00)",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.01,
            format="%.2f",
        )
        face_band_left = st.number_input(
            "顔帯域 左端 (0.00–1.00)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            format="%.2f",
        )
        face_band_right = st.number_input(
            "顔帯域 右端 (0.00–1.00)",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.01,
            format="%.2f",
        )
        use_face_canny_detail = st.checkbox("顔帯域で Canny 細部を使う", value=False)

    use_overlay = st.sidebar.checkbox("グレースケール下絵を重ねる", value=True)

    uploaded_file = st.file_uploader("画像ファイルを選択", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None and line_mode == "detail":
        _show_face_band_preview(
            uploaded_file,
            float(face_band_top),
            float(face_band_bottom),
            float(face_band_left),
            float(face_band_right),
        )

    st.markdown("### 処理ステップ")
    st.write("任意の段階の結果を個別に確認できます。")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    stage = None
    stage_label = None

    if col1.button("① 線画を生成"):
        stage = "line"
        stage_label = "線画 (特徴抽出まで)"
    elif col2.button("② 線画を一筆書きにする"):
        stage = "unicursal"
        stage_label = "線画を一筆パスに変換"
    elif col3.button("③ 一筆迷路を生成"):
        stage = "maze"
        stage_label = "一筆パスのみの迷路（solver 情報付き）"
    elif col4.button("④ ダミー迷路を構築"):
        stage = "dummy"
        stage_label = "ダミー枝付きの最終迷路"

    if stage is not None:
        if uploaded_file is None:
            st.warning("先に画像ファイルをアップロードしてください。")
            return

        try:
            with st.spinner(f"{stage_label} を生成中..."):
                payload = _call_api(
                    uploaded_file=uploaded_file,
                    width=int(width),
                    height=int(height),
                    stroke_width=float(stroke_width),
                    line_mode=line_mode,
                    face_band_top=float(face_band_top),
                    face_band_bottom=float(face_band_bottom),
                    face_band_left=float(face_band_left),
                    face_band_right=float(face_band_right),
                    use_overlay=bool(use_overlay),
                    use_face_canny_detail=bool(use_face_canny_detail),
                    stage=stage,
                    debug_path_scoring=False,
                    spur_length=int(spur_length),
                    min_edge_size=int(min_edge_size),
                )

            maze_id = payload.get("maze_id", "maze")
            svg_str = payload.get("svg", "")
            png_b64 = payload.get("png_base64", "")
            timings = payload.get("timings") or {}
            num_solutions = payload.get("num_solutions")
            difficulty_score = payload.get("difficulty_score")

            if not png_b64:
                st.error("PNG データがレスポンスに含まれていません。")
                return

            png_bytes = base64.b64decode(png_b64)

            st.subheader("生成結果")
            st.caption(f"ステージ: {stage_label} / Maze ID: {maze_id}")
            st.image(png_bytes, use_column_width=True)

            if svg_str:
                st.download_button(
                    label="SVG をダウンロード",
                    data=svg_str.encode("utf-8"),
                    file_name=f"{maze_id}_{stage}.svg",
                    mime="image/svg+xml",
                )

            st.download_button(
                label="PNG をダウンロード",
                data=png_bytes,
                file_name=f"{maze_id}_{stage}.png",
                mime="image/png",
            )
            debug_b64 = payload.get("path_weight_debug_base64")
            if debug_b64:
                debug_bytes = base64.b64decode(debug_b64)
                st.download_button(
                    label="Path weight debug PNG",
                    data=debug_bytes,
                    file_name=f"{maze_id}_{stage}_path_weight_debug.png",
                    mime="image/png",
                )


            if timings:
                st.subheader("処理時間の内訳 (ms)")
                line_ms = timings.get("line_and_features_ms")
                path_ms = timings.get("graph_and_path_ms")
                render_ms = timings.get("render_ms")
                total_ms = timings.get("total_ms")

                if line_ms is not None:
                    st.write(f"- 線画＋特徴抽出: {line_ms:.1f}")
                if path_ms is not None:
                    st.write(f"- グラフ＋一筆パス: {path_ms:.1f}")
                if render_ms is not None:
                    st.write(f"- 描画: {render_ms:.1f}")
                if total_ms is not None:
                    st.write(f"- 合計: {total_ms:.1f}")

            if num_solutions is not None:
                st.subheader("解の個数 / 難易度")
                if num_solutions == 0:
                    st.write("・解なし (start/goal 間にパスがありません)")
                elif num_solutions == 1:
                    st.write("・一意解 (start→goal のパスは 1 本)")
                else:
                    st.write("・複数解 (2 本以上のパスあり)")

                if difficulty_score is not None:
                    st.write(f"・難易度スコア: {difficulty_score:.3f}")

        except requests.RequestException as exc:
            st.error(f"API 呼び出し中にエラーが発生しました: {exc}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"予期しないエラーが発生しました: {exc}")


if __name__ == "__main__":
    main()
