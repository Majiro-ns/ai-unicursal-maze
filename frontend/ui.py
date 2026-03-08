import base64
import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image, ImageDraw

API_URL = (os.environ.get("MAZE_API_URL", "http://localhost:8000/api/generate_maze") or "").strip()
DENSITY_API_URL = (os.environ.get("DENSITY_API_URL", "http://localhost:8000/api/generate_density_maze") or "").strip()


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
    maze_weight: float = 0.0,    # T-10 追加
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
        "maze_weight": f"{maze_weight:.2f}",   # T-10
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


def _call_density_api(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,  # type: ignore[attr-defined]
    grid_size: int,
    width: int,
    height: int,
    stroke_width: float,
    show_solution: bool,
    density_factor: float,
    max_side: int,
    edge_weight: float,
    edge_sigma: float,
    edge_low_threshold: float,
    edge_high_threshold: float,
    contrast_boost: float,
    use_texture: bool,
    use_heuristic: bool,
    bias_strength: float,
    preset: str,
    n_segments: int,
    extra_removal_rate: float = 0.0,
    dark_threshold: float = 0.3,
    light_threshold: float = 0.7,
) -> dict:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    data = {
        "grid_size": str(int(grid_size)),
        "width": str(int(width)),
        "height": str(int(height)),
        "stroke_width": f"{float(stroke_width):.1f}",
        "show_solution": "true" if show_solution else "false",
        "density_factor": f"{float(density_factor):.2f}",
        "max_side": str(int(max_side)),
        "edge_weight": f"{float(edge_weight):.2f}",
        "edge_sigma": f"{float(edge_sigma):.1f}",
        "edge_low_threshold": f"{float(edge_low_threshold):.3f}",
        "edge_high_threshold": f"{float(edge_high_threshold):.3f}",
        "contrast_boost": f"{float(contrast_boost):.2f}",
        "use_texture": "true" if use_texture else "false",
        "use_heuristic": "true" if use_heuristic else "false",
        "bias_strength": f"{float(bias_strength):.2f}",
        "preset": preset,
        "n_segments": str(int(n_segments)),
        "extra_removal_rate": f"{float(extra_removal_rate):.2f}",
        "dark_threshold": f"{float(dark_threshold):.2f}",
        "light_threshold": f"{float(light_threshold):.2f}",
    }
    response = requests.post(DENSITY_API_URL, files=files, data=data, timeout=120)
    response.raise_for_status()
    return response.json()


def _density_maze_tab() -> None:
    """密度迷路（Phase 2）のパラメータチューニングUI。"""
    st.markdown("## 密度迷路モード（画像→グリッド迷路）")
    st.caption(
        "画像の明暗を通路密度に変換します。暗い領域=通路が密集（黒く見える）、"
        "明るい領域=通路が疎（白く見える）。"
    )

    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.markdown("### 基本パラメータ")

        grid_size = st.slider(
            "グリッドサイズ",
            min_value=10,
            max_value=150,
            value=50,
            step=5,
            help="横・縦のセル数（正方形）。大きいほど細密な迷路。10〜150。",
        )

        density_factor = st.slider(
            "密度倍率",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="輝度差の強調度。大きいほど暗部と明部の通路差が際立つ。0.5〜3.0。",
        )

        st.markdown("### Phase 2: 輝度前処理")

        contrast_boost = st.slider(
            "コントラスト強調 (CLAHE)",
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="CLAHE適応ヒストグラム均等化の強度。0.0=無効、1.0=標準、3.0=最大。",
        )

        st.markdown("### Phase 2 Stage 4: エッジ強調")

        edge_weight = st.slider(
            "エッジ強調（Canny）",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help=(
                "Cannyエッジ検出で元画像の輪郭部分の壁を保持する強度。"
                "0.0=なし（Phase1相当）、1.0=最大（輪郭が白い線として残る）。"
            ),
        )

        if edge_weight > 0.0:
            with st.expander("Cannyパラメータ（詳細）", expanded=False):
                edge_sigma = st.slider(
                    "ガウシアンぼかし (sigma)",
                    min_value=0.3,
                    max_value=5.0,
                    value=1.0,
                    step=0.1,
                    help="Canny前のガウシアンぼかし強度。大きいほどノイズ耐性が上がるが、細かいエッジを見逃す。",
                )
                edge_low_threshold = st.slider(
                    "Canny 下限閾値",
                    min_value=0.01,
                    max_value=0.30,
                    value=0.05,
                    step=0.01,
                    help="Canny二重閾値の下限（0〜1）。小さいほど多くのエッジを検出。",
                )
                edge_high_threshold = st.slider(
                    "Canny 上限閾値",
                    min_value=0.05,
                    max_value=0.60,
                    value=0.20,
                    step=0.01,
                    help="Canny二重閾値の上限（0〜1）。大きいほど強いエッジのみ検出。",
                )
        else:
            edge_sigma = 1.0
            edge_low_threshold = 0.05
            edge_high_threshold = 0.20

        st.markdown("### テクスチャ（Phase 2）")

        use_texture = st.checkbox(
            "テクスチャ分類を使う",
            value=False,
            help="K-meansで輝度を分割しテクスチャパターン（RANDOM/DIRECTIONAL/SPIRAL）を適用。",
        )

        if use_texture:
            preset = st.selectbox(
                "テクスチャプリセット",
                options=["generic", "face", "landscape"],
                index=0,
                format_func=lambda x: {"generic": "汎用", "face": "顔画像向け", "landscape": "風景向け"}[x],
            )
            n_segments = st.slider(
                "クラスタ数",
                min_value=2,
                max_value=8,
                value=4,
                step=1,
                help="K-meansの輝度クラスタ数。2〜8。",
            )
            bias_strength = st.slider(
                "テクスチャバイアス強度",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="テクスチャ方向バイアスの強度。0=なし、1=最大。",
            )
        else:
            preset = "generic"
            n_segments = 4
            bias_strength = 0.5

        use_heuristic = st.checkbox(
            "解ヒューリスティクスを使う",
            value=False,
            help="視覚的に美しい解経路（輝度変化が大きい経路）を選択するヒューリスティクス。",
        )

        st.markdown("### Phase 2b: 密度制御（ループ許容）")

        extra_removal_rate = st.slider(
            "追加壁除去率（暗部ループ）",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help=(
                "暗いセルで追加の壁を除去しループを作る強度。"
                "0.0=完全なspanning tree（デフォルト）、1.0=最大ループ。"
                "大きいほど暗部の通路が密集する。"
            ),
        )

        if extra_removal_rate > 0.0:
            with st.expander("密度制御パラメータ（詳細）", expanded=False):
                dark_threshold = st.slider(
                    "暗部閾値",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.05,
                    help="この輝度未満のセルを「暗部」として追加壁除去の対象にする。",
                )
                light_threshold = st.slider(
                    "明部閾値",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.05,
                    help="この輝度超のセルを「明部」として通路削除の対象にする（連結性を保護）。",
                )
        else:
            dark_threshold = 0.3
            light_threshold = 0.7

        st.markdown("### 出力設定")

        width = st.number_input("出力幅 (px)", min_value=200, max_value=3000, value=800, step=100)
        height = st.number_input("出力高さ (px)", min_value=200, max_value=3000, value=600, step=100)
        stroke_width = st.slider("線の太さ", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
        max_side = st.slider(
            "画像前処理サイズ上限",
            min_value=64,
            max_value=1024,
            value=512,
            step=64,
            help="入力画像の長辺をこのピクセル数以下にリサイズしてから処理。大きいほど高精度。",
        )
        show_solution = st.checkbox("解経路を表示", value=True)

    with col_r:
        st.markdown("### 画像アップロードと生成")

        uploaded_density = st.file_uploader(
            "画像ファイル（JPG / PNG）",
            type=["png", "jpg", "jpeg"],
            key="density_uploader",
        )

        if uploaded_density is not None:
            preview_img = Image.open(BytesIO(uploaded_density.getvalue()))
            st.image(preview_img, caption="入力画像プレビュー", use_column_width=True)

        if st.button("密度迷路を生成", type="primary"):
            if uploaded_density is None:
                st.warning("先に画像をアップロードしてください。")
                return

            try:
                with st.spinner("密度迷路を生成中..."):
                    payload = _call_density_api(
                        uploaded_file=uploaded_density,
                        grid_size=int(grid_size),
                        width=int(width),
                        height=int(height),
                        stroke_width=float(stroke_width),
                        show_solution=bool(show_solution),
                        density_factor=float(density_factor),
                        max_side=int(max_side),
                        edge_weight=float(edge_weight),
                        edge_sigma=float(edge_sigma),
                        edge_low_threshold=float(edge_low_threshold),
                        edge_high_threshold=float(edge_high_threshold),
                        contrast_boost=float(contrast_boost),
                        use_texture=bool(use_texture),
                        use_heuristic=bool(use_heuristic),
                        bias_strength=float(bias_strength),
                        preset=str(preset),
                        n_segments=int(n_segments),
                        extra_removal_rate=float(extra_removal_rate),
                        dark_threshold=float(dark_threshold),
                        light_threshold=float(light_threshold),
                    )

                maze_id = payload.get("maze_id", "density-maze")
                png_b64 = payload.get("maze_png_base64", "")
                svg_str = payload.get("maze_svg", "")
                solution_path = payload.get("solution_path", [])
                grid_rows = payload.get("grid_rows", 0)
                grid_cols = payload.get("grid_cols", 0)
                entrance = payload.get("entrance", {})
                exit_info = payload.get("exit", {})

                if not png_b64:
                    st.error("PNG データがレスポンスに含まれていません。")
                    return

                png_bytes = base64.b64decode(png_b64)

                st.subheader("生成結果")
                st.caption(f"Maze ID: {maze_id} / グリッド: {grid_rows}×{grid_cols}")
                st.image(png_bytes, use_column_width=True)

                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.metric("解経路の長さ（セル数）", len(solution_path))
                    if entrance:
                        st.write(f"入口: 行 {entrance.get('row')}, 列 {entrance.get('col')}")
                with col_d2:
                    st.metric("グリッドセル総数", grid_rows * grid_cols if grid_rows and grid_cols else 0)
                    if exit_info:
                        st.write(f"出口: 行 {exit_info.get('row')}, 列 {exit_info.get('col')}")

                if svg_str:
                    st.download_button(
                        label="SVG をダウンロード",
                        data=svg_str.encode("utf-8"),
                        file_name=f"{maze_id}.svg",
                        mime="image/svg+xml",
                    )
                st.download_button(
                    label="PNG をダウンロード",
                    data=png_bytes,
                    file_name=f"{maze_id}.png",
                    mime="image/png",
                )

            except requests.RequestException as exc:
                st.error(f"API 呼び出し中にエラーが発生しました: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"予期しないエラーが発生しました: {exc}")


def main() -> None:
    st.title("AI 迷路ジェネレーター")

    tab1, tab2 = st.tabs(["一筆書き迷路（V1）", "密度迷路（Phase 2）"])

    with tab2:
        _density_maze_tab()

    with tab1:
        st.write(
            "画像から線画を抽出し、一筆書き→迷路→ダミー枝追加という 4 段階で確認できるビューです。"
        )

    st.sidebar.header("出力オプション（一筆書き迷路）")

    st.sidebar.markdown("#### 解像度（T-7）")
    width = st.sidebar.number_input(
        "出力幅 (px)",
        min_value=100,
        max_value=4000,
        value=800,
        step=50,
        help="生成画像の幅。100〜4000。デフォルト 800。",
    )
    height = st.sidebar.number_input(
        "出力高さ (px)",
        min_value=100,
        max_value=4000,
        value=600,
        step=50,
        help="生成画像の高さ。100〜4000。デフォルト 600。",
    )

    st.sidebar.markdown("#### 線の太さ（T-7）")
    stroke_width = st.sidebar.slider(
        "線の太さ",
        min_value=1.0,
        max_value=20.0,
        value=6.0,
        step=0.5,
        help="迷路の線の太さ（ピクセル）。1.0〜20.0。デフォルト 6.0。",
    )

    st.sidebar.markdown("#### その他表示")
    line_mode = st.sidebar.selectbox(
        "線画モード",
        options=["default", "detail"],
        index=0,
        format_func=lambda x: "標準" if x == "default" else "細部重視 (detail)",
        help="線画抽出のモード。detail は顔帯域で細部を強調。",
    )

    st.sidebar.markdown("### トレードオフ設定")
    maze_weight = st.sidebar.slider(
        "迷路性 ↔ 顔らしさ (maze_weight)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="0.0=元画像の顔に沿ったパス（顔らしさ優先） / 1.0=曲がりくねった迷路（迷路性優先）",
    )

    st.sidebar.markdown("### 迷路の粗さ（T-7）")
    spur_length = st.sidebar.slider(
        "スパー最大長",
        min_value=0,
        max_value=12,
        value=4,
        step=1,
        help="スケルトンの短い突起を除去する最大長（ピクセル）。大きいほど迷路が粗くなる。0〜12、デフォルト 4。",
    )
    min_edge_size = st.sidebar.slider(
        "ノイズ除去閾値",
        min_value=1,
        max_value=32,
        value=8,
        step=1,
        help="スケルトン化前に除去する小領域の最大ピクセル数。大きいほど細かい線・ノイズが除去される。1〜32、デフォルト 8。",
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
                    maze_weight=float(maze_weight),    # T-10 追加
                )

            maze_id = payload.get("maze_id", "maze")
            svg_str = payload.get("svg", "")
            png_b64 = payload.get("png_base64", "")
            timings = payload.get("timings") or {}
            num_solutions = payload.get("num_solutions")
            difficulty_score = payload.get("difficulty_score")
            turn_count = payload.get("turn_count")        # T-9
            path_length = payload.get("path_length")      # T-9
            dead_end_count = payload.get("dead_end_count")  # T-9

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
                if turn_count is not None:
                    st.write(f"・曲がり角の数: {turn_count}")
                if path_length is not None:
                    st.write(f"・経路長（ノード数）: {path_length}")
                if dead_end_count is not None:
                    st.write(f"・袋小路の数: {dead_end_count}")

        except requests.RequestException as exc:
            st.error(f"API 呼び出し中にエラーが発生しました: {exc}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"予期しないエラーが発生しました: {exc}")


if __name__ == "__main__":
    main()
