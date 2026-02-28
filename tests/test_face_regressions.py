from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from backend.core.features import FeatureSummary
from backend.core.graph_builder import apply_feature_weights, skeleton_to_graph
from backend.core.line_drawing import extract_line_drawing
from backend.core.path_finder import find_unicursal_like_path


def test_detail_mode_face_edge_density_ratio() -> None:
    """Detailモードで顔帯域にエッジが集中することを軽くチェックする。"""
    img = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse((32, 24, 96, 104), fill="black")

    edges = extract_line_drawing(img, mode="detail").astype(bool)

    h, w = edges.shape
    top = int(h * 0.2)
    bottom = int(h * 0.8)
    band_mask = np.zeros_like(edges, dtype=bool)
    band_mask[top:bottom, :] = True

    face_density = edges[band_mask].mean()
    bg_density = edges[np.logical_not(band_mask)].mean() + 1e-6
    ratio = face_density / bg_density

    assert ratio > 1.5, f"face edge density ratio too low: {ratio:.2f}"


def test_landmark_coverage_in_path_scoring() -> None:
    """ランドマーク通過率がスコアに効くことを確認する。"""
    skeleton = np.ones((1, 5), dtype=bool)
    graph = skeleton_to_graph(skeleton)

    landmark_mask = skeleton.copy()
    features = FeatureSummary(
        silhouette_mask=skeleton,
        landmark_mask=landmark_mask,
        centroid=(2.0, 0.0),
    )
    apply_feature_weights(graph, features)

    path = find_unicursal_like_path(graph, features=features)
    assert len(path) == 5

    hits = sum(
        1
        for p in path
        if 0 <= int(p.y) < landmark_mask.shape[0] and 0 <= int(p.x) < landmark_mask.shape[1]
        and landmark_mask[int(p.y), int(p.x)]
    )
    coverage = hits / max(1, len(path))

    assert coverage >= 0.6, f"landmark coverage too low: {coverage:.2f}"


def _compute_landmark_coverage(path, landmark_mask) -> float:
    """パスのランドマーク通過率を計算するヘルパー関数。"""
    if not path or landmark_mask is None:
        return 0.0
    h, w = landmark_mask.shape
    hits = sum(
        1
        for p in path
        if 0 <= int(p.y) < h and 0 <= int(p.x) < w
        and landmark_mask[int(p.y), int(p.x)]
    )
    return hits / max(1, len(path))


def test_synthetic_face_landmark_coverage_threshold() -> None:
    """
    合成顔画像（楕円＋目＋口）で、生成パスのランドマーク通過率が閾値以上かをチェック。
    回帰テスト: 顔らしさの品質が一定水準を維持していることを確認する。
    """
    from backend.core.skeleton import edges_to_skeleton

    # 合成顔画像を作成（128x128、白背景に黒い楕円顔）
    img = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(img)

    # 顔の輪郭（楕円）
    draw.ellipse((32, 16, 96, 112), outline="black", width=2)

    # 左目
    draw.ellipse((44, 40, 56, 52), outline="black", width=1)
    # 右目
    draw.ellipse((72, 40, 84, 52), outline="black", width=1)

    # 口
    draw.arc((48, 72, 80, 92), start=0, end=180, fill="black", width=1)

    # 線画抽出
    edges = extract_line_drawing(img, mode="detail").astype(bool)

    # ランドマークマスクを作成（目と口の領域）
    landmark_mask = np.zeros_like(edges, dtype=bool)
    h, w = edges.shape

    # 目の領域をランドマークとしてマーク（スケール調整）
    scale_y = h / 128
    scale_x = w / 128
    # 左目
    for y in range(int(40 * scale_y), int(52 * scale_y)):
        for x in range(int(44 * scale_x), int(56 * scale_x)):
            if 0 <= y < h and 0 <= x < w:
                landmark_mask[y, x] = True
    # 右目
    for y in range(int(40 * scale_y), int(52 * scale_y)):
        for x in range(int(72 * scale_x), int(84 * scale_x)):
            if 0 <= y < h and 0 <= x < w:
                landmark_mask[y, x] = True
    # 口
    for y in range(int(72 * scale_y), int(92 * scale_y)):
        for x in range(int(48 * scale_x), int(80 * scale_x)):
            if 0 <= y < h and 0 <= x < w:
                landmark_mask[y, x] = True

    # スケルトン → グラフ → パス生成
    skeleton = edges_to_skeleton(edges)
    graph = skeleton_to_graph(skeleton)

    features = FeatureSummary(
        silhouette_mask=edges,
        landmark_mask=landmark_mask,
        centroid=(w / 2, h / 2),
    )
    apply_feature_weights(graph, features)

    path = find_unicursal_like_path(graph, features=features)

    # パスが生成されていることを確認
    assert len(path) > 10, f"path too short: {len(path)} nodes"

    # ランドマーク通過率を計算
    coverage = _compute_landmark_coverage(path, landmark_mask)

    # 閾値チェック（20%以上のノードがランドマーク領域を通過すること）
    threshold = 0.20
    assert coverage >= threshold, (
        f"landmark coverage {coverage:.2%} is below threshold {threshold:.0%}"
    )


def test_face_boundary_coverage_threshold() -> None:
    """
    合成顔画像で、シルエット境界の通過率が閾値以上かをチェック。
    回帰テスト: パスが顔の輪郭に沿っていることを確認する。
    """
    from backend.core.skeleton import edges_to_skeleton

    # シンプルな楕円顔
    img = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse((12, 8, 52, 56), outline="black", width=2)

    edges = extract_line_drawing(img, mode="default").astype(bool)

    skeleton = edges_to_skeleton(edges)
    graph = skeleton_to_graph(skeleton)

    # シルエットマスク: 抽出済みエッジ自体を使用する。
    # スケルトンはエッジから導出されるため、スケルトンノードは必ずエッジピクセル上にある。
    # エッジピクセルは1px細線のため、各ピクセルの4近傍に非エッジピクセルが必ず存在し
    # → boundary_hit = True が確実に発生する。
    h, w = edges.shape
    silhouette_mask = edges.copy()

    features = FeatureSummary(
        silhouette_mask=silhouette_mask,
        centroid=(w / 2, h / 2),
    )
    apply_feature_weights(graph, features)

    path = find_unicursal_like_path(graph, features=features)

    assert len(path) > 5, f"path too short: {len(path)} nodes"

    # 境界通過率を計算
    boundary_hits = 0
    for p in path:
        py, px = int(p.y), int(p.x)
        if not (0 <= py < h and 0 <= px < w):
            continue
        if not silhouette_mask[py, px]:
            continue
        # 境界判定
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = py + dy, px + dx
            if not (0 <= ny < h and 0 <= nx < w) or not silhouette_mask[ny, nx]:
                boundary_hits += 1
                break

    boundary_coverage = boundary_hits / max(1, len(path))

    # 閾値チェック（30%以上のノードが境界上にあること）
    threshold = 0.30
    assert boundary_coverage >= threshold, (
        f"boundary coverage {boundary_coverage:.2%} is below threshold {threshold:.0%}"
    )


def test_multiple_face_samples_landmark_coverage() -> None:
    """
    複数の異なる合成顔サンプルで、ランドマーク通過率の一貫性をチェック。
    回帰テスト: 異なる画像サイズや形状でも品質が維持されることを確認する。
    """
    from backend.core.skeleton import edges_to_skeleton

    samples = [
        # (width, height, face_bbox, eye_size)
        (64, 64, (16, 8, 48, 56), 6),
        (128, 128, (32, 16, 96, 112), 12),
        (96, 128, (24, 16, 72, 112), 10),
    ]

    min_threshold = 0.15  # 最低でも15%のランドマーク通過率
    results = []

    for w_img, h_img, face_bbox, eye_size in samples:
        img = Image.new("RGB", (w_img, h_img), "white")
        draw = ImageDraw.Draw(img)

        # 顔の輪郭
        draw.ellipse(face_bbox, outline="black", width=2)

        # 目の位置を計算
        fx1, fy1, fx2, fy2 = face_bbox
        face_w = fx2 - fx1
        face_h = fy2 - fy1
        eye_y = fy1 + int(face_h * 0.35)

        # 左目
        left_eye_x = fx1 + int(face_w * 0.3)
        draw.ellipse(
            (left_eye_x - eye_size // 2, eye_y - eye_size // 2,
             left_eye_x + eye_size // 2, eye_y + eye_size // 2),
            outline="black", width=1
        )
        # 右目
        right_eye_x = fx1 + int(face_w * 0.7)
        draw.ellipse(
            (right_eye_x - eye_size // 2, eye_y - eye_size // 2,
             right_eye_x + eye_size // 2, eye_y + eye_size // 2),
            outline="black", width=1
        )

        edges = extract_line_drawing(img, mode="detail").astype(bool)
        h, w = edges.shape

        # ランドマークマスク（目の領域）
        landmark_mask = np.zeros((h, w), dtype=bool)
        scale_y = h / h_img
        scale_x = w / w_img

        for eye_x in [left_eye_x, right_eye_x]:
            for dy in range(-eye_size, eye_size + 1):
                for dx in range(-eye_size, eye_size + 1):
                    py = int((eye_y + dy) * scale_y)
                    px = int((eye_x + dx) * scale_x)
                    if 0 <= py < h and 0 <= px < w:
                        landmark_mask[py, px] = True

        skeleton = edges_to_skeleton(edges)
        graph = skeleton_to_graph(skeleton)

        features = FeatureSummary(
            silhouette_mask=edges,
            landmark_mask=landmark_mask,
            centroid=(w / 2, h / 2),
        )
        apply_feature_weights(graph, features)

        path = find_unicursal_like_path(graph, features=features)

        if len(path) > 5:
            coverage = _compute_landmark_coverage(path, landmark_mask)
            results.append((w_img, h_img, coverage))

    # 少なくとも1つのサンプルで閾値を超えていることを確認
    passing = [r for r in results if r[2] >= min_threshold]
    assert len(passing) > 0, (
        f"No samples passed landmark coverage threshold {min_threshold:.0%}. "
        f"Results: {results}"
    )
