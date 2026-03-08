# -*- coding: utf-8 -*-
"""
maze-artisan テスト用画像ファクトリ。
実画像の代わりに Pythonで決定論的に生成するグレースケール画像を提供する。
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def make_circular_gradient(w: int = 256, h: int = 256) -> Image.Image:
    """
    中心が白(255)、周辺が黒(0)の円形グラデーション画像。
    masterpiece: 中央を通る明るい解経路を期待。
    """
    cx, cy = w / 2.0, h / 2.0
    max_r = min(cx, cy)
    ys, xs = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    arr = np.clip(255 * (1.0 - dist / max_r), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def make_horizontal_stripe(w: int = 256, h: int = 256, n_stripes: int = 8) -> Image.Image:
    """
    水平ストライプ（明暗交互）画像。
    masterpiece: ストライプに沿った経路を期待。
    """
    arr = np.zeros((h, w), dtype=np.uint8)
    stripe_h = h // n_stripes
    for i in range(n_stripes):
        if i % 2 == 0:
            y0 = i * stripe_h
            y1 = min(y0 + stripe_h, h)
            arr[y0:y1, :] = 255
    return Image.fromarray(arr, mode="L")


def make_checkerboard(w: int = 256, h: int = 256, n_tiles: int = 8) -> Image.Image:
    """
    チェッカーボード（市松模様）画像。
    tile_size = w/n_tiles × h/n_tiles の白黒格子。
    """
    tile_w = max(1, w // n_tiles)
    tile_h = max(1, h // n_tiles)
    ys, xs = np.mgrid[0:h, 0:w]
    arr = (((xs // tile_w) + (ys // tile_h)) % 2 * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def make_diagonal_gradient(w: int = 256, h: int = 256) -> Image.Image:
    """
    左上が黒(0)、右下が白(255)の対角グラデーション。
    find_image_guided_path が暗角を入口にするため、右下→左上の経路を期待。
    """
    xs = np.linspace(0, 255, w, dtype=np.float32)
    ys = np.linspace(0, 255, h, dtype=np.float32)
    arr = ((xs[np.newaxis, :] + ys[:, np.newaxis]) / 2.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def make_all_white(w: int = 64, h: int = 64) -> Image.Image:
    return Image.fromarray(np.full((h, w), 255, dtype=np.uint8), mode="L")


def make_all_black(w: int = 64, h: int = 64) -> Image.Image:
    return Image.fromarray(np.zeros((h, w), dtype=np.uint8), mode="L")


def make_concentric_rings(w: int = 256, h: int = 256, n_rings: int = 6) -> Image.Image:
    """
    同心円リング画像（circle より線的なパターン）。
    中心からの距離に sin 波を掛けて明暗リングを生成する。
    masterpiece: リングの明部に沿った経路を期待。
    """
    cx, cy = w / 2.0, h / 2.0
    max_r = min(cx, cy)
    ys, xs = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    norm_dist = np.clip(dist / max_r, 0.0, 1.0)
    arr = ((np.sin(norm_dist * n_rings * np.pi) + 1.0) / 2.0 * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def make_face_silhouette(w: int = 256, h: int = 256) -> Image.Image:
    """
    簡易顔シルエット画像（楕円頭部 + 円目2つ + 口横棒）。
    明部（顔）と暗部（目・口）のコントラストが明確。
    masterpiece: 顔領域の明部を通る経路を期待。
    """
    arr = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w / 2.0, h / 2.0
    ys, xs = np.mgrid[0:h, 0:w]

    # 頭部楕円（明部）
    rx, ry = w * 0.35, h * 0.40
    face_cy = cy * 0.90
    face_mask = ((xs - cx) / rx) ** 2 + ((ys - face_cy) / ry) ** 2 <= 1.0
    arr[face_mask] = 200

    # 左目（暗部）
    ex1, ey1, er = cx - w * 0.12, cy * 0.75, w * 0.06
    eye1_mask = (xs - ex1) ** 2 + (ys - ey1) ** 2 <= er ** 2
    arr[eye1_mask] = 40

    # 右目（暗部）
    ex2, ey2 = cx + w * 0.12, cy * 0.75
    eye2_mask = (xs - ex2) ** 2 + (ys - ey2) ** 2 <= er ** 2
    arr[eye2_mask] = 40

    # 口（水平バー・暗部）
    mouth_y1 = int(cy * 1.10)
    mouth_y2 = mouth_y1 + max(2, h // 20)
    mouth_x1 = int(cx - w * 0.12)
    mouth_x2 = int(cx + w * 0.12)
    arr[mouth_y1:mouth_y2, mouth_x1:mouth_x2] = 40

    return Image.fromarray(arr, mode="L")


def make_text_pattern(w: int = 256, h: int = 256) -> Image.Image:
    """
    大文字「A」に似た幾何パターン（逆 V 字の太線 + 横棒）。
    白線（線画）が暗背景に描かれた構造。
    masterpiece: 白い太線に沿った経路を期待。
    """
    arr = np.zeros((h, w), dtype=np.uint8)
    thickness = max(2, w // 16)

    apex_x = w // 2
    apex_y = int(h * 0.10)
    base_y = int(h * 0.90)
    base_half = int(w * 0.30)

    # 左脚・右脚を太線で描画
    steps = base_y - apex_y
    for i in range(steps):
        t = i / max(1, steps)
        y = apex_y + i
        x_left = int(apex_x - t * base_half)
        x_right = int(apex_x + t * base_half)

        # 左脚
        x0 = max(0, x_left - thickness // 2)
        x1 = min(w, x_left + thickness // 2)
        arr[y, x0:x1] = 255

        # 右脚
        x0 = max(0, x_right - thickness // 2)
        x1 = min(w, x_right + thickness // 2)
        arr[y, x0:x1] = 255

    # 横棒（A の中段）
    bar_t = 0.55
    bar_y_center = int(apex_y + bar_t * steps)
    bar_x_left = int(apex_x - bar_t * base_half) + thickness
    bar_x_right = int(apex_x + bar_t * base_half) - thickness
    bar_y0 = max(0, bar_y_center - thickness // 2)
    bar_y1 = min(h, bar_y_center + thickness // 2)
    arr[bar_y0:bar_y1, max(0, bar_x_left):min(w, bar_x_right)] = 255

    return Image.fromarray(arr, mode="L")
