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
