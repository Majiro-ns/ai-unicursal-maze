"""
密度迷路 Phase 1: 前処理（リサイズ、グレースケール、CLAHEコントラスト補正）。
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def preprocess_image(
    image: Image.Image,
    max_side: int = 512,
    contrast_boost: float = 1.0,
) -> np.ndarray:
    """
    画像をリサイズしグレースケール配列で返す。
    戻り値: (H, W) float 0.0〜1.0。長辺が max_side 以下になるようリサイズ。

    contrast_boost: CLAHEの強度（0.0 で無効、1.0 で標準 clipLimit=0.03）。
    cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 相当は contrast_boost=1.0。
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        gray = arr.astype(np.float64) / 255.0
    else:
        # RGB or RGBA
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        gray = (arr @ [0.299, 0.587, 0.114]).astype(np.float64) / 255.0

    h, w = gray.shape
    if max(h, w) > max_side and max_side > 0:
        scale = max_side / max(h, w)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        from skimage import transform
        gray = transform.resize(gray, (new_h, new_w), anti_aliasing=True)

    # CLAHE: 均一画像（std < 1e-4）はスキップ（均等化で全1.0になるため）
    if contrast_boost > 0.0 and float(np.std(gray)) > 1e-4:
        from skimage.exposure import equalize_adapthist
        gray = equalize_adapthist(gray, kernel_size=None, clip_limit=0.03 * contrast_boost, nbins=256)

    return np.clip(gray, 0.0, 1.0)
