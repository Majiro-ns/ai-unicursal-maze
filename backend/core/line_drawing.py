from __future__ import annotations

"""
Line drawing extractor for the unicursal maze pipeline.
- default: coarse + fine Canny blend.
- detail : face band uses stronger thresholds to drop weak hair/background, outside keeps silhouette.
- Face band now uses multi-scale Canny with local contrast enhancement for clearer facial contours.
"""

from typing import Literal, Optional, Tuple

import numpy as np
from PIL import Image
from skimage import feature, filters, morphology, segmentation, transform
from skimage.color import rgb2gray
from skimage.exposure import equalize_adapthist


def _build_face_band_mask(
    h: int,
    w: int,
    top_ratio: float,
    bottom_ratio: float,
    left_ratio: float,
    right_ratio: float,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    top = max(0, min(int(h * top_ratio), h - 1))
    bottom = max(top + 1, min(int(h * bottom_ratio), h))
    left = max(0, min(int(w * left_ratio), w - 1))
    right = max(left + 1, min(int(w * right_ratio), w))
    mask[top:bottom, left:right] = True
    return mask


def _enhance_face_contrast(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply local contrast enhancement (CLAHE) to the face band region.
    This makes subtle facial contours more visible for edge detection.
    """
    enhanced = gray.copy()
    if not mask.any():
        return enhanced

    # Extract face region bounding box
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return enhanced

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Apply CLAHE to face region
    face_region = gray[rmin : rmax + 1, cmin : cmax + 1]
    if face_region.size > 0:
        face_enhanced = equalize_adapthist(
            face_region, kernel_size=None, clip_limit=0.02, nbins=256
        )
        # Blend enhanced region back
        enhanced[rmin : rmax + 1, cmin : cmax + 1] = face_enhanced

    return enhanced


def _multiscale_canny_face(
    gray: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> np.ndarray:
    """
    Multi-scale Canny for face region: combine edges from multiple sigma values.
    This captures both fine details (eyes, lips) and broader contours (jawline).
    """
    # Fine scale: captures small details (eyes, nostrils, lip edges)
    edges_fine = feature.canny(
        gray,
        sigma=0.6,
        low_threshold=low_threshold * 0.7,
        high_threshold=high_threshold * 0.8,
    )

    # Medium scale: main facial contours
    edges_medium = feature.canny(
        gray,
        sigma=1.0,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )

    # Coarse scale: broad structural lines (chin, forehead)
    edges_coarse = feature.canny(
        gray,
        sigma=1.4,
        low_threshold=low_threshold * 1.2,
        high_threshold=high_threshold * 1.3,
    )

    # Combine: prioritize medium, add fine details, reinforce with coarse
    combined = np.logical_or(edges_medium, edges_fine)
    combined = np.logical_or(combined, edges_coarse)

    return combined


def _sharpen_face_edges(edges: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Sharpen and clean up edges in the face region.
    Remove isolated pixels, thin the lines, then slightly dilate for clarity.
    """
    if not mask.any():
        return edges

    result = edges.copy()

    # Extract face region edges
    face_edges = np.logical_and(edges, mask)

    # Remove very small components (noise)
    face_edges_clean = morphology.remove_small_objects(
        face_edges, max_size=3, connectivity=2
    )

    # Skeletonize to get thin, clean lines
    face_skeleton = morphology.skeletonize(face_edges_clean)

    # Slight dilation for visibility (1-pixel disk)
    selem_small = morphology.disk(1)
    face_final = morphology.dilation(face_skeleton, selem_small)

    # Replace face region edges
    result[mask] = face_final[mask]

    return result


def extract_line_drawing(
    image: Image.Image,
    target_size: Tuple[int, int] = (256, 256),
    max_side: Optional[int] = 512,
    mode: Literal["default", "detail"] = "default",
    face_band_top: Optional[float] = None,
    face_band_bottom: Optional[float] = None,
    face_band_left: Optional[float] = None,
    face_band_right: Optional[float] = None,
    face_canny_face_low: Optional[float] = None,
    face_canny_face_high: Optional[float] = None,
    face_canny_bg_low: Optional[float] = None,
    face_canny_bg_high: Optional[float] = None,
    face_gamma: Optional[float] = None,
    face_smooth_sigma: Optional[float] = None,
) -> np.ndarray:
    """
    Return a bool edge mask from the input image.
    - default: coarse + fine Canny for general use.
    - detail : face band emphasized, background suppressed.
    """
    arr = np.asarray(image)

    if arr.ndim == 2:
        gray = arr.astype(float) / 255.0
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr_rgb = np.asarray(image.convert("RGB"))
        gray = rgb2gray(arr_rgb)
    else:
        gray = rgb2gray(arr)

    # Resize to keep longest side <= max_side
    if max_side is not None and gray.ndim == 2 and gray.size > 0:
        h0, w0 = gray.shape
        longest = max(h0, w0)
        if longest > 0:
            scale = min(1.0, float(max_side) / float(longest))
            if scale <= 0.0:
                scale = 1.0
            new_h = max(1, int(round(h0 * scale)))
            new_w = max(1, int(round(w0 * scale)))
            resize_target = (new_h, new_w)
        else:
            resize_target = target_size
    else:
        resize_target = target_size

    if gray.shape != resize_target:
        gray_resized = transform.resize(
            gray,
            resize_target,
            anti_aliasing=True,
        )
    else:
        gray_resized = gray

    h_resized, w_resized = gray_resized.shape

    # Face band mask (only used in detail mode)
    band_mask = np.zeros_like(gray_resized, dtype=bool)
    if mode == "detail":
        top_r = 0.2 if face_band_top is None else float(face_band_top)
        bottom_r = 0.8 if face_band_bottom is None else float(face_band_bottom)
        left_r = 0.0 if face_band_left is None else float(face_band_left)
        right_r = 1.0 if face_band_right is None else float(face_band_right)

        top_r = max(0.0, min(1.0, top_r))
        bottom_r = max(0.0, min(1.0, bottom_r))
        left_r = max(0.0, min(1.0, left_r))
        right_r = max(0.0, min(1.0, right_r))

        if bottom_r <= top_r:
            bottom_r = min(1.0, top_r + 0.1)
        if right_r <= left_r:
            right_r = min(1.0, left_r + 0.1)

        band_mask = _build_face_band_mask(h_resized, w_resized, top_r, bottom_r, left_r, right_r)

    # Preprocess per band: smooth background, enhance face band contrast
    gray_proc = gray_resized.copy()
    gray_face_enhanced = gray_resized.copy()  # Separate version for face processing

    if band_mask.any():
        # Background: apply Gaussian smoothing to reduce noise
        smooth_sigma = 1.4 if face_smooth_sigma is None else float(face_smooth_sigma)
        smoothed = filters.gaussian(gray_proc, sigma=smooth_sigma, preserve_range=True)
        gray_proc[np.logical_not(band_mask)] = smoothed[np.logical_not(band_mask)]

        # Face band: apply CLAHE for local contrast enhancement
        gray_face_enhanced = _enhance_face_contrast(gray_resized, band_mask)

        # Also apply gamma correction to face region
        face_vals = gray_face_enhanced[band_mask]
        if face_vals.size > 0:
            vmin, vmax = np.percentile(face_vals, (2, 98))  # Wider percentile for better range
            if vmax <= vmin:
                vmax = vmin + 1e-3
            face_norm = np.clip((face_vals - vmin) / (vmax - vmin), 0.0, 1.0)
            gamma = 1.3 if face_gamma is None else max(0.5, float(face_gamma))
            face_gamma_vals = np.power(face_norm, 1.0 / gamma)
            gray_face_enhanced[band_mask] = face_gamma_vals
            gray_proc[band_mask] = face_gamma_vals

    # Base Canny
    edges_coarse = feature.canny(gray_proc, sigma=1.0)
    edges_fine = feature.canny(
        gray_proc,
        sigma=0.7,
        low_threshold=0.05,
        high_threshold=0.12,
    )

    if mode == "detail" and band_mask.any():
        # Optimized thresholds: lower for face to capture more contours
        face_low = 0.08 if face_canny_face_low is None else float(face_canny_face_low)
        face_high = 0.20 if face_canny_face_high is None else float(face_canny_face_high)
        bg_low = 0.10 if face_canny_bg_low is None else float(face_canny_bg_low)
        bg_high = 0.22 if face_canny_bg_high is None else float(face_canny_bg_high)

        # Face region: use multi-scale Canny on contrast-enhanced image
        edges_face = _multiscale_canny_face(
            gray_face_enhanced,
            low_threshold=face_low,
            high_threshold=face_high,
        )

        # Background: standard Canny with slightly higher thresholds to suppress noise
        edges_bg = feature.canny(
            gray_proc,
            sigma=0.9,
            low_threshold=bg_low,
            high_threshold=bg_high,
        )

        # Combine face and background edges
        edges = np.zeros_like(edges_coarse, dtype=bool)
        edges[band_mask] = edges_face[band_mask]
        edges[np.logical_not(band_mask)] = np.logical_or(edges_bg, edges_coarse)[np.logical_not(band_mask)]

        # Sharpen face edges for clarity
        edges = _sharpen_face_edges(edges, band_mask)

        # Band-wise small component pruning
        edges = morphology.remove_small_objects(edges, max_size=4, connectivity=2)
        edges_out = morphology.remove_small_objects(
            np.logical_and(edges, np.logical_not(band_mask)),
            max_size=11,
            connectivity=2,
        )
        edges_in = np.logical_and(edges, band_mask)
        # Keep more detail in face region
        edges_in = morphology.remove_small_objects(edges_in, max_size=2, connectivity=2)
        edges = np.logical_or(edges_in, edges_out)
    else:
        edges = np.logical_or(edges_coarse, edges_fine)

    # Dilate -> close to fill small gaps
    selem = morphology.disk(1)
    dilated = morphology.dilation(edges, selem)
    closed = morphology.closing(dilated, selem)

    outline = segmentation.find_boundaries(closed, mode="inner")
    if not outline.any():
        outline = edges

    combined = np.logical_or(outline, edges)
    return combined.astype(bool)
