from __future__ import annotations

"""
顔まわりのセマンティクス抽出 (顔マスク・ランドマーク線) を安定供給する。
- 顔マスク: Mediapipe FaceMesh の FACE_OVAL からポリゴンマスクを生成。
- ランドマーク線: 目/眉/口/輪郭など主要接続を線マスクとして生成。
mediapipe / skimage が無い場合のみ None を返す。検出不可時は None を返し、
上位パイプラインでフォールバックさせる。
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# 依存ライブラリの可用性キャッシュ
# ═══════════════════════════════════════════════════════════════
_SKIMAGE_AVAILABLE: Optional[bool] = None
_MEDIAPIPE_AVAILABLE: Optional[bool] = None


def _check_skimage() -> bool:
    """skimage が利用可能か確認する（結果をキャッシュ）。"""
    global _SKIMAGE_AVAILABLE
    if _SKIMAGE_AVAILABLE is None:
        try:
            from skimage import draw, transform  # type: ignore[import]
            _SKIMAGE_AVAILABLE = True
            logger.debug("skimage is available.")
        except ImportError:
            logger.warning("skimage is not installed. Face mask features will be disabled.")
            _SKIMAGE_AVAILABLE = False
    return _SKIMAGE_AVAILABLE


def _check_mediapipe() -> bool:
    """mediapipe が利用可能か確認する（結果をキャッシュ）。"""
    global _MEDIAPIPE_AVAILABLE
    if _MEDIAPIPE_AVAILABLE is None:
        try:
            import mediapipe as mp  # type: ignore[import]
            _MEDIAPIPE_AVAILABLE = True
            logger.debug("mediapipe is available.")
        except ImportError:
            logger.warning("mediapipe is not installed. Face detection features will be disabled.")
            _MEDIAPIPE_AVAILABLE = False
    return _MEDIAPIPE_AVAILABLE


@dataclass
class FaceSemanticsResult:
    """1回の顔検出で得られる両マスクを保持する。"""
    face_mask: Optional[np.ndarray]  # shape=(H, W) bool, 顔輪郭ポリゴン
    landmark_mask: Optional[np.ndarray]  # shape=(H, W) bool, ランドマーク接続線


def _resize_mask(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """target_size = (W, H) に最近傍リサイズする。"""
    if not _check_skimage():
        return mask

    try:
        from skimage import transform  # type: ignore[import]
    except ImportError:
        logger.warning("_resize_mask: Failed to import skimage.transform.")
        return mask

    target_w, target_h = target_size
    h, w = mask.shape
    if (w, h) == (target_w, target_h):
        return mask

    try:
        resized = transform.resize(
            mask.astype(float),
            (target_h, target_w),
            order=0,
            anti_aliasing=False,
            preserve_range=True,
        )
        return (resized >= 0.5).astype(bool)
    except ValueError as e:
        logger.warning("_resize_mask: Invalid dimensions (ValueError): %s", e)
        return mask
    except Exception as e:
        logger.warning("_resize_mask: Unexpected error: %s", e)
        return mask


def _detect_landmarks(image: Image.Image):
    """
    Mediapipe FaceMesh で 1 顔のランドマーク配列を取得（T-12: 顔・髪再現性の入力）。
    戻り値: (mp_module or None, landmarks or None)
    失敗時は呼び出し元で face_mask/landmark_edges=None のままフォールバック。
    """
    if not _check_mediapipe():
        return None, None

    try:
        import mediapipe as mp  # type: ignore[import]
    except ImportError:
        logger.warning("_detect_landmarks: Failed to import mediapipe.")
        return None, None

    # 画像をRGB配列に変換
    try:
        img_rgb = np.asarray(image.convert("RGB"))
    except Exception as e:
        logger.warning("_detect_landmarks: Failed to convert image to RGB: %s", e)
        return None, None

    mp_fm = mp.solutions.face_mesh

    # FaceMesh の初期化と処理
    try:
        with mp_fm.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as mesh:
            results = mesh.process(img_rgb)
    except RuntimeError as e:
        logger.warning("_detect_landmarks: FaceMesh RuntimeError (GPU/memory issue?): %s", e)
        return mp_fm, None
    except ValueError as e:
        logger.warning("_detect_landmarks: FaceMesh ValueError (invalid input?): %s", e)
        return mp_fm, None
    except Exception as e:
        logger.warning("_detect_landmarks: FaceMesh unexpected error: %s", e)
        return mp_fm, None

    if results is None:
        logger.info("_detect_landmarks: FaceMesh returned None.")
        return mp_fm, None

    if not results.multi_face_landmarks:
        logger.info("_detect_landmarks: No face detected in the image.")
        return mp_fm, None

    return mp_fm, results.multi_face_landmarks[0].landmark


def extract_face_mask(
    image: Image.Image,
    target_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    """
    顔輪郭 (FACE_OVAL) からポリゴンマスクを生成して返す。
    戻り値: shape=(H, W) bool, target_size に従う。検出不可や依存欠如なら None。
    """
    # 依存ライブラリの事前チェック
    if not _check_mediapipe():
        logger.info("extract_face_mask: mediapipe not available.")
        return None
    if not _check_skimage():
        logger.info("extract_face_mask: skimage not available.")
        return None

    # ランドマーク検出
    mp_fm, lm = _detect_landmarks(image)
    if mp_fm is None:
        logger.info("extract_face_mask: FaceMesh module unavailable.")
        return None
    if lm is None:
        logger.info("extract_face_mask: No landmarks detected.")
        return None

    # 画像サイズ取得
    try:
        img_rgb = np.asarray(image.convert("RGB"))
        h, w, _ = img_rgb.shape
    except Exception as e:
        logger.warning("extract_face_mask: Failed to get image dimensions: %s", e)
        return None

    # skimage.draw のインポート
    try:
        from skimage import draw as skdraw  # type: ignore[import]
    except ImportError:
        logger.warning("extract_face_mask: Failed to import skimage.draw.")
        return None

    # FACE_OVAL インデックスの取得
    try:
        oval_edges = list(mp_fm.FACEMESH_FACE_OVAL)
    except AttributeError:
        logger.warning("extract_face_mask: FACEMESH_FACE_OVAL not found in mediapipe.")
        return None
    except TypeError as e:
        logger.warning("extract_face_mask: FACE_OVAL is not iterable: %s", e)
        return None
    except Exception as e:
        logger.warning("extract_face_mask: Failed to get FACE_OVAL: %s", e)
        return None

    # インデックスセットの構築
    idx_set: set[int] = set()
    try:
        for i, j in oval_edges:
            idx_set.add(i)
            idx_set.add(j)
    except (TypeError, ValueError) as e:
        logger.warning("extract_face_mask: Invalid oval_edges format: %s", e)
        return None

    if not idx_set:
        logger.info("extract_face_mask: No valid FACE_OVAL indices.")
        return None

    # 座標の取得
    idx_list = sorted(idx_set)
    try:
        xs = np.array([lm[i].x * w for i in idx_list], dtype=float)
        ys = np.array([lm[i].y * h for i in idx_list], dtype=float)
    except IndexError as e:
        logger.warning("extract_face_mask: Landmark index out of range: %s", e)
        return None
    except AttributeError as e:
        logger.warning("extract_face_mask: Invalid landmark structure: %s", e)
        return None

    # 頂点を角度順に並べて輪郭ポリゴンを作る
    try:
        cx = xs.mean()
        cy = ys.mean()
        angles = np.arctan2(ys - cy, xs - cx)
        order = np.argsort(angles)
        xs = xs[order]
        ys = ys[order]
    except Exception as e:
        logger.warning("extract_face_mask: Failed to sort vertices: %s", e)
        return None

    # ポリゴンマスクの描画
    try:
        mask_full = np.zeros((h, w), dtype=bool)
        rr, cc = skdraw.polygon(ys, xs, (h, w))
        mask_full[rr, cc] = True
    except Exception as e:
        logger.warning("extract_face_mask: Failed to draw polygon: %s", e)
        return None

    return _resize_mask(mask_full, target_size)


def _draw_landmark_connections(
    mask: np.ndarray,
    landmarks: Iterable,
    connections: Iterable[Tuple[int, int]],
) -> bool:
    """
    ランドマーク接続線をマスクに描画するヘルパ。
    成功時 True、失敗時 False を返す。
    """
    try:
        from skimage import draw as skdraw  # type: ignore[import]
    except ImportError:
        logger.warning("_draw_landmark_connections: Failed to import skimage.draw.")
        return False

    try:
        h, w = mask.shape
        lm_list = list(landmarks)
    except Exception as e:
        logger.warning("_draw_landmark_connections: Failed to prepare data: %s", e)
        return False

    draw_count = 0
    error_count = 0
    for conn in connections:
        try:
            i, j = conn
            if i >= len(lm_list) or j >= len(lm_list):
                continue
            p1 = lm_list[i]
            p2 = lm_list[j]
            x1 = int(p1.x * w)
            y1 = int(p1.y * h)
            x2 = int(p2.x * w)
            y2 = int(p2.y * h)
            rr, cc = skdraw.line(y1, x1, y2, x2)
            valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            mask[rr[valid], cc[valid]] = True
            draw_count += 1
        except (TypeError, ValueError, AttributeError) as e:
            error_count += 1
            if error_count <= 3:  # 最初の数件だけログ
                logger.debug("_draw_landmark_connections: Skipping invalid connection: %s", e)
            continue
        except Exception as e:
            error_count += 1
            if error_count <= 3:
                logger.debug("_draw_landmark_connections: Unexpected error: %s", e)
            continue

    if error_count > 3:
        logger.debug("_draw_landmark_connections: %d more errors suppressed.", error_count - 3)

    if draw_count == 0:
        logger.warning("_draw_landmark_connections: No valid connections drawn.")
        return False

    logger.debug("_draw_landmark_connections: Drew %d connections.", draw_count)
    return True


def extract_face_landmark_edges(
    image: Image.Image,
    target_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    """
    目・眉・口・輪郭など主要ランドマーク接続を線マスクで返す。
    戻り値: shape=(H, W) bool, target_size に従う。依存欠如/検出不可なら None。
    """
    # 依存ライブラリの事前チェック
    if not _check_mediapipe():
        logger.info("extract_face_landmark_edges: mediapipe not available.")
        return None
    if not _check_skimage():
        logger.info("extract_face_landmark_edges: skimage not available.")
        return None

    # ランドマーク検出
    mp_fm, lm = _detect_landmarks(image)
    if mp_fm is None:
        logger.info("extract_face_landmark_edges: FaceMesh module unavailable.")
        return None
    if lm is None:
        logger.info("extract_face_landmark_edges: No landmarks detected.")
        return None

    # 画像サイズ取得
    try:
        img_rgb = np.asarray(image.convert("RGB"))
        h, w, _ = img_rgb.shape
    except Exception as e:
        logger.warning("extract_face_landmark_edges: Failed to get image dimensions: %s", e)
        return None

    # 接続情報の収集
    connections: set[Tuple[int, int]] = set()
    connection_sources: List[Tuple[str, str]] = [
        ("FACEMESH_LEFT_EYE", "左目"),
        ("FACEMESH_RIGHT_EYE", "右目"),
        ("FACEMESH_LEFT_EYEBROW", "左眉"),
        ("FACEMESH_RIGHT_EYEBROW", "右眉"),
        ("FACEMESH_LIPS", "唇"),
        ("FACEMESH_FACE_OVAL", "輪郭"),
    ]

    for attr_name, desc in connection_sources:
        try:
            if hasattr(mp_fm, attr_name):
                conn_data = getattr(mp_fm, attr_name)
                connections.update(conn_data)
                logger.debug("extract_face_landmark_edges: Loaded %s (%s).", attr_name, desc)
            else:
                logger.debug("extract_face_landmark_edges: %s not available.", attr_name)
        except TypeError as e:
            logger.warning("extract_face_landmark_edges: %s is not iterable: %s", attr_name, e)
        except Exception as e:
            logger.warning("extract_face_landmark_edges: Failed to get %s: %s", attr_name, e)

    # 鼻は別途チェック（オプショナル）
    try:
        if hasattr(mp_fm, "FACEMESH_NOSE"):
            connections.update(mp_fm.FACEMESH_NOSE)
            logger.debug("extract_face_landmark_edges: Loaded FACEMESH_NOSE.")
    except Exception as e:
        logger.debug("extract_face_landmark_edges: FACEMESH_NOSE not available: %s", e)

    if not connections:
        logger.warning("extract_face_landmark_edges: No valid connections found.")
        return None

    # マスクの描画
    mask_full = np.zeros((h, w), dtype=bool)
    success = _draw_landmark_connections(mask_full, lm, connections)

    if not success:
        logger.warning("extract_face_landmark_edges: Failed to draw connections.")
        return None

    return _resize_mask(mask_full, target_size)


def extract_face_semantics(
    image: Image.Image,
    target_size: Tuple[int, int],
) -> FaceSemanticsResult:
    """
    1回の顔検出で face_mask と landmark_mask の両方を取得する統合関数。

    顔検出は重い処理のため、2つのマスクが必要な場合はこの関数を使うこと。
    検出不可や依存欠如の場合は各フィールドが None になる。

    Args:
        image: 入力画像 (PIL Image)
        target_size: 出力マスクのサイズ (W, H)

    Returns:
        FaceSemanticsResult: face_mask と landmark_mask を含む結果
    """
    # 依存ライブラリの事前チェック
    if not _check_mediapipe():
        logger.info("extract_face_semantics: mediapipe not available.")
        return FaceSemanticsResult(face_mask=None, landmark_mask=None)
    if not _check_skimage():
        logger.info("extract_face_semantics: skimage not available.")
        return FaceSemanticsResult(face_mask=None, landmark_mask=None)

    # ランドマーク検出（1回のみ）
    mp_fm, lm = _detect_landmarks(image)

    if mp_fm is None or lm is None:
        logger.info("extract_face_semantics: Face detection failed.")
        return FaceSemanticsResult(face_mask=None, landmark_mask=None)

    # 画像サイズ取得
    try:
        img_rgb = np.asarray(image.convert("RGB"))
        h, w, _ = img_rgb.shape
    except Exception as e:
        logger.warning("extract_face_semantics: Failed to get image dimensions: %s", e)
        return FaceSemanticsResult(face_mask=None, landmark_mask=None)

    # skimage のインポート
    try:
        from skimage import draw as skdraw  # type: ignore[import]
    except ImportError:
        logger.warning("extract_face_semantics: Failed to import skimage.draw.")
        return FaceSemanticsResult(face_mask=None, landmark_mask=None)

    # --- face_mask: FACE_OVAL からポリゴンマスク ---
    face_mask: Optional[np.ndarray] = None
    try:
        oval_edges = list(mp_fm.FACEMESH_FACE_OVAL)
        idx_set: set[int] = set()
        for i, j in oval_edges:
            idx_set.add(i)
            idx_set.add(j)

        if idx_set:
            idx_list = sorted(idx_set)
            xs = np.array([lm[i].x * w for i in idx_list], dtype=float)
            ys = np.array([lm[i].y * h for i in idx_list], dtype=float)

            # 頂点を角度順に並べて輪郭ポリゴンを作る
            cx = xs.mean()
            cy = ys.mean()
            angles = np.arctan2(ys - cy, xs - cx)
            order = np.argsort(angles)
            xs = xs[order]
            ys = ys[order]

            mask_full = np.zeros((h, w), dtype=bool)
            rr, cc = skdraw.polygon(ys, xs, (h, w))
            mask_full[rr, cc] = True
            face_mask = _resize_mask(mask_full, target_size)
            logger.debug("extract_face_semantics: face_mask created successfully.")
    except AttributeError as e:
        logger.warning("extract_face_semantics: FACE_OVAL attribute error: %s", e)
    except IndexError as e:
        logger.warning("extract_face_semantics: Landmark index error in face_mask: %s", e)
    except Exception as e:
        logger.warning("extract_face_semantics: face_mask extraction failed: %s", e)

    # --- landmark_mask: 主要ランドマーク接続線 ---
    landmark_mask: Optional[np.ndarray] = None
    try:
        connections: set[Tuple[int, int]] = set()
        connection_attrs = [
            "FACEMESH_LEFT_EYE",
            "FACEMESH_RIGHT_EYE",
            "FACEMESH_LEFT_EYEBROW",
            "FACEMESH_RIGHT_EYEBROW",
            "FACEMESH_LIPS",
            "FACEMESH_FACE_OVAL",
        ]
        for attr_name in connection_attrs:
            if hasattr(mp_fm, attr_name):
                try:
                    connections.update(getattr(mp_fm, attr_name))
                except TypeError:
                    logger.debug("extract_face_semantics: %s is not iterable.", attr_name)

        if hasattr(mp_fm, "FACEMESH_NOSE"):
            try:
                connections.update(mp_fm.FACEMESH_NOSE)
            except TypeError:
                pass

        if connections:
            mask_full = np.zeros((h, w), dtype=bool)
            success = _draw_landmark_connections(mask_full, lm, connections)
            if success:
                landmark_mask = _resize_mask(mask_full, target_size)
                logger.debug("extract_face_semantics: landmark_mask created successfully.")
            else:
                logger.warning("extract_face_semantics: landmark_mask drawing failed.")
    except Exception as e:
        logger.warning("extract_face_semantics: landmark_mask extraction failed: %s", e)

    return FaceSemanticsResult(face_mask=face_mask, landmark_mask=landmark_mask)
