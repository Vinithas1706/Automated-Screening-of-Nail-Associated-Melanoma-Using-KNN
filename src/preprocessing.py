"""Image reading and preprocessing.

The SAME function (`preprocess`) is used for training images and for images
uploaded through the web app, so training and prediction always match.
"""
from pathlib import Path

import cv2
import numpy as np

from . import config


def read_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw bytes to a BGR image. Raises ValueError if unreadable."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError("Image could not be decoded (corrupted or unsupported file).")
    return img


def read_image_path(path) -> np.ndarray:
    """Read from disk via bytes (works with non-ASCII Windows paths)."""
    return read_image_bytes(Path(path).read_bytes())


def segment_nail(bgr: np.ndarray):
    """Heuristic nail/foreground segmentation (Otsu threshold + morphology).

    Tries both polarities and keeps the largest blob that is reasonably sized,
    central, and not touching the image border. Falls back to the whole frame
    when nothing suitable is found (roi_found=False).
    Returns (mask uint8 0/255, contour, roi_found).
    """
    size = bgr.shape[0]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, m1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    best, best_score = None, None
    for m in (m1, cv2.bitwise_not(m1)):
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area_ratio = cv2.contourArea(c) / float(size * size)
        if not 0.05 <= area_ratio <= 0.90:
            continue
        x, y, w, h = cv2.boundingRect(c)
        touches = int(x <= 1) + int(y <= 1) + int(x + w >= size - 1) + int(y + h >= size - 1)
        mom = cv2.moments(c)
        if mom["m00"] == 0:
            continue
        cx, cy = mom["m10"] / mom["m00"] / size, mom["m01"] / mom["m00"] / size
        score = float(np.hypot(cx - 0.5, cy - 0.5)) + 0.4 * touches
        if best_score is None or score < best_score:
            best, best_score = c, score

    if best is None:
        full = np.array([[[0, 0]], [[size - 1, 0]], [[size - 1, size - 1]], [[0, size - 1]]], dtype=np.int32)
        mask = np.full((size, size), 255, dtype=np.uint8)
        return mask, full, False

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.drawContours(mask, [best], -1, 255, thickness=-1)
    return mask, best, True


def preprocess(bgr: np.ndarray) -> dict:
    """Run the full preprocessing pipeline and return every stage.

    resized   : 224x224 BGR
    denoised  : bilateral-filtered (edge-preserving noise reduction); used for COLOUR features
    gray      : grayscale of the denoised image
    enhanced  : CLAHE contrast enhancement on the L channel (LAB); used for TEXTURE/SHAPE
    normalized: denoised pixels scaled to [0, 1] float32
    """
    s = config.IMG_SIZE
    resized = cv2.resize(bgr, (s, s), interpolation=cv2.INTER_AREA)
    denoised = cv2.bilateralFilter(resized, d=7, sigmaColor=40, sigmaSpace=40)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    normalized = denoised.astype(np.float32) / 255.0
    mask, contour, roi_found = segment_nail(enhanced)

    return {
        "resized": resized,
        "denoised": denoised,
        "gray": gray,
        "enhanced": enhanced,
        "enhanced_gray": enhanced_gray,
        "normalized": normalized,
        "mask": mask,
        "contour": contour,
        "roi_found": roi_found,
    }
