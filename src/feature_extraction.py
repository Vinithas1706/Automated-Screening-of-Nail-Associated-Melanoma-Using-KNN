"""Hand-crafted feature extraction (colour + texture + shape).

`extract_from_image` is the single entry point used by BOTH training and the
web app, which guarantees identical features at train and predict time.
"""
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

from . import preprocessing

HIST_BINS = 8
_COLOUR_RGB = [f"{c}_mean" for c in "RGB"] + [f"{c}_std" for c in "RGB"]
_COLOUR_HIST = [f"hist_{c}_{i}" for c in "RGB" for i in range(HIST_BINS)]
_COLOUR_HSV = ["S_mean", "S_std", "V_mean", "V_std"]
_TEXTURE = ["glcm_contrast", "glcm_correlation", "glcm_energy", "glcm_homogeneity", "entropy"]
_SHAPE = ["nail_area_ratio", "perimeter_ratio", "aspect_ratio", "solidity",
          "circularity", "edge_density", "gradient_mean", "gradient_std"]

FEATURE_GROUPS = {
    "colour": _COLOUR_RGB + _COLOUR_HIST + _COLOUR_HSV,
    "texture": _TEXTURE,
    "shape": _SHAPE,
}
FEATURE_NAMES = FEATURE_GROUPS["colour"] + FEATURE_GROUPS["texture"] + FEATURE_GROUPS["shape"]


def _colour_features(denoised_bgr, mask):
    f = {}
    sel = mask > 0
    rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)[sel].astype(np.float32) / 255.0
    for i, c in enumerate("RGB"):
        f[f"{c}_mean"] = float(rgb[:, i].mean())
        f[f"{c}_std"] = float(rgb[:, i].std())

    rgb_img = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
    for i, c in enumerate("RGB"):
        h = cv2.calcHist([rgb_img], [i], mask, [HIST_BINS], [0, 256]).flatten()
        h = h / max(h.sum(), 1e-9)
        for j in range(HIST_BINS):
            f[f"hist_{c}_{j}"] = float(h[j])

    hsv = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2HSV)[sel].astype(np.float32) / 255.0
    f["S_mean"], f["S_std"] = float(hsv[:, 1].mean()), float(hsv[:, 1].std())
    f["V_mean"], f["V_std"] = float(hsv[:, 2].mean()), float(hsv[:, 2].std())
    return f


def _texture_features(gray_enh, mask, contour):
    x, y, w, h = cv2.boundingRect(contour)
    crop = gray_enh[y:y + h, x:x + w]
    if crop.size < 16:
        crop = gray_enh
    q = (crop // 8).astype(np.uint8)  # 32 grey levels
    glcm = graycomatrix(q, distances=[1, 2],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    f = {}
    for prop, name in [("contrast", "glcm_contrast"), ("correlation", "glcm_correlation"),
                       ("energy", "glcm_energy"), ("homogeneity", "glcm_homogeneity")]:
        f[name] = float(np.nan_to_num(graycoprops(glcm, prop).mean()))

    vals = gray_enh[mask > 0]
    hist = np.bincount(vals, minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    f["entropy"] = float(-(p * np.log2(p)).sum())
    return f


def _shape_features(gray_enh, mask, contour, size):
    area = float(cv2.contourArea(contour))
    perim = float(cv2.arcLength(contour, True))
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    aspect = float(max(rw, rh) / min(rw, rh)) if min(rw, rh) > 0 else 1.0
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = area / hull_area if hull_area > 0 else 1.0
    circularity = min(1.0, 4 * np.pi * area / (perim ** 2)) if perim > 0 else 0.0

    sel = mask > 0
    edges = cv2.Canny(gray_enh, 50, 150)
    edge_density = float((edges[sel] > 0).mean())
    g = gray_enh.astype(np.float32) / 255.0
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)[sel]

    return {
        "nail_area_ratio": area / float(size * size),
        "perimeter_ratio": perim / float(4 * size),
        "aspect_ratio": aspect,
        "solidity": float(solidity),
        "circularity": float(circularity),
        "edge_density": edge_density,
        "gradient_mean": float(mag.mean()),
        "gradient_std": float(mag.std()),
    }


def extract_from_image(bgr):
    """Return (feature_vector, feature_dict, stages) for a BGR image."""
    stages = preprocessing.preprocess(bgr)
    size = stages["resized"].shape[0]
    f = {}
    f.update(_colour_features(stages["denoised"], stages["mask"]))
    f.update(_texture_features(stages["enhanced_gray"], stages["mask"], stages["contour"]))
    f.update(_shape_features(stages["enhanced_gray"], stages["mask"], stages["contour"], size))
    vec = np.array([f[n] for n in FEATURE_NAMES], dtype=np.float64)
    if not np.all(np.isfinite(vec)):
        raise ValueError("Non-finite feature value produced.")
    return vec, f, stages


def extract_from_path(path):
    """Used in parallel during training. Returns (vector | None, error | None)."""
    try:
        vec, _, _ = extract_from_image(preprocessing.read_image_path(path))
        return vec, None
    except Exception as exc:  # noqa: BLE001 - report and skip bad images
        return None, str(exc)
