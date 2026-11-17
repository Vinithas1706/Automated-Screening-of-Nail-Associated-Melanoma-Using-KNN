"""Display-only stages: skin removal (nail cut-out), strong enhancement, edge map.

These images are for the user to SEE what happened. The KNN features are still
computed by `feature_extraction.extract_from_image` exactly as during training,
so the saved model stays valid (no retraining needed).
"""
import cv2
import numpy as np

BG = (245, 244, 240)  # light neutral background (BGR) used behind the cut-out nail


def _largest_component(mask, must_touch=None):
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    if must_touch is not None:
        hits = np.unique(lab[(must_touch > 0) & (lab > 0)])
        if len(hits):
            best = max(hits, key=lambda i: stats[i, cv2.CC_STAT_AREA])
            return np.where(lab == best, 255, 0).astype(np.uint8)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(lab == best, 255, 0).astype(np.uint8)


def nail_mask(resized_bgr, rough_mask, roi_found):
    """Refine the rough Otsu mask with GrabCut so surrounding skin is dropped.

    Returns (mask uint8 0/255, refined_ok). Falls back to the rough mask if
    GrabCut fails or gives an implausible area.
    """
    h, w = resized_bgr.shape[:2]
    total = float(h * w)
    gc = np.full((h, w), cv2.GC_BGD, np.uint8)
    if roi_found:
        k = max(3, h // 16) | 1
        core = cv2.erode(rough_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        outer = cv2.dilate(rough_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        gc[outer > 0] = cv2.GC_PR_BGD
        gc[rough_mask > 0] = cv2.GC_PR_FGD
        gc[core > 0] = cv2.GC_FGD
        seed = core
    else:  # segmentation failed: assume the nail is in the middle of the frame
        y0, y1, x0, x1 = int(h * .2), int(h * .8), int(w * .25), int(w * .75)
        gc[y0:y1, x0:x1] = cv2.GC_PR_FGD
        gc[int(h * .35):int(h * .65), int(w * .4):int(w * .6)] = cv2.GC_FGD
        seed = (gc == cv2.GC_FGD).astype(np.uint8) * 255
    try:
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(resized_bgr, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        m = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2)
        m = _largest_component(m, must_touch=seed)
        m = cv2.GaussianBlur(m, (5, 5), 0)
        m = np.where(m > 127, 255, 0).astype(np.uint8)
        frac = m.sum() / 255.0 / total
        if 0.03 <= frac <= 0.95:
            return m, True
    except cv2.error:
        pass
    return rough_mask, False


def _strong_enhance(bgr, mask):
    """Detail enhancement for viewing: CLAHE on lightness + unsharp mask + saturation boost."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 2.0)
    out = cv2.addWeighted(out, 1.6, blur, -0.6, 0)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.25, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _crop_square(img, mask, out=224, pad=0.12, bg=BG):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return cv2.resize(img, (out, out))
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = int(max(x1 - x0, y1 - y0) * (1 + pad * 2)) + 2
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    sx, sy = cx - side // 2, cy - side // 2
    canvas = np.full((side, side, 3), bg, np.uint8)
    # copy the overlapping window
    ix0, iy0 = max(sx, 0), max(sy, 0)
    ix1, iy1 = min(sx + side, img.shape[1]), min(sy + side, img.shape[0])
    canvas[iy0 - sy:iy1 - sy, ix0 - sx:ix1 - sx] = img[iy0:iy1, ix0:ix1]
    return cv2.resize(canvas, (out, out), interpolation=cv2.INTER_AREA)


def build_views(stages):
    """Return dict of BGR images for the processing page."""
    resized, denoised = stages["resized"], stages["denoised"]
    mask, ok = nail_mask(resized, stages["mask"], stages["roi_found"])

    soft = cv2.GaussianBlur(mask, (7, 7), 0).astype(np.float32)[..., None] / 255.0
    bg = np.full_like(resized, BG)
    cutout = (denoised.astype(np.float32) * soft + bg.astype(np.float32) * (1 - soft)).astype(np.uint8)

    strong = _strong_enhance(denoised, mask)
    nail_enh = (strong.astype(np.float32) * soft + bg.astype(np.float32) * (1 - soft)).astype(np.uint8)

    mask_view = resized.copy()
    tint = np.zeros_like(resized)
    tint[:] = (166, 148, 13)
    mask_view = np.where(mask[..., None] > 0, cv2.addWeighted(resized, 0.45, tint, 0.55, 0), (resized * 0.35).astype(np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask_view, cnts, -1, (255, 255, 255), 1)

    g = cv2.cvtColor(nail_enh, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 40, 110)
    edges = cv2.bitwise_and(edges, cv2.erode(mask, np.ones((3, 3), np.uint8)))
    edge_view = np.full_like(resized, (40, 33, 15))
    edge_view[edges > 0] = (200, 220, 90)
    cv2.drawContours(edge_view, cnts, -1, (120, 120, 235), 1)

    return {
        "cutout": cutout,
        "skin_mask": mask_view,
        "nail_only": _crop_square(cutout, mask),
        "nail_enhanced": _crop_square(nail_enh, mask),
        "edges": _crop_square(edge_view, mask, bg=(40, 33, 15)),
        "_refined": ok,
        "_area_pct": float(mask.sum() / 255.0 / (mask.shape[0] * mask.shape[1]) * 100),
    }
