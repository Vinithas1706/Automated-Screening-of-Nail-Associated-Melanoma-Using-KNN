"""Prediction using the trained KNN model + fitted scaler (never random or hard-coded)."""
import json
from datetime import datetime

import cv2
import joblib
import numpy as np

from . import config
from .visualize import build_views
from .feature_extraction import FEATURE_GROUPS, FEATURE_NAMES, HIST_BINS, extract_from_image


class ModelNotTrainedError(RuntimeError):
    pass


def _downscale(img, max_side=640):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def save_stage_images(original, stages, out_dir, uid):
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = stages["resized"].copy()
    cv2.drawContours(overlay, [stages["contour"]], -1, (166, 148, 13), 2)  # teal (BGR)
    files = {
        "original": _downscale(original),
        "resized": stages["resized"],
        "gray": stages["gray"],
        "enhanced": stages["enhanced"],
        "roi": overlay,
        "denoised": stages["denoised"],
    }
    views = build_views(stages)
    for k in ("cutout", "skin_mask", "nail_only", "nail_enhanced", "edges"):
        files[k] = views[k]
    names = {}
    for key, img in files.items():
        name = f"{uid}_{key}.jpg"
        cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        names[key] = name
    names["_refined"] = views["_refined"]
    names["_area_pct"] = round(views["_area_pct"], 1)
    return names


class NailScreeningModel:
    def __init__(self):
        for p in (config.MODEL_PATH, config.SCALER_PATH, config.METADATA_PATH):
            if not p.exists():
                raise ModelNotTrainedError(
                    "No trained model found. Add your dataset and run:  python -m src.train_model")
        self.model = joblib.load(config.MODEL_PATH)
        self.scaler = joblib.load(config.SCALER_PATH)
        self.meta = json.loads(config.METADATA_PATH.read_text())
        if self.meta["feature_names"] != FEATURE_NAMES or self.scaler.n_features_in_ != len(FEATURE_NAMES):
            raise ModelNotTrainedError(
                "The saved model was trained with a different feature set. Re-run: python -m src.train_model")

    def predict(self, bgr, uid, out_dir=config.UPLOAD_DIR):
        vec, fdict, stages = extract_from_image(bgr)
        x = self.scaler.transform(vec.reshape(1, -1))
        label_id = int(self.model.predict(x)[0])
        proba = self.model.predict_proba(x)[0]
        classes = list(self.model.classes_)
        p_mel = float(proba[classes.index(1)])
        p_non = float(proba[classes.index(0)])
        dist = float(self.model.kneighbors(x)[0].mean())

        grouped = {g: {n: fdict[n] for n in names if not n.startswith("hist_")}
                   for g, names in FEATURE_GROUPS.items()}
        hist = {c: [fdict[f"hist_{c}_{i}"] for i in range(HIST_BINS)] for c in "RGB"}
        result = {
            "uid": uid,
            "created": datetime.now().isoformat(timespec="seconds"),
            "label_id": label_id,
            "label": config.CLASS_NAMES[label_id],
            "headline": ("Melanoma-labelled pattern detected." if label_id == 1
                         else "Non-melanoma-labelled pattern detected."),
            "prob_melanoma": p_mel,
            "prob_non_melanoma": p_non,
            "n_features": len(FEATURE_NAMES),
            "k": self.meta["k"],
            "metric": self.meta["metric"],
            "weights": self.meta["weights"],
            "mean_neighbour_distance": dist,
            "roi_found": bool(stages["roi_found"]),
            "features": grouped,
            "hist": hist,
            "images": save_stage_images(bgr, stages, out_dir, uid),
        }
        return result
