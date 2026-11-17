"""Dataset loading and auditing.

Expected layout:
    dataset/melanoma/*.jpg|jpeg|png
    dataset/non_melanoma/*.jpg|jpeg|png
Optional subject info (prevents the same patient appearing in several splits):
    dataset/subjects.csv   with columns:  filename,subject_id
    ("filename" may be "melanoma/img1.jpg" or just "img1.jpg")

Run an audit:   python -m src.dataset_loader
Nothing here is invented: every number comes from the files actually found.
"""
import hashlib
import json

import cv2
import numpy as np
import pandas as pd

from . import config, preprocessing


class DatasetError(RuntimeError):
    pass


def scan_dataset(dataset_dir=config.DATASET_DIR) -> pd.DataFrame:
    rows, unsupported = [], []
    for folder, label in config.CLASS_FOLDERS.items():
        d = dataset_dir / folder
        if not d.is_dir():
            raise DatasetError(f"Missing folder: {d}\nCreate dataset/melanoma and dataset/non_melanoma and put images inside.")
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.name == ".gitkeep":
                continue
            if p.suffix.lower() in config.VALID_EXTENSIONS:
                rows.append({"path": str(p), "rel": f"{folder}/{p.relative_to(d).as_posix()}",
                             "filename": p.name, "class_name": folder, "label": label})
            else:
                unsupported.append(str(p))
    df = pd.DataFrame(rows, columns=["path", "rel", "filename", "class_name", "label"])
    df.attrs["unsupported"] = unsupported
    return df


def _dhash(img_bgr, size=16) -> str:  # 16x16 -> 256-bit hash; strict to avoid merging merely similar photos
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return "".join("1" if b else "0" for b in bits)


def _attach_subjects(df):
    df["group"] = None
    if not config.SUBJECTS_CSV.exists():
        return df, {"subjects_file": False}
    sub = pd.read_csv(config.SUBJECTS_CSV, dtype=str)
    if not {"filename", "subject_id"} <= set(sub.columns):
        raise DatasetError("subjects.csv must have columns: filename,subject_id")
    by_rel = dict(zip(sub["filename"].str.replace("\\", "/", regex=False), sub["subject_id"]))
    by_name = dict(zip(sub["filename"].map(lambda s: s.replace("\\", "/").split("/")[-1]), sub["subject_id"]))
    df["group"] = [by_rel.get(r, by_name.get(n)) for r, n in zip(df["rel"], df["filename"])]
    missing = int(df["group"].isna().sum())
    # Images without subject info each become their own group.
    df.loc[df["group"].isna(), "group"] = ["unknown_" + str(i) for i in df.index[df["group"].isna()]]
    return df, {"subjects_file": True, "images_without_subject_id": missing,
                "n_subjects": int(df["group"].nunique())}


def audit_dataset(df: pd.DataFrame, save=True):
    """Check readability and duplicates. Returns (clean_df, report dict)."""
    found = df["class_name"].value_counts().to_dict()
    report = {
        "images_found_per_class": {k: int(found.get(k, 0)) for k in config.CLASS_FOLDERS},
        "unsupported_files_ignored": len(df.attrs.get("unsupported", [])),
    }

    unreadable, md5s, dhashes, too_small = [], [], [], []
    for p in df["path"]:
        try:
            data = open(p, "rb").read()
            img = preprocessing.read_image_bytes(data)
            if min(img.shape[:2]) < config.MIN_IMAGE_SIDE:
                too_small.append(p)
            md5s.append(hashlib.md5(data).hexdigest())
            dhashes.append(_dhash(img))
        except Exception:  # noqa: BLE001
            unreadable.append(p)
            md5s.append(None)
            dhashes.append(None)
    df = df.copy()
    df["md5"], df["dhash"] = md5s, dhashes

    bad = df["md5"].isna() | df["path"].isin(too_small)
    report["corrupted_or_unreadable"] = int(df["md5"].isna().sum())
    report["too_small_images"] = len(too_small)
    report["unreadable_files"] = unreadable[:50]
    df = df[~bad].copy()

    # Same image content (or visually identical) under different labels -> ambiguous, drop all copies.
    df["dup_key"] = df["dhash"]
    conflict_keys = df.groupby("dup_key")["label"].nunique()
    conflict_keys = set(conflict_keys[conflict_keys > 1].index)
    conflicts = df["dup_key"].isin(conflict_keys)
    report["label_conflict_images_removed"] = int(conflicts.sum())
    df = df[~conflicts]

    exact = df.duplicated("md5", keep="first")
    report["exact_duplicates_removed"] = int(exact.sum())
    df = df[~exact]
    perceptual = df.duplicated("dhash", keep="first")
    report["near_duplicates_removed"] = int(perceptual.sum())
    df = df[~perceptual].copy()

    df, subj = _attach_subjects(df)
    report["subject_info"] = subj
    final = df["class_name"].value_counts().to_dict()
    report["clean_images_per_class"] = {k: int(final.get(k, 0)) for k in config.CLASS_FOLDERS}
    report["clean_total"] = int(len(df))
    n0, n1 = report["clean_images_per_class"]["non_melanoma"], report["clean_images_per_class"]["melanoma"]
    report["class_imbalance_ratio"] = round(max(n0, n1) / max(min(n0, n1), 1), 2) if (n0 and n1) else None
    report["note"] = ("Near-duplicates are detected with a perceptual (difference) hash; identical hash "
                      "= treated as a duplicate. Hash collisions on very similar images are possible.")

    df = df.drop(columns=["dup_key"]).reset_index(drop=True)
    if save:
        config.REPORTS_DIR.mkdir(exist_ok=True)
        config.AUDIT_PATH.write_text(json.dumps(report, indent=2))
        df.drop(columns=["md5", "dhash"]).to_csv(config.CLEAN_LIST_PATH, index=False)
    return df, report


def load_clean_dataset(save_reports=True):
    df = scan_dataset()
    if df.empty:
        raise DatasetError("No images found in dataset/melanoma or dataset/non_melanoma.\n"
                           "Add your labelled nail images (JPG/JPEG/PNG) and run again.")
    clean, report = audit_dataset(df, save=save_reports)
    for k, v in report["clean_images_per_class"].items():
        if v == 0:
            raise DatasetError(f"After cleaning, class '{k}' has 0 images - cannot train a two-class model.")
    return clean, report


def main():
    try:
        _, report = load_clean_dataset()
    except DatasetError as e:
        print("DATASET ERROR:", e)
        raise SystemExit(1)
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {config.AUDIT_PATH}\nSaved: {config.CLEAN_LIST_PATH}")


if __name__ == "__main__":
    main()
