"""Train the KNN model.

    python -m src.train_model

Pipeline: audit dataset -> extract features -> 70/15/15 split (stratified, subject-aware
if dataset/subjects.csv exists) -> fit scaler on TRAIN only -> grid-search K / metric / weighting
on VALIDATION -> fit final model on TRAIN -> evaluate on TEST exactly once -> save everything.
"""
import argparse
import json
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from . import config, dataset_loader, evaluate_model
from .feature_extraction import FEATURE_NAMES, extract_from_path


def build_feature_matrix(df, jobs):
    results = Parallel(n_jobs=jobs)(delayed(extract_from_path)(p) for p in df["path"])
    keep, vecs, failed = [], [], []
    for i, (vec, err) in enumerate(results):
        if vec is None:
            failed.append((df.iloc[i]["path"], err))
        else:
            keep.append(i)
            vecs.append(vec)
    if failed:
        print(f"WARNING: feature extraction failed for {len(failed)} image(s); they were skipped.")
    df = df.iloc[keep].reset_index(drop=True)
    return np.vstack(vecs), df["label"].to_numpy(), df, failed


def split_data(y, groups, seed):
    """Return (train_idx, val_idx, test_idx). Stratified; grouped by subject when available."""
    n = len(y)
    idx = np.arange(n)
    g_series = pd.Series(groups)
    has_groups = bool(g_series.notna().all() and g_series.nunique() < n
                      and not g_series.astype(str).str.startswith("unknown_").all())
    if has_groups:
        sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
        fold_of = np.empty(n, dtype=int)
        for i, (_, te) in enumerate(sgkf.split(idx, y, groups)):
            fold_of[te] = i
        val, test = idx[fold_of < 3], idx[(fold_of >= 3) & (fold_of < 6)]
        train = idx[fold_of >= 6]
        g = np.asarray(groups)
        assert not (set(g[train]) & set(g[val]) or set(g[train]) & set(g[test]) or set(g[val]) & set(g[test])), \
            "subject leakage detected"
        mode = "stratified + subject-grouped (StratifiedGroupKFold)"
    else:
        tr_val, test = train_test_split(idx, test_size=0.15, stratify=y, random_state=seed)
        train, val = train_test_split(tr_val, test_size=0.15 / 0.85, stratify=y[tr_val], random_state=seed)
        mode = "stratified random split (no subject information available)"
    return train, val, test, mode


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers for feature extraction")
    ap.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    args = ap.parse_args(argv)

    # 1. dataset + audit
    try:
        df, audit = dataset_loader.load_clean_dataset()
    except dataset_loader.DatasetError as e:
        print("DATASET ERROR:", e)
        return 1
    print("Dataset audit:", json.dumps(audit["clean_images_per_class"]), "(after cleaning)")

    # 2-3. features
    print("Extracting features ...")
    X, y, df, failed = build_feature_matrix(df, args.jobs)
    groups = df["group"].to_numpy()

    # 5. split
    train, val, test, split_mode = split_data(y, groups, args.seed)
    for name, ix in (("train", train), ("validation", val), ("test", test)):
        if len(set(y[ix])) < 2:
            print(f"ERROR: the {name} split contains only one class ({len(ix)} images). "
                  "You need more images per class (ideally hundreds).")
            return 1
    split_counts = {n: {"non_melanoma": int((y[i] == 0).sum()), "melanoma": int((y[i] == 1).sum())}
                    for n, i in (("train", train), ("validation", val), ("test", test))}
    print("Split mode:", split_mode, split_counts)
    if min(min(c.values()) for c in split_counts.values()) < 10:
        print("WARNING: some split has fewer than 10 images of a class - metrics will be very unreliable.")

    # 4. scaling - fit on TRAIN only
    scaler = StandardScaler().fit(X[train])
    Xtr, Xva, Xte = scaler.transform(X[train]), scaler.transform(X[val]), scaler.transform(X[test])
    ytr, yva, yte = y[train], y[val], y[test]

    # 6. hyper-parameter search on VALIDATION
    rows = []
    ks = [k for k in config.K_VALUES if k <= len(ytr)]
    for metric in config.DISTANCE_METRICS:
        for weights in config.WEIGHTINGS:
            for k in ks:
                knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights).fit(Xtr, ytr)
                m = evaluate_model.compute_metrics(yva, knn.predict(Xva))
                rows.append({"k": k, "metric": metric, "weights": weights,
                             "val_accuracy": m["accuracy"], "val_balanced_accuracy": m["balanced_accuracy"],
                             "val_precision": m["precision"], "val_recall": m["recall"],
                             "val_specificity": m["specificity"], "val_f1": m["f1"]})
    grid = pd.DataFrame(rows)
    ranked = grid.sort_values(["val_balanced_accuracy", "val_recall", "k"],
                              ascending=[False, False, True]).reset_index(drop=True)
    best = ranked.iloc[0]
    best_k, best_metric, best_weights = int(best["k"]), best["metric"], best["weights"]
    print(f"Selected on validation (balanced accuracy): K={best_k}, metric={best_metric}, weights={best_weights}")

    # final model: fit on TRAIN only, evaluate TEST once
    model = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric, weights=best_weights).fit(Xtr, ytr)
    val_metrics = evaluate_model.compute_metrics(yva, model.predict(Xva))
    test_metrics = evaluate_model.compute_metrics(yte, model.predict(Xte))   # <- the only test evaluation

    majority = max((yte == 0).mean(), (yte == 1).mean())
    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "selected": {"k": best_k, "metric": best_metric, "weights": best_weights,
                     "selection_criterion": "validation balanced accuracy (ties: recall, then smaller K)"},
        "validation": val_metrics,
        "test": test_metrics,
        "test_majority_class_baseline_accuracy": float(majority),
        "split_mode": split_mode,
        "split_counts": split_counts,
        "n_features": len(FEATURE_NAMES),
        "audit": audit,
        "feature_extraction_failures": len(failed),
        "top_configs": ranked.head(10).to_dict(orient="records"),
        "test_evaluated_once": True,
    }

    # save
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    config.METADATA_PATH.write_text(json.dumps({
        "classes": {"0": "non_melanoma", "1": "melanoma"},
        "feature_names": FEATURE_NAMES,
        "img_size": config.IMG_SIZE,
        "k": best_k, "metric": best_metric, "weights": best_weights,
        "n_train": int(len(ytr)),
        "trained_at": metrics["trained_at"],
        "sklearn_version": sklearn.__version__,
    }, indent=2))
    grid.to_csv(config.GRID_PATH, index=False)
    evaluate_model.save_json(config.METRICS_PATH, metrics)
    evaluate_model.generate_reports(json.loads(config.METRICS_PATH.read_text()), grid)

    t = test_metrics
    print("\n=== TEST SET RESULTS (evaluated once) ===")
    for k in ("accuracy", "precision", "recall", "specificity", "f1"):
        print(f"{k:12s}: {t[k]:.3f}" if t[k] is not None else f"{k:12s}: n/a")
    print("confusion matrix [[TN FP] [FN TP]]:", t["confusion_matrix"])
    print(f"majority-class baseline accuracy on test: {majority:.3f}")
    print("\nSaved model, scaler, metrics and charts. Now run: python app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
