"""Metrics and charts.

`compute_metrics` is used by train_model.py. Running this file directly
(`python -m src.evaluate_model`) only REDRAWS charts from the saved results;
it never re-evaluates the model on the test set (test set is used once, in training).
"""
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)

from . import config

TEAL, BLUE, SOFT = "#0d9488", "#3b82f6", "#e6f4f3"


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(float(o)) or math.isinf(float(o))) else float(o)
    return o


def compute_metrics(y_true, y_pred) -> dict:
    """Melanoma (label 1) is the positive class."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return _clean({
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": spec,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "support": {"non_melanoma": int(tn + fp), "melanoma": int(fn + tp)},
    })


def save_json(path, obj):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(_clean(obj), indent=2))


def _save(fig, name):
    config.CHARTS_DIR.mkdir(exist_ok=True, parents=True)
    fig.tight_layout()
    fig.savefig(config.CHARTS_DIR / name, dpi=140, facecolor="white")
    plt.close(fig)


def plot_confusion_matrix(cm, name="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap=sns.light_palette(TEAL, as_cmap=True),
                cbar=False, linewidths=1, linecolor="white", annot_kws={"size": 16},
                xticklabels=["Non-Melanoma", "Melanoma"], yticklabels=["Non-Melanoma", "Melanoma"], ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix (test set)")
    _save(fig, name)


def plot_metrics_bar(test_metrics, val_metrics, name="metrics_bar.png"):
    keys = ["accuracy", "precision", "recall", "specificity", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "Specificity", "F1"]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, [val_metrics[k] or 0 for k in keys], 0.4, label="Validation", color=BLUE)
    ax.bar(x + 0.2, [test_metrics[k] or 0 for k in keys], 0.4, label="Test", color=TEAL)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Validation vs test metrics (selected model)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    sns.despine()
    _save(fig, name)


def plot_k_curve(grid: pd.DataFrame, name="k_curve.png"):
    fig, ax = plt.subplots(figsize=(7, 4))
    palette = sns.color_palette([TEAL, BLUE, "#64748b", "#f59e0b"])
    for color, (metric, g) in zip(palette, grid.groupby("metric")):
        best = g.groupby("k")["val_balanced_accuracy"].max()
        ax.plot(best.index, best.values, marker="o", ms=4, label=metric, color=color)
    ax.set_xlabel("K (number of neighbours)")
    ax.set_ylabel("Validation balanced accuracy")
    ax.set_title("Effect of K and distance metric (validation set)")
    ax.legend(frameon=False, title="Metric")
    sns.despine()
    _save(fig, name)


def plot_metric_comparison(grid: pd.DataFrame, name="metric_comparison.png"):
    best = grid.groupby("metric")[["val_balanced_accuracy", "val_f1", "val_recall"]].max()
    best = best.rename(columns={"val_balanced_accuracy": "Balanced acc.", "val_f1": "F1", "val_recall": "Recall"})
    ax = best.plot(kind="bar", figsize=(7, 4), color=[TEAL, BLUE, "#94a3b8"], width=0.75)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Best validation score")
    ax.set_xlabel("")
    ax.set_title("Distance-metric comparison (best K per metric)")
    plt.xticks(rotation=0)
    ax.legend(frameon=False)
    sns.despine()
    _save(ax.figure, name)


def plot_class_distribution(split_counts, name="class_distribution.png"):
    df = pd.DataFrame(split_counts).T[["non_melanoma", "melanoma"]]
    df.columns = ["Non-Melanoma", "Melanoma"]
    ax = df.plot(kind="bar", stacked=True, figsize=(6, 4), color=[BLUE, TEAL], width=0.6)
    ax.set_ylabel("Images")
    ax.set_title("Images per split and class")
    plt.xticks(rotation=0)
    ax.legend(frameon=False)
    sns.despine()
    _save(ax.figure, name)


def generate_reports(metrics: dict, grid: pd.DataFrame):
    plot_confusion_matrix(metrics["test"]["confusion_matrix"])
    plot_metrics_bar(metrics["test"], metrics["validation"])
    plot_k_curve(grid)
    plot_metric_comparison(grid)
    plot_class_distribution(metrics["split_counts"])


def main():
    if not config.METRICS_PATH.exists() or not config.GRID_PATH.exists():
        print("No saved results found. Run: python -m src.train_model")
        raise SystemExit(1)
    metrics = json.loads(config.METRICS_PATH.read_text())
    grid = pd.read_csv(config.GRID_PATH)
    generate_reports(metrics, grid)
    print("Charts regenerated in", config.CHARTS_DIR, "(no new test-set evaluation was done).")


if __name__ == "__main__":
    main()
