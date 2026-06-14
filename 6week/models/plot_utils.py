"""
Shared result-chart helpers for the model trainers.

Writes two PNGs into the trainer's --output-dir:
  - confusion_matrix.png  (report Fig 6.5)
  - metrics_bar.png       (report Fig 6.4 — accuracy / F1 / precision / recall)

The model API route uploads every PNG it finds in the output dir to the
`models` bucket, so these surface in the app automatically. All failures are
swallowed so charting can never break a training run.
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np


def save_classification_charts(y_true, y_pred, output_dir, model_name="model", metrics=None):
    """Save confusion-matrix and metrics-bar charts. Returns list of paths written."""
    os.makedirs(output_dir, exist_ok=True)
    y_true = list(y_true)
    y_pred = list(y_pred)
    saved = []

    # --- Confusion matrix (Fig 6.5) ---
    try:
        from sklearn.metrics import confusion_matrix
        labels = sorted(set(y_true) | set(y_pred), key=lambda v: str(v))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([str(l) for l in labels], rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([str(l) for l in labels])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {model_name}")
        thresh = cm.max() / 2 if cm.size and cm.max() else 0
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        path = os.path.join(output_dir, "confusion_matrix.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        saved.append(path)
    except Exception as e:
        print(f"[plot_utils] confusion matrix skipped: {e}")

    # --- Metrics bar (Fig 6.4) ---
    try:
        metrics = dict(metrics or {})
        if not isinstance(metrics.get("accuracy"), (int, float)):
            from sklearn.metrics import accuracy_score
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        if not isinstance(metrics.get("f1_score"), (int, float)):
            from sklearn.metrics import f1_score
            metrics["f1_score"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        order = ["accuracy", "f1_score", "precision", "recall"]
        keys = [k for k in order if isinstance(metrics.get(k), (int, float))]
        vals = [float(metrics[k]) for k in keys]
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = ["#4F79E0", "#22A06B", "#FFA94D", "#7B61FF"]
        bars = ax.bar([k.replace("_", " ").title() for k in keys], vals, color=colors[:len(keys)])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("score")
        ax.set_title(f"Performance Metrics - {model_name}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        path = os.path.join(output_dir, "metrics_bar.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        saved.append(path)
    except Exception as e:
        print(f"[plot_utils] metrics bar skipped: {e}")

    return saved
