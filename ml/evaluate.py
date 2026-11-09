"""
Operation Ward Zero – Model Evaluation
=======================================

Runs the full evaluation pipeline on a held-out test set:
  - Regression metrics: MAE, RMSE, R²
  - Classification metrics (bucketed): Precision, Recall, F1, Confusion Matrix
  - Saves reports as JSON and human-readable text to ml/reports/
  - Generates confusion-matrix PNG plot

All metrics are computed from actual predictions – nothing is fabricated.
"""

import os
import sys
import json
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib  # type: ignore

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = pathlib.Path(__file__).resolve().parent
ARTIFACTS_DIR = _THIS_DIR / "artifacts"
REPORTS_DIR = _THIS_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "risk_model.joblib"

# Import data generator from train module
sys.path.insert(0, str(_THIS_DIR.parent))
from ml.train import generate_synthetic_data, FEATURE_NAMES

# ---------------------------------------------------------------------------
# Configurable thresholds (must match predict.py defaults)
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {"low": 0.3, "high": 0.6}
CATEGORY_LABELS = ["Low", "Medium", "High"]


def _score_to_category(score: float, thresholds: dict | None = None) -> str:
    t = thresholds or DEFAULT_THRESHOLDS
    if score < t["low"]:
        return "Low"
    elif score <= t["high"]:
        return "Medium"
    return "High"


def _scores_to_categories(scores, thresholds=None):
    return np.array([_score_to_category(s, thresholds) for s in scores])


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(cm, labels, save_path):
    """Save a confusion-matrix heatmap as PNG using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title("Confusion Matrix – Risk Categories")
        fig.colorbar(im, ax=ax)

        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(labels)

        # Cell labels
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        ax.set_ylabel("True Category")
        ax.set_xlabel("Predicted Category")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"      Confusion matrix plot -> {save_path}")
        return True
    except ImportError:
        print("      [WARN] matplotlib not available - skipping plot.")
        return False


def _plot_scatter(y_true, y_pred, save_path):
    """Predicted vs True scatter plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, edgecolors="none")
        ax.plot([0, 1], [0, 1], "r--", lw=1, label="Perfect")
        ax.set_xlabel("True Risk Score")
        ax.set_ylabel("Predicted Risk Score")
        ax.set_title("Predicted vs True - Experimental Risk Score")
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"      Scatter plot -> {save_path}")
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(data_seed: int = 42, test_size: float = 0.2,
             thresholds: dict | None = None):
    """Run full evaluation and save reports.

    Uses the same data-generation seed and split as training so the test set
    is guaranteed to be the held-out portion the model never saw.
    """
    print("=" * 60)
    print("Operation Ward Zero - Model Evaluation")
    print("=" * 60)

    # 1. Load model
    print("\n[1/5] Loading model ...")
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}. Run ml/train.py first.")
        sys.exit(1)
    model = joblib.load(MODEL_PATH)
    print(f"      Loaded from {MODEL_PATH}")

    # 2. Regenerate data with same seed & split
    print("\n[2/5] Generating test split (seed={}, test_size={}) ...".format(
        data_seed, test_size))
    X, y = generate_synthetic_data(n=6000, seed=data_seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=data_seed
    )
    print(f"      Test samples: {X_test.shape[0]}")

    # 3. Predict
    print("\n[3/5] Predicting on test set ...")
    y_pred = model.predict(X_test).clip(0, 1)

    # 4. Regression metrics
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = r2_score(y_test, y_pred)

    print(f"\n      MAE:  {mae:.6f}")
    print(f"      RMSE: {rmse:.6f}")
    print(f"      R2:   {r2:.6f}")

    # 5. Classification metrics (bucketed)
    print("\n[4/5] Computing classification metrics ...")
    y_cat_true = _scores_to_categories(y_test, thresholds)
    y_cat_pred = _scores_to_categories(y_pred, thresholds)

    # Ensure all labels present for consistent matrix
    present_labels = sorted(set(y_cat_true) | set(y_cat_pred),
                            key=lambda l: CATEGORY_LABELS.index(l))

    cm = confusion_matrix(y_cat_true, y_cat_pred, labels=present_labels)
    cls_report = classification_report(
        y_cat_true, y_cat_pred, labels=present_labels, output_dict=True
    )
    cls_report_text = classification_report(
        y_cat_true, y_cat_pred, labels=present_labels
    )

    print(f"\n{cls_report_text}")

    # 6. Save reports
    print("[5/5] Saving reports ...")

    report = {
        "disclaimer": (
            "These metrics reflect the model's ability to reproduce a synthetic "
            "rule-based risk score.  They do NOT measure clinical predictive power."
        ),
        "regression": {
            "MAE": round(mae, 6),
            "RMSE": round(rmse, 6),
            "R2": round(r2, 6),
        },
        "classification": {
            "thresholds": thresholds or DEFAULT_THRESHOLDS,
            "labels": present_labels,
            "confusion_matrix": cm.tolist(),
            "report": cls_report,
        },
        "test_samples": int(X_test.shape[0]),
        "data_seed": data_seed,
        "test_size": test_size,
    }

    json_path = REPORTS_DIR / "evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"      JSON report -> {json_path}")

    # Human-readable text report
    text_lines = [
        "Operation Ward Zero - Evaluation Report",
        "=" * 50,
        "",
        "DISCLAIMER: These metrics reflect the model's ability to reproduce",
        "a synthetic rule-based risk score. They do NOT measure clinical",
        "predictive power.",
        "",
        "--- Regression Metrics ---",
        f"  MAE:  {mae:.6f}",
        f"  RMSE: {rmse:.6f}",
        f"  R2:   {r2:.6f}",
        "",
        "--- Classification Metrics (Bucketed) ---",
        f"  Thresholds: Low < {DEFAULT_THRESHOLDS['low']}, "
        f"Med {DEFAULT_THRESHOLDS['low']}-{DEFAULT_THRESHOLDS['high']}, "
        f"High > {DEFAULT_THRESHOLDS['high']}",
        "",
        cls_report_text,
        "",
        "--- Confusion Matrix ---",
        f"  Labels: {present_labels}",
        str(cm),
        "",
        f"Test samples: {X_test.shape[0]}",
        f"Data seed: {data_seed}",
    ]
    txt_path = REPORTS_DIR / "evaluation_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))
    print(f"      Text report -> {txt_path}")

    # Plots
    cm_png = REPORTS_DIR / "confusion_matrix.png"
    _plot_confusion_matrix(cm, present_labels, cm_png)

    scatter_png = REPORTS_DIR / "pred_vs_true.png"
    _plot_scatter(y_test, y_pred, scatter_png)

    print("\n[OK] Evaluation complete.\n")
    return report


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    report = evaluate()
    print(json.dumps(report["regression"], indent=2))
