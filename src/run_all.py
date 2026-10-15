"""Initial skeleton. Run examples with: python -m src.run_all --demo.
Synthetic examples are not final project results.
"""
from __future__ import annotations
import argparse
from . import evaluate


def regression_metrics(y_true, y_pred):
    """Both arrays must be on the same scale: price or log-price."""
    return {
        "RMSE": evaluate.rmse(y_true, y_pred),
        "MAE": evaluate.mae(y_true, y_pred),
        "R2": evaluate.r2(y_true, y_pred),
    }


def classification_metrics(y_true, y_pred, scores):
    precision, recall, f1 = evaluate.precision_recall_f1(y_true, y_pred)
    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": evaluate.roc_auc(y_true, scores),
    }


def demo():
    print("SYNTHETIC EXAMPLES ONLY: these are not trained-model results.")
    evaluate.compare({"Regression example": regression_metrics([10, 20, 30], [12, 18, 33])})
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    evaluate.compare({"Classification example": classification_metrics(y_true, y_pred, scores)})
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(evaluate.confusion_matrix(y_true, y_pred))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Run synthetic metric examples.")
    args = parser.parse_args()
    if args.demo:
        demo()
        return
    # Team integration order:
    # 1. Load/clean; split once; fit preprocessing on training only.
    # 2. Derive tier threshold from training prices; apply to val/test.
    # 3. Tune using validation; train matched sklearn baselines.
    # 4. Obtain test predictions and scores in unchanged sample order.
    # 5. Compute metrics, CV uncertainty, timing and error analysis.
    # 6. Save labelled report figures and print comparison tables.
    parser.exit(status=2, message="Full pipeline pending team data/model integration. Use --demo for examples.")


if __name__ == "__main__":
    main()
