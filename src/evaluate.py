"""NumPy metrics: premium=1, standard=0. No sklearn dependency."""
from __future__ import annotations
import numpy as np


def _paired_arrays(y_true, values):
    """Validate inputs to prevent silent broadcasting or invalid results."""
    y_true = np.asarray(y_true, dtype=float)
    values = np.asarray(values, dtype=float)
    if y_true.ndim != 1 or values.ndim != 1:
        raise ValueError("Inputs must be one-dimensional arrays.")
    if y_true.size == 0 or y_true.shape != values.shape:
        raise ValueError("Inputs must be non-empty and have the same shape.")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(values)):
        raise ValueError("Inputs must contain only finite numbers.")
    return y_true, values


def _binary_labels(labels):
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("Labels must be 0 (standard) or 1 (premium).")


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _paired_arrays(y_true, y_pred)
    errors = y_true - y_pred
    return float(np.sqrt(np.mean(errors ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _paired_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    """One sample: NaN. Constant target: 1 if perfect, otherwise 0."""
    y_true, y_pred = _paired_arrays(y_true, y_pred)
    if y_true.size < 2:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)


def confusion_matrix(y_true, y_pred):
    """Rows=true, columns=predicted: [[TN, FP], [FN, TP]]."""
    y_true, y_pred = _paired_arrays(y_true, y_pred)
    _binary_labels(y_true)
    _binary_labels(y_pred)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def precision_recall_f1(y_true, y_pred):
    """Positive-class metrics; undefined divisions return 0.0."""
    matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = matrix.ravel()
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    denominator = precision + recall
    f1 = float(2 * precision * recall / denominator) if denominator else 0.0
    return precision, recall, f1


def roc_auc(y_true, scores) -> float:
    """Average-rank AUC. Larger scores mean premium; one class gives NaN.

    Pass SVM margins or premium probabilities, rather than predicted labels.
    """
    y_true, scores = _paired_arrays(y_true, scores)
    _binary_labels(y_true)
    positive = y_true == 1
    n_positive = int(np.sum(positive))
    n_negative = y_true.size - n_positive
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=float)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum = np.sum(ranks[positive])
    wins = rank_sum - n_positive * (n_positive + 1) / 2
    return float(wins / (n_positive * n_negative))


def compare(name_to_metrics: dict) -> None:
    """Print {model_name: {scalar_metric: value}}; missing values are n/a.

    Print confusion matrices separately. Use identical splits for comparisons.
    """
    if not name_to_metrics:
        print("No model results available.")
        return
    metric_names = list(dict.fromkeys(key for metrics in name_to_metrics.values() for key in metrics))
    rows = []
    for name, metrics in name_to_metrics.items():
        row = [str(name)]
        for metric in metric_names:
            value = metrics.get(metric)
            row.append("n/a" if value is None else f"{float(value):.4f}")
        rows.append(row)
    headers = ["Model", *metric_names]
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    def format_row(row):
        return " | ".join(cell.ljust(width) for cell, width in zip(row, widths))
    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))
